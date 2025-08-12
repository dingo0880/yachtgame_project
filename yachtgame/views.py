import json
import numpy as np
import joblib
from pathlib import Path
from collections import Counter
from django.conf import settings
from pathlib import Path
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt  # 프론트에서 CSRF 안 넣으면 임시로 사용
from django.shortcuts import render
# === 설정 ===
CATEGORIES = [
    "Ones","Twos","Threes","Fours","Fives","Sixes",
    "Four of a Kind","Full House","Small Straight","Large Straight","Yahtzee","Chance"
]
UPPER = ["Ones","Twos","Threes","Fours","Fives","Sixes"]
MODEL_DIR = Path(settings.BASE_DIR) / "yachtgame_project" / "ml_models"
def index(request):
    # templates/yachtgame/index.html 렌더
    return render(request, "yachtgame/index.html")
# === 모델 로더 (lazy singleton) ===
class _ML:
    loaded = False
    keep_model = None
    keep_mask = None
    cat_model = None
    cat_mask = None
    cat_classes = None

    @classmethod
    def load(cls):
        if cls.loaded:
            return
        try:
            kp = MODEL_DIR / "keep_model.pkl"
            km = MODEL_DIR / "keep_feature_mask.pkl"
            cp = MODEL_DIR / "cat_model.pkl"
            cm = MODEL_DIR / "cat_feature_mask.pkl"
            cj = MODEL_DIR / "cat_label_classes.json"

            if kp.exists() and km.exists():
                cls.keep_model = joblib.load(kp)
                cls.keep_mask  = joblib.load(km)
            if cp.exists() and cm.exists() and cj.exists():
                cls.cat_model  = joblib.load(cp)
                cls.cat_mask   = joblib.load(cm)
                cls.cat_classes = json.loads(cj.read_text(encoding="utf-8"))
            cls.loaded = True
        except Exception as e:
            cls.loaded = True
            print("[ML][load] error:", e)

# === 유틸 (학습과 동일한 피처 순서) ===
def parse_board(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(str(s).replace("'", '"'))
    except Exception:
        return {k: None for k in CATEGORIES}

def parse_dice5(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = [int(x) for x in v]
    else:
        arr = [int(x) for x in str(v).split(",") if str(x).strip() != ""]
    return arr if len(arr)==5 and all(1<=a<=6 for a in arr) else []

def dice_counts_vec(d):
    out=[0]*6
    for x in d:
        if 1<=x<=6: out[x-1]+=1
    return out

def is_small_straight(d):
    s=set(d)
    return ({1,2,3,4}<=s) or ({2,3,4,5}<=s) or ({3,4,5,6}<=s)

def is_large_straight(d):
    s=set(d)
    return s=={1,2,3,4,5} or s=={2,3,4,5,6}

def immediate_scores_dict(d):
    cnt = dice_counts_vec(d)
    def up(face): return sum(1 for x in d if x==face)*face
    return {
        "Ones": up(1),
        "Twos": up(2),
        "Threes": up(3),
        "Fours": up(4),
        "Fives": up(5),
        "Sixes": up(6),
        "Four of a Kind": sum(d) if max(cnt)>=4 else 0,
        "Full House": 25 if (3 in cnt and 2 in cnt) else 0,
        "Small Straight": 15 if is_small_straight(d) else 0,
        "Large Straight": 30 if is_large_straight(d) else 0,
        "Yahtzee": 50 if 5 in cnt else 0,
        "Chance": sum(d),
    }

def immediate_scores_vector(d):
    s = immediate_scores_dict(d)
    return [s[c] for c in CATEGORIES]

def board_features(board):
    upper_sub = sum(v for k,v in board.items() if k in UPPER and isinstance(v,int))
    need_bonus = max(0, 63-upper_sub)
    filled = sum(1 for v in board.values() if v is not None)
    turns_left = len(CATEGORIES)-filled
    avail_mask = [0 if isinstance(board.get(c), int) else 1 for c in CATEGORIES]
    return [upper_sub, need_bonus, len(CATEGORIES)-filled, turns_left], avail_mask

def build_keep_feature_row(turn, roll_idx, board, dice):
    svec, amask = board_features(board)
    cnt = dice_counts_vec(dice)
    feats = [
        int(turn), int(roll_idx),
        *svec, *amask,
        *cnt, int(2 in cnt), int(3 in cnt), int(4 in cnt), int(5 in cnt),
        int(is_small_straight(dice)), int(is_large_straight(dice)),
        *immediate_scores_vector(dice)
    ]
    return np.array(feats, dtype=float).reshape(1, -1)

def build_cat_feature_row(turn, board, dice):
    svec, amask = board_features(board)
    cnt = dice_counts_vec(dice)
    feats = [
        int(turn),
        *svec, *amask,
        *cnt, int(2 in cnt), int(3 in cnt), int(4 in cnt), int(5 in cnt),
        int(is_small_straight(dice)), int(is_large_straight(dice)),
        *immediate_scores_vector(dice)
    ]
    return np.array(feats, dtype=float).reshape(1, -1)

def apply_mask(X, mask):
    mask = np.asarray(mask, bool)
    X = X[:, mask]
    # 학습시 sanitize_X에서 NaN/Inf -> 0 처리됨
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] == 0:
        X = np.zeros((X.shape[0], 1), dtype=float)
    return X

# === 헬스 체크 ===
def ml_health(request):
    _ML.load()
    ok = (_ML.keep_model is not None and _ML.keep_mask is not None and
          _ML.cat_model  is not None and _ML.cat_mask  is not None and
          _ML.cat_classes is not None)
    return JsonResponse({"loaded": ok})

# === KEEP 결정 ===
@csrf_exempt
@require_POST
def ml_keep_decision(request):
    _ML.load()
    if _ML.keep_model is None or _ML.keep_mask is None:
        return JsonResponse({"error":"model_not_ready"}, status=503)

    try:
        body = json.loads(request.body.decode("utf-8"))
        turn = body.get("turn")
        roll_idx = body.get("roll_idx")   # 1 or 2
        board = parse_board(body.get("board"))
        dice = parse_dice5(body.get("dice"))
        if turn is None or roll_idx not in [1,2] or len(dice) != 5:
            return HttpResponseBadRequest("invalid payload")
    except Exception as e:
        return HttpResponseBadRequest(str(e))

    X = build_keep_feature_row(turn, roll_idx, board, dice)
    X = apply_mask(X, _ML.keep_mask)
    pred = _ML.keep_model.predict(X).astype(int).tolist()[0]  # [0/1]*5
    return JsonResponse({"keep_mask": pred})

# === 카테고리 선택 ===
@csrf_exempt
@require_POST
def ml_category_decision(request):
    _ML.load()
    if _ML.cat_model is None or _ML.cat_mask is None or _ML.cat_classes is None:
        return JsonResponse({"error":"model_not_ready"}, status=503)

    try:
        body = json.loads(request.body.decode("utf-8"))
        turn = body.get("turn")
        board = parse_board(body.get("board"))
        dice = parse_dice5(body.get("dice"))
        if turn is None or len(dice) != 5:
            return HttpResponseBadRequest("invalid payload")
    except Exception as e:
        return HttpResponseBadRequest(str(e))

    X = build_cat_feature_row(turn, board, dice)
    X = apply_mask(X, _ML.cat_mask)
    y = _ML.cat_model.predict(X)[0]
    # y가 정수 label이면 classes에서 매핑
    try:
        cat = _ML.cat_classes[int(y)]
    except Exception:
        cat = str(y)
    return JsonResponse({"category": cat})
