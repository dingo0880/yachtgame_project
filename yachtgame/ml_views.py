# yachtgame/ml_views.py
import json
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

# ------------------ pickle 호환용 래퍼 ------------------
class MultiOutputWithFallback(BaseEstimator, ClassifierMixin):
    """
    슬롯별로:
      - 학습 라벨이 단일 클래스면 DummyClassifier(most_frequent)
      - 아니면 base_estimator(deepcopy) 사용
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.estimators_ = None

    def fit(self, X, Y):
        Y = np.asarray(Y)
        self.estimators_ = []
        for k in range(Y.shape[1]):
            yk = Y[:, k]
            uniq = np.unique(yk)
            if len(uniq) < 2:
                est = DummyClassifier(strategy="most_frequent")
                est.fit(X, yk)
            else:
                est = deepcopy(self.base_estimator)
                est.fit(X, yk)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        cols = [est.predict(X).reshape(-1, 1) for est in self.estimators_]
        return np.hstack(cols)

# ★ pickled 객체가 __main__.MultiOutputWithFallback 를 찾도록 패치
sys.modules['__main__'].MultiOutputWithFallback = MultiOutputWithFallback

# ------------------ 상수/경로 ------------------
CATEGORIES = [
    "Ones","Twos","Threes","Fours","Fives","Sixes",
    "Four of a Kind","Full House","Small Straight","Large Straight","Yahtzee","Chance"
]
UPPER = ["Ones","Twos","Threes","Fours","Fives","Sixes"]

# 모델 파일이 있는 실제 위치에 맞춰 지정
# /home/ubuntu/yachtgame_project/yachtgame_project/ml_models
MODEL_DIR = Path(settings.BASE_DIR) / "yachtgame_project" / "ml_models"

# ------------------ 모델 로더 ------------------
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

            if kp.exists():
                cls.keep_model = joblib.load(kp)
            if km.exists():
                cls.keep_mask  = joblib.load(km)
            if cp.exists():
                cls.cat_model  = joblib.load(cp)
            if cm.exists():
                cls.cat_mask   = joblib.load(cm)
            if cj.exists():
                cls.cat_classes = json.loads(cj.read_text(encoding="utf-8"))

            # ✅ 배열/리스트는 길이로, 객체는 None 체크로 판단
            def _is_good_mask(m):
                try:
                    import numpy as _np
                    return m is not None and len(_np.atleast_1d(m)) > 0
                except Exception:
                    return False

            cls.loaded = True

            # (선택) 로드 결과 간단 로그
            # print("[ML][load] keep_model:", cls.keep_model is not None,
            #       "keep_mask:", _is_good_mask(cls.keep_mask),
            #       "cat_model:", cls.cat_model is not None,
            #       "cat_mask:", _is_good_mask(cls.cat_mask),
            #       "cat_classes:", bool(cls.cat_classes))

        except Exception as e:
            cls.loaded = True   # 다음에 다시 들어와도 재시도 않게(원래 로직 유지)
            print("[ML][load] error:", e)


# ------------------ 유틸 ------------------
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
    return arr if len(arr) == 5 and all(1 <= a <= 6 for a in arr) else []

def dice_counts_vec(d):
    out = [0] * 6
    for x in d:
        if 1 <= x <= 6:
            out[x - 1] += 1
    return out

def is_small_straight(d):
    s = set(d)
    return ({1,2,3,4} <= s) or ({2,3,4,5} <= s) or ({3,4,5,6} <= s)

def is_large_straight(d):
    s = set(d)
    return s == {1,2,3,4,5} or s == {2,3,4,5,6}

def immediate_scores_dict(d):
    cnt = dice_counts_vec(d)
    def up(face): return sum(1 for x in d if x == face) * face
    return {
        "Ones": up(1),
        "Twos": up(2),
        "Threes": up(3),
        "Fours": up(4),
        "Fives": up(5),
        "Sixes": up(6),
        "Four of a Kind": sum(d) if max(cnt) >= 4 else 0,
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
    upper_sub = sum(v for k, v in board.items() if k in UPPER and isinstance(v, int))
    need_bonus = max(0, 63 - upper_sub)
    filled = sum(1 for v in board.values() if v is not None)
    turns_left = len(CATEGORIES) - filled
    avail_mask = [0 if isinstance(board.get(c), int) else 1 for c in CATEGORIES]
    return [upper_sub, need_bonus, len(CATEGORIES) - filled, turns_left], avail_mask

def build_keep_feature_row(turn, roll_idx, board, dice):
    svec, amask = board_features(board)
    cnt = dice_counts_vec(dice)
    feats = [
        int(turn), int(roll_idx),
        *svec, *amask,
        *cnt, int(2 in cnt), int(3 in cnt), int(4 in cnt), int(5 in cnt),
        int(is_small_straight(dice)), int(is_large_straight(dice)),
        *immediate_scores_vector(dice),
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
        *immediate_scores_vector(dice),
    ]
    return np.array(feats, dtype=float).reshape(1, -1)

def apply_mask(X, mask):
    mask = np.asarray(mask, bool)
    X = X[:, mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] == 0:
        X = np.zeros((X.shape[0], 1), dtype=float)
    return X

# ------------------ 헬스체크 ------------------
def ml_health(request):
    _ML.load()

    # ✅ 여기서도 배열을 직접 and로 쓰지 말고 안전 체크
    import numpy as _np

    def _is_good_mask(m):
        try:
            return m is not None and len(_np.atleast_1d(m)) > 0
        except Exception:
            return False

    ok = (
        (_ML.keep_model is not None) and
        _is_good_mask(_ML.keep_mask)  and
        (_ML.cat_model  is not None) and
        _is_good_mask(_ML.cat_mask)   and
        (bool(_ML.cat_classes))
    )

    return JsonResponse({"loaded": ok})

# ------------------ API ------------------
@csrf_exempt
@require_POST
def ml_keep_decision(request):
    _ML.load()
    if _ML.keep_model is None or _ML.keep_mask is None:
        return JsonResponse({"error": "model_not_ready"}, status=503)
    try:
        body = json.loads(request.body.decode("utf-8"))
        turn = body.get("turn")
        roll_idx = body.get("roll_idx")   # 1 or 2
        board = parse_board(body.get("board"))
        dice = parse_dice5(body.get("dice"))
        if turn is None or roll_idx not in [1, 2] or len(dice) != 5:
            return HttpResponseBadRequest("invalid payload")
    except Exception as e:
        return HttpResponseBadRequest(str(e))

    X = build_keep_feature_row(turn, roll_idx, board, dice)
    X = apply_mask(X, _ML.keep_mask)
    pred = _ML.keep_model.predict(X).astype(int).tolist()[0]  # [0/1]*5
    return JsonResponse({"keep_mask": pred})

@csrf_exempt
@require_POST
def ml_category_decision(request):
    _ML.load()
    if _ML.cat_model is None or _ML.cat_mask is None or _ML.cat_classes is None:
        return JsonResponse({"error": "model_not_ready"}, status=503)
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
    try:
        cat = _ML.cat_classes[int(y)]
    except Exception:
        cat = str(y)
    return JsonResponse({"category": cat})
