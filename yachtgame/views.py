# yachtgame/views.py

import csv
import itertools
import json
import random
import statistics
import uuid
import hashlib
import requests
import os
import threading
from copy import deepcopy
from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone

from django.conf import settings
from django.db import transaction
from django.http import (
    JsonResponse,
    StreamingHttpResponse,
    HttpResponseBadRequest,
    HttpResponseForbidden,
    HttpResponseNotFound
)
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .models import GameSession, TurnLog

# -------------------- 상수 --------------------
CATEGORIES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "Three of a Kind", "Four of a Kind", "Full House", "Small Straight",
    "Large Straight", "Yahtzee", "Chance"
]
CPU_TYPES = ["엘리트형", "도박형", "공격형", "안정형", "일반형", "ML형"]
BASE_WEIGHTS = {
    "Ones": 0.3, "Twos": 0.4, "Threes": 0.6, "Fours": 0.8, "Fives": 1.0, "Sixes": 1.2,
    "Three of a Kind": 1.5, "Four of a Kind": 1.8, "Full House": 2.0,
    "Small Straight": 1.1, "Large Straight": 1.6, "Yahtzee": 3.0, "Chance": 1.0
}
DEV_PASSWORD = "Split5234" # 요청에 따라 하드코딩

CPU_LOG_FILE_PATH = os.path.join(settings.MEDIA_ROOT, 'cpu_turn_logs.csv')
CPU_LOG_HEADERS = [
    "id", "game_id", "player_name", "turn_number",
    "score_state_before",
    "dice_roll_1", "kept_after_roll_1", "dice_roll_2", "kept_after_roll_2",
    "final_dice_state", "chosen_category", "score_obtained", "created_at"
]
csv_writer_lock = threading.Lock()
cpu_log_id_counter = 0
cpu_log_id_lock = threading.Lock()

# -------------------- CPU 로그 기록 함수 --------------------
def log_cpu_turn_to_csv(log_data):
    """CPU 턴 로그를 CSV 파일에 기록합니다."""
    global cpu_log_id_counter
    with csv_writer_lock:
        file_exists = os.path.isfile(CPU_LOG_FILE_PATH)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        header_mismatch = False
        if file_exists and os.path.getsize(CPU_LOG_FILE_PATH) > 0:
            try:
                with open(CPU_LOG_FILE_PATH, 'r', encoding='utf-8', newline='') as rf:
                    r = csv.reader(rf)
                    existing_header = next(r, [])
                if existing_header != CPU_LOG_HEADERS:
                    header_mismatch = True
            except Exception:
                header_mismatch = True

        if header_mismatch:
            backup = CPU_LOG_FILE_PATH.replace(
                '.csv',
                f'_{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.bak.csv'
            )
            try:
                os.rename(CPU_LOG_FILE_PATH, backup)
            except OSError:
                pass
            file_exists = False
            with cpu_log_id_lock:
                cpu_log_id_counter = 0

        with cpu_log_id_lock:
            if cpu_log_id_counter == 0 and file_exists and os.path.getsize(CPU_LOG_FILE_PATH) > 0:
                try:
                    with open(CPU_LOG_FILE_PATH, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        last_id = 0
                        for row in reader:
                            try:
                                last_id = int(row.get('id', 0))
                            except (ValueError, TypeError):
                                continue
                        cpu_log_id_counter = last_id
                except Exception:
                    cpu_log_id_counter = 0

            cpu_log_id_counter += 1
            log_data['id'] = cpu_log_id_counter

        log_to_write = {key: log_data.get(key, "") for key in CPU_LOG_HEADERS}

        with open(CPU_LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CPU_LOG_HEADERS)
            if not file_exists or os.path.getsize(CPU_LOG_FILE_PATH) == 0:
                writer.writeheader()
            writer.writerow(log_to_write)

# -------------------- 점수 계산 --------------------
def score_category(dice, category):
    counts = Counter(dice)
    dice_set = set(dice)
    values = list(counts.values())
    sum_dice = sum(dice)

    if category == "Ones": return counts.get(1, 0) * 1
    elif category == "Twos": return counts.get(2, 0) * 2
    elif category == "Threes": return counts.get(3, 0) * 3
    elif category == "Fours": return counts.get(4, 0) * 4
    elif category == "Fives": return counts.get(5, 0) * 5
    elif category == "Sixes": return counts.get(6, 0) * 6
    elif category == "Three of a Kind": return sum_dice if any(v >= 3 for v in values) else 0
    elif category == "Four of a Kind": return sum_dice if any(v >= 4 for v in values) else 0
    elif category == "Full House": return 25 if sorted(values) == [2, 3] else 0
    elif category == "Small Straight":
        uniq = "".join(map(str, sorted(list(dice_set))))
        return 15 if ("1234" in uniq or "2345" in uniq or "3456" in uniq) else 0
    elif category == "Large Straight":
        uniq = sorted(list(dice_set))
        return 30 if len(uniq) == 5 and uniq[-1] - uniq[0] == 4 else 0
    elif category == "Yahtzee": return 50 if 5 in values else 0
    elif category == "Chance": return sum_dice
    return 0

def calculate_upper_score(scoreboard):
    upper = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes"]
    return sum(scoreboard.get(cat, 0) for cat in upper if scoreboard.get(cat) is not None)

def calculate_bonus(upper_score):
    return 35 if upper_score >= 63 else 0

# -------------------- CPU 의사결정 전략 (리팩토링) --------------------
def dynamic_weights_elite(turn, scoreboard):
    w = deepcopy(BASE_WEIGHTS)
    upper_score = calculate_upper_score(scoreboard)
    upper_categories_left = [c for c in CATEGORIES[:6] if scoreboard.get(c) is None]
    if upper_score < 63 and upper_categories_left:
        urgency_factor = 1.0 + ((12 - turn) / 20.0)
        for cat in upper_categories_left:
            w[cat] *= (1.5 * urgency_factor)
    if turn >= 8 or upper_score >= 63:
        for cat in ("Yahtzee", "Full House", "Large Straight", "Four of a Kind"):
            if scoreboard.get(cat) is None:
                w[cat] *= 1.5
    return w

def cpu_select_category_elite(dice, scoreboard, turn):
    possible = [c for c in CATEGORIES if scoreboard.get(c) is None]
    if not possible: return "Chance"
    scores = {cat: score_category(dice, cat) for cat in possible}
    for cat in ["Yahtzee", "Large Straight", "Full House"]:
        if cat in scores and scores[cat] > 0:
            return cat
    if "Small Straight" in scores and scores["Small Straight"] > 0 and turn <= 8:
        return "Small Straight"
    w = dynamic_weights_elite(turn, scoreboard)
    weighted = [(cat, scores[cat] * w.get(cat, 1.0)) for cat in possible]
    best, _ = max(weighted, key=lambda x: x[1])
    if scores[best] == 0 and turn < 12:
        non_zero = [item for item in weighted if scores[item[0]] > 0]
        if non_zero:
            return max(non_zero, key=lambda x: x[1])[0]
        for sac in ["Yahtzee", "Ones", "Twos", "Chance"]:
            if scoreboard.get(sac) is None:
                return sac
    return best

def cpu_select_category_normal(dice, scoreboard, turn):
    possible = [c for c in CATEGORIES if scoreboard.get(c) is None]
    if not possible: return "Chance"
    scores = {cat: score_category(dice, cat) for cat in possible}
    return max(possible, key=lambda cat: scores[cat])

def cpu_select_category_gambler(dice, scoreboard, turn):
    possible = [c for c in CATEGORIES if scoreboard.get(c) is None]
    if not possible: return "Chance"
    scores = {cat: score_category(dice, cat) for cat in possible}
    high_value_gambles = ["Yahtzee", "Large Straight", "Four of a Kind", "Full House"]
    high_value_gambles = [c for c in high_value_gambles if c in possible and scores.get(c, 0) > 0]
    if high_value_gambles:
        return max(high_value_gambles, key=lambda cat: scores[cat] * 1.5)
    
    return cpu_select_category_elite(dice, scoreboard, turn)

def cpu_select_category_dispatcher(dice, scoreboard, cpu_type, turn):
    if cpu_type in ["엘리트형"]: return cpu_select_category_elite(dice, scoreboard, turn)
    if cpu_type in ["도박형", "공격형"]: return cpu_select_category_gambler(dice, scoreboard, turn)
    return cpu_select_category_normal(dice, scoreboard, turn)

def estimate_expected_score(dice, keep_idxs, scoreboard, turn, rolls_left, n_sim=50):
    total = 0
    for _ in range(n_sim):
        sim = list(dice)
        reroll = [i for i in range(5) if i not in keep_idxs]
        for _ in range(rolls_left):
            for i in reroll:
                sim[i] = random.randint(1, 6)
        best = cpu_select_category_elite(sim, scoreboard, turn)
        total += score_category(sim, best)
    return total / n_sim

def get_candidate_keeps(dice):
    unique_cands = set()
    for r in range(6):
        for c in itertools.combinations(range(5), r):
            keep_values = tuple(sorted(dice[i] for i in c))
            unique_cands.add(keep_values)
    return sorted(list(unique_cands))


def strategic_keep_elite(dice, scoreboard, turn, rolls_left):
    counts = Counter(dice)
    if sorted(counts.values()) == [2, 3] and scoreboard.get("Full House") is None:
        return list(range(5))
    
    best_keep, best_ev = [], -1
    unique_keeps = get_candidate_keeps(dice)
    for keep in unique_keeps:
        keep_idxs = [i for i, d in enumerate(dice) if d in keep]
        ev = estimate_expected_score(dice, keep_idxs, scoreboard, turn, rolls_left)
        if ev > best_ev:
            best_ev, best_keep = ev, keep_idxs
    return best_keep

def strategic_keep_gambler(dice, scoreboard, turn, rolls_left):
    counts = Counter(dice)
    if scoreboard.get("Yahtzee") is None and counts and counts.most_common(1)[0][1] >= 4:
        num = counts.most_common(1)[0][0]
        return [i for i, d in enumerate(dice) if d == num]
    
    return strategic_keep_elite(dice, scoreboard, turn, rolls_left)

def strategic_keep_attack(dice, scoreboard, turn, rolls_left):
    counts = Counter(dice)
    if scoreboard.get("Yahtzee") is None and counts.most_common(1) and counts.most_common(1)[0][1] >= 3:
        num = counts.most_common(1)[0][0]
        return [i for i, d in enumerate(dice) if d == num]
    
    high_value_nums = [i for i, d in enumerate(dice) if d >= 5]
    if len(high_value_nums) >= 2:
        return high_value_nums
        
    return strategic_keep_normal(dice, scoreboard, turn, rolls_left)

def strategic_keep_defense(dice, scoreboard, turn, rolls_left):
    upper_score = calculate_upper_score(scoreboard)
    if upper_score < 63:
        rec_cat = max(CATEGORIES[:6], key=lambda c: score_category(dice, c) if scoreboard[c] is None else -1)
        if rec_cat in CATEGORIES[:6]:
            face = CATEGORIES.index(rec_cat) + 1
            return [i for i, d in enumerate(dice) if d == face]
            
    counts = Counter(dice)
    if counts:
        num = counts.most_common(1)[0][0]
        return [i for i, d in enumerate(dice) if d == num]
    return []

def strategic_keep_normal(dice, scoreboard, turn, rolls_left):
    counts = Counter(dice)
    if counts:
        num = counts.most_common(1)[0][0]
        return [i for i, d in enumerate(dice) if d == num]
    return []

def cpu_decide_dice_to_keep(dice, scoreboard, cpu_type, turn, rolls_left):
    if cpu_type == "엘리트형": return strategic_keep_elite(dice, scoreboard, turn, rolls_left)
    if cpu_type == "도박형": return strategic_keep_gambler(dice, scoreboard, turn, rolls_left)
    if cpu_type == "공격형": return strategic_keep_attack(dice, scoreboard, turn, rolls_left)
    if cpu_type == "안정형": return strategic_keep_defense(dice, scoreboard, turn, rolls_left)
    return strategic_keep_normal(dice, scoreboard, turn, rolls_left)

# -------------------- 턴 버퍼 --------------------
def _init_turn_buf():
    return {
        "dice_roll_1": None, "kept_after_roll_1": None,
        "dice_roll_2": None, "kept_after_roll_2": None,
        "score_state_before": None,
    }

def _get_turn_buf(game_state):
    buf = game_state.get("turn_buf")
    if not buf:
        buf = _init_turn_buf()
        game_state["turn_buf"] = buf
    return buf

def _reset_turn_buf(game_state):
    game_state["turn_buf"] = _init_turn_buf()

def _end_turn(game_state, player, category, score, turn_buf):
    with transaction.atomic():
        game_session, _ = GameSession.objects.get_or_create(
            game_id=game_state['game_id'],
            defaults={'player_name': player['display_name'], 'ip_address': 'CPU_PLAYER' if player['is_cpu'] else ''}
        )
        TurnLog.objects.create(
            game_session=game_session,
            player_name=player['display_name'],
            cpu_type=player.get('type'),
            turn_number=game_state['current_turn'],
            score_state_before=turn_buf.get('score_state_before') or {},
            dice_roll_1=turn_buf.get('dice_roll_1') or "",
            kept_after_roll_1=turn_buf.get('kept_after_roll_1') or "",
            dice_roll_2=turn_buf.get('dice_roll_2') or "",
            kept_after_roll_2=turn_buf.get('kept_after_roll_2') or "",
            final_dice_state=",".join(map(str, game_state['dice'])),
            chosen_category=category,
            score_obtained=score
        )
    
    if player.get('is_cpu'):
        log_data = {
            "id": None,
            "game_id": game_state['game_id'],
            "player_name": player['display_name'],
            "turn_number": game_state['current_turn'],
            "score_state_before": json.dumps(turn_buf['score_state_before'] or {}, ensure_ascii=False),
            "dice_roll_1": turn_buf['dice_roll_1'] or "",
            "kept_after_roll_1": turn_buf['kept_after_roll_1'] or "",
            "dice_roll_2": turn_buf.get('dice_roll_2', '') or "",
            "kept_after_roll_2": turn_buf.get('kept_after_roll_2', '') or "",
            "final_dice_state": ",".join(map(str, game_state['dice'])),
            "chosen_category": category,
            "score_obtained": score,
            "created_at": timezone.now().astimezone(dt_timezone.utc).isoformat()
        }
        log_cpu_turn_to_csv(log_data)
        
    game_state['current_player_index'] = (game_state['current_player_index'] + 1) % len(game_state['players'])
    if game_state['current_player_index'] == 0:
        game_state['current_turn'] += 1
    
    game_state['rolls_left'] = 3
    game_state['dice'] = [0, 0, 0, 0, 0]
    game_state['kept_indices'] = []
    _reset_turn_buf(game_state)
    
    if game_state['current_turn'] > 12:
        game_state['is_over'] = True
        for p in game_state['players']:
            if not p['is_cpu']:
                up = calculate_upper_score(p['scoreboard'])
                bonus = calculate_bonus(up)
                total = sum(v for v in p['scoreboard'].values() if v is not None) + bonus
                GameSession.objects.filter(game_id=game_state['game_id']).update(
                    total_score=total, player_name=p['display_name']
                )

# -------------------- 페이지 --------------------
@ensure_csrf_cookie
def index(request):
    return render(request, 'yachtgame/index.html')

# -------------------- 게임 API --------------------
@require_POST
def start_game_api(request):
    try:
        data = json.loads(request.body)
        players_info = data.get('players')
        if not players_info or not isinstance(players_info, list):
            return JsonResponse({'error': 'Invalid player data'}, status=400)

        game_id = str(uuid.uuid4())
        unique_tag = game_id[:4].upper()
        game_state = {
            'game_id': game_id, 'players': [], 'current_turn': 1,
            'current_player_index': 0, 'rolls_left': 3, 'dice': [0, 0, 0, 0, 0],
            'kept_indices': [], 'is_over': False, 'log': [],
            'turn_buf': _init_turn_buf(),
        }
        for i, p in enumerate(players_info):
            original_name = p.get('name', f'Player {i+1}')
            display_name = f"{original_name}#{unique_tag}" if not p.get('is_cpu') \
                               else f"{p.get('name', 'CPU')} ({p.get('type')})"
            player_state = {
                'id': i, 'name': original_name, 'display_name': display_name,
                'is_cpu': p.get('is_cpu', False),
                'type': p.get('type') if p.get('is_cpu') else None,
                'scoreboard': {cat: None for cat in CATEGORIES}
            }
            game_state['players'].append(player_state)

        request.session['game_state'] = game_state
        return JsonResponse(game_state)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return JsonResponse({'error': f'게임 시작 중 오류가 발생했습니다: {e}'}, status=500)

@require_POST
def roll_dice_api(request):
    try:
        game_state = request.session.get('game_state')
        if not game_state or game_state.get('is_over'):
            return JsonResponse({'error': '활성화된 게임이 없습니다.'}, status=400)
        if game_state.get('rolls_left', 0) <= 0:
            return JsonResponse({'error': '남은 굴림 횟수가 없습니다.'}, status=400)
        
        new_dice = game_state.get('dice', [0, 0, 0, 0, 0])
        kept_indices = game_state.get('kept_indices', [])
        for i in range(5):
            if i not in kept_indices:
                new_dice[i] = random.randint(1, 6)
        
        game_state['dice'] = new_dice
        game_state['rolls_left'] -= 1
        pidx = game_state.get('current_player_index', 0)
        player = game_state.get('players', [])[pidx]
        pname = player.get('display_name', 'Unknown')
        game_state.get('log', []).append(f"[{pname}] 굴림 결과: [{', '.join(map(str, new_dice))}]")

        turn_buf = _get_turn_buf(game_state)
        if game_state['rolls_left'] == 2: turn_buf['dice_roll_1'] = ",".join(map(str, new_dice))
        elif game_state['rolls_left'] == 1: turn_buf['dice_roll_2'] = ",".join(map(str, new_dice))
        
        request.session['game_state'] = game_state
        return JsonResponse(game_state)
    except (KeyError, ValueError, IndexError) as e:
        return JsonResponse({'error': f'주사위 굴림 중 오류 발생: {e}'}, status=500)

@require_POST
def keep_dice_api(request):
    try:
        game_state = request.session.get('game_state')
        if not game_state or game_state.get('is_over'):
            return JsonResponse({'error': '활성화된 게임이 없습니다.'}, status=400)

        data = json.loads(request.body)
        indices_to_keep = data.get('kept_indices', [])
        if not all(isinstance(i, int) and 0 <= i < 5 for i in indices_to_keep):
            return JsonResponse({'error': '잘못된 주사위 번호입니다.'}, status=400)

        game_state['kept_indices'] = indices_to_keep
        turn_buf = _get_turn_buf(game_state)
        kept_values = [str(game_state['dice'][i]) for i in indices_to_keep]
        if game_state['rolls_left'] == 2:
            turn_buf['kept_after_roll_1'] = ",".join(kept_values) if kept_values else ""
        elif game_state['rolls_left'] == 1:
            turn_buf['kept_after_roll_2'] = ",".join(kept_values) if kept_values else ""

        request.session['game_state'] = game_state
        return JsonResponse(game_state)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return JsonResponse({'error': f'주사위 고정 중 오류 발생: {e}'}, status=500)

@require_POST
def select_category_api(request):
    try:
        game_state = request.session.get('game_state')
        if not game_state or game_state.get('is_over'):
            return JsonResponse({'error': '활성화된 게임이 없습니다.'}, status=400)

        data = json.loads(request.body)
        category = data.get('category')
        player = game_state['players'][game_state['current_player_index']]

        if player.get('is_cpu'):
            return JsonResponse({'error': 'CPU 턴에는 사용할 수 없습니다.'}, status=400)

        if category not in CATEGORIES or player['scoreboard'][category] is not None or game_state['rolls_left'] == 3:
            return JsonResponse({'error': '잘못된 행동입니다.'}, status=400)

        turn_buf = _get_turn_buf(game_state)
        turn_buf['score_state_before'] = deepcopy(player['scoreboard'])
        score = score_category(game_state['dice'], category)
        player['scoreboard'][category] = score
        game_state['log'].append(f"[{player['display_name']}]가 '{category}'를 선택하여 {score}점을 얻었습니다.")

        _end_turn(request, game_state, player, category, score, turn_buf)
        
        request.session['game_state'] = game_state
        return JsonResponse(game_state)
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        return JsonResponse({'error': f'서버 오류: {e}'}, status=500)

@require_POST
def select_category_cpu_api(request):
    try:
        game_state = request.session.get('game_state')
        if not game_state or game_state.get('is_over'):
            return JsonResponse({'error': '활성화된 게임이 없습니다.'}, status=400)

        player = game_state['players'][game_state['current_player_index']]
        if not player.get('is_cpu'):
            return JsonResponse({'error': '사람 플레이어는 일반 엔드포인트를 사용하세요.'}, status=400)

        turn_buf = _get_turn_buf(game_state)
        turn_buf['score_state_before'] = deepcopy(player['scoreboard'])

        category = cpu_select_category_dispatcher(game_state['dice'], player['scoreboard'], player['type'], game_state['current_turn'])
        score = score_category(game_state['dice'], category)
        player['scoreboard'][category] = score
        game_state['log'].append(f"[{player['display_name']}]가 '{category}'를 선택하여 {score}점을 얻었습니다.")

        _end_turn(request, game_state, player, category, score, turn_buf)
        
        request.session['game_state'] = game_state
        return JsonResponse(game_state)

    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        return JsonResponse({'error': f'CPU 카테고리 확정 중 오류: {e}'}, status=500)

@require_POST
def play_cpu_turn_api(request):
    try:
        game_state = request.session.get('game_state')
        if not game_state or game_state.get('is_over'):
            return JsonResponse({'error': '활성화된 게임이 없습니다.'}, status=400)

        player = game_state['players'][game_state['current_player_index']]
        if not player['is_cpu']:
            return JsonResponse({'error': 'CPU 턴이 아닙니다.'}, status=400)
        
        steps = []
        turn_buf = _init_turn_buf()
        
        dice = [random.randint(1, 6) for _ in range(5)]
        
        game_state['dice'] = dice
        game_state['rolls_left'] = 2
        game_state['log'].append(f"[{player['display_name']}] 1차 굴림 결과: {dice}")
        steps.append(deepcopy(game_state))
        
        turn_buf['dice_roll_1'] = ",".join(map(str, dice))
        
        kept_indices = cpu_decide_dice_to_keep(dice, player['scoreboard'], player['type'], game_state['current_turn'], 2)
        game_state['kept_indices'] = kept_indices
        kept_dice = [dice[idx] for idx in kept_indices]
        
        if kept_dice:
            game_state['log'].append(f"[{player['display_name']}] 고정: {kept_dice}")
            steps.append(deepcopy(game_state))

        turn_buf['kept_after_roll_1'] = ",".join(map(str, kept_dice)) if kept_dice else ""
        
        if len(kept_indices) < 5:
            reroll_count = 5 - len(kept_indices)
            dice = kept_dice + [random.randint(1, 6) for _ in range(reroll_count)]

            game_state['dice'] = dice
            game_state['rolls_left'] = 1
            game_state['log'].append(f"[{player['display_name']}] 2차 굴림 결과: {dice}")
            steps.append(deepcopy(game_state))

            turn_buf['dice_roll_2'] = ",".join(map(str, dice))

            kept_indices = cpu_decide_dice_to_keep(dice, player['scoreboard'], player['type'], game_state['current_turn'], 1)
            game_state['kept_indices'] = kept_indices
            kept_dice = [dice[idx] for idx in kept_indices]

            if kept_dice:
                game_state['log'].append(f"[{player['display_name']}] 고정: {kept_dice}")
                steps.append(deepcopy(game_state))

            turn_buf['kept_after_roll_2'] = ",".join(map(str, kept_dice)) if kept_dice else ""
            
            if len(kept_indices) < 5:
                reroll_count = 5 - len(kept_indices)
                dice = kept_dice + [random.randint(1, 6) for _ in range(reroll_count)]
        
        game_state['rolls_left'] = 0
        game_state['dice'] = dice
        steps.append(deepcopy(game_state))
        
        turn_buf['score_state_before'] = deepcopy(player['scoreboard'])
        category = cpu_select_category_dispatcher(dice, player['scoreboard'], player['type'], game_state['current_turn'])
        score = score_category(dice, category)
        player['scoreboard'][category] = score
        game_state['log'].append(f"[{player['display_name']}]가 '{category}'를 선택하여 {score}점을 얻었습니다.")

        _end_turn(request, game_state, player, category, score, turn_buf)
        
        request.session['game_state'] = game_state
        return JsonResponse({'steps': steps, 'final_state': game_state})
    except (KeyError, ValueError, IndexError) as e:
        return JsonResponse({'error': f'CPU 턴 진행 중 오류: {e}'}, status=500)

@require_POST
def collect_cpu_logs_api(request):
    try:
        data = json.loads(request.body)
        cpu_type = data.get('cpu_type')
        count = int(data.get('count', 1))

        if cpu_type not in ['엘리트형', '도박형', '공격형', '안정형', '일반형']:
            return JsonResponse({'error': '로그 수집은 엘리트형 또는 도박형만 가능합니다.'}, status=400)
        if not (1 <= count <= 100):
            return JsonResponse({'error': '시뮬레이션 횟수는 1에서 100 사이여야 합니다.'}, status=400)
        
        total_turns_collected = 0
        all_scores = []
        for _ in range(count):
            game_id = f"cpu-log-{uuid.uuid4()}"
            player_name = f"CPU_Logger_{cpu_type}#{game_id[:4].upper()}"
            scoreboard = {cat: None for cat in CATEGORIES}
            
            for turn in range(1, 13):
                turn_buf = _init_turn_buf()
                dice = [random.randint(1, 6) for _ in range(5)]
                
                game_state_sim = {
                    'game_id': game_id, 'players': [], 'current_turn': turn,
                    'current_player_index': 0, 'rolls_left': 3, 'dice': dice,
                    'kept_indices': [], 'is_over': False, 'log': [],
                    'turn_buf': turn_buf,
                }

                turn_buf['dice_roll_1'] = ",".join(map(str, dice))
                kept_indices = cpu_decide_dice_to_keep(dice, scoreboard, cpu_type, turn, 2)
                kept_dice = [dice[idx] for idx in kept_indices]
                turn_buf['kept_after_roll_1'] = ",".join(map(str, kept_dice)) if kept_dice else ""

                if len(kept_indices) < 5:
                    reroll_count = 5 - len(kept_indices)
                    dice = kept_dice + [random.randint(1, 6) for _ in range(reroll_count)]
                    
                    turn_buf['dice_roll_2'] = ",".join(map(str, dice))
                    kept_indices = cpu_decide_dice_to_keep(dice, scoreboard, cpu_type, turn, 1)
                    kept_dice = [dice[idx] for idx in kept_indices]
                    turn_buf['kept_after_roll_2'] = ",".join(map(str, kept_dice)) if kept_dice else ""
                    
                    if len(kept_indices) < 5:
                        reroll_count = 5 - len(kept_indices)
                        dice = kept_dice + [random.randint(1, 6) for _ in range(reroll_count)]

                final_dice = list(dice)
                turn_buf['score_state_before'] = deepcopy(scoreboard)
                category = cpu_select_category_dispatcher(final_dice, scoreboard, cpu_type, turn)
                score = score_category(final_dice, category)
                scoreboard[category] = score

                log_data = {
                    "id": None,
                    "game_id": game_id,
                    "player_name": player_name,
                    "turn_number": turn,
                    "score_state_before": json.dumps(turn_buf['score_state_before'] or {}, ensure_ascii=False),
                    "dice_roll_1": turn_buf['dice_roll_1'] or "",
                    "kept_after_roll_1": turn_buf['kept_after_roll_1'] or "",
                    "dice_roll_2": turn_buf.get('dice_roll_2', '') or "",
                    "kept_after_roll_2": turn_buf.get('kept_after_roll_2', '') or "",
                    "final_dice_state": ",".join(map(str, final_dice)),
                    "chosen_category": category,
                    "score_obtained": score,
                    "created_at": timezone.now().astimezone(dt_timezone.utc).isoformat()
                }
                log_cpu_turn_to_csv(log_data)
                total_turns_collected += 1

            up = calculate_upper_score(scoreboard)
            bonus = calculate_bonus(up)
            total = sum(v for v in scoreboard.values() if v is not None) + bonus
            all_scores.append(total)

        return JsonResponse({
            'message': '로그 수집 완료',
            'games_collected': count,
            'turns_collected': total_turns_collected,
            'average_score': statistics.mean(all_scores) if all_scores else 0
        })
    except (json.JSONDecodeError, KeyError, ValueError, IndexError, TypeError) as e:
        return JsonResponse({'error': f'로그 수집 중 오류: {e}'}, status=500)

@require_GET
def get_game_state_api(request):
    game_state = request.session.get('game_state')
    if game_state and not game_state.get('is_over'):
        return JsonResponse(game_state)
    return JsonResponse({'message': 'No active game found.'}, status=404)

@require_POST
def analyze_cpu_api(request):
    try:
        data = json.loads(request.body)
        cpu_type = data.get('cpu_type')
        count = int(data.get('count', 50))

        if cpu_type not in CPU_TYPES and cpu_type != "ML형":
            return JsonResponse({'error': '잘못된 CPU 유형입니다.'}, status=400)
        if count <= 0 or count > 10000:
            return JsonResponse({'error': '시뮬레이션 횟수 범위를 확인하세요.'}, status=400)

        scores = []
        for _ in range(count):
            scoreboard = {cat: None for cat in CATEGORIES}
            for turn in range(1, 13):
                dice = [random.randint(1, 6) for _ in range(5)]
                for i in range(2):
                    kept = cpu_decide_dice_to_keep(dice, scoreboard, cpu_type, turn, 2 - i)
                    if len(kept) == 5: break
                    reroll_count = 5 - len(kept)
                    dice = [dice[i] for i in kept] + [random.randint(1, 6) for _ in range(reroll_count)]

                category = cpu_select_category_dispatcher(dice, scoreboard, cpu_type, turn)
                score = score_category(dice, category)
                scoreboard[category] = score

            up = calculate_upper_score(scoreboard)
            bonus = calculate_bonus(up)
            total = sum(v for v in scoreboard.values() if v is not None) + bonus
            scores.append(total)

        return JsonResponse({
            'count': count, 'mean': statistics.mean(scores), 'max': max(scores), 'min': min(scores),
        })
    except (json.JSONDecodeError, KeyError, ValueError, IndexError, TypeError) as e:
        return JsonResponse({'error': f'CPU 분석 중 오류: {e}'}, status=500)

@require_GET
def get_hall_of_fame(request):
    try:
        now_kst = timezone.localtime(timezone.now())
        today_start_kst = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
        if now_kst.hour < 9: today_start_kst -= timedelta(days=1)
        hall = GameSession.objects.filter(
            created_at__gte=today_start_kst, total_score__isnull=False
        ).order_by('-total_score')[:10]
        return JsonResponse(list(hall.values('player_name', 'total_score')), safe=False)
    except Exception:
        return JsonResponse([], safe=False)

@require_GET
def get_weekly_high_scores(request):
    try:
        now_kst = timezone.localtime(timezone.now())
        days_since_wed = (now_kst.weekday() - 2 + 7) % 7
        weekly_start_kst = (now_kst - timedelta(days=days_since_wed)).replace(hour=9, minute=0, second=0, microsecond=0)
        if now_kst < weekly_start_kst: weekly_start_kst -= timedelta(days=7)
        weekly_end_kst = weekly_start_kst + timedelta(days=7)
        weekly = GameSession.objects.filter(
            created_at__gte=weekly_start_kst, created_at__lt=weekly_end_kst, total_score__isnull=False
        ).order_by('-total_score')[:10]
        return JsonResponse(list(weekly.values('player_name', 'total_score')), safe=False)
    except Exception:
        return JsonResponse([], safe=False)

@require_GET
def get_notice(request):
    return render(request, 'yachtgame/notice.html')

@require_GET
def get_patch_notes(request):
    return render(request, 'yachtgame/patch_notes.html')

@require_GET
def get_all_logs_api(request):
    logs = TurnLog.objects.all().order_by('-created_at')[:100]
    data = list(logs.values(
        'player_name', 'turn_number',
        'dice_roll_1', 'kept_after_roll_1',
        'dice_roll_2', 'kept_after_roll_2',
        'final_dice_state', 'chosen_category', 'score_obtained',
        'created_at'
    ))
    for entry in data:
        entry['created_at'] = timezone.localtime(entry['created_at']).strftime('%Y-%m-%d %H:%M:%S')
    return JsonResponse(data, safe=False)

# -------------------- CSV Export (개발자용) --------------------
class _Echo:
    def write(self, value):
        return value

def _parse_date(d):
    if not d: return None
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        return timezone.make_aware(dt)
    except Exception:
        return None

def _filename(prefix, start_dt, end_dt):
    if not start_dt and not end_dt:
        return f"{prefix}_ALL.csv"
    s = start_dt.strftime("%Y%m%d") if start_dt else "NA"
    e = (end_dt - timedelta(days=1)).strftime("%Y%m%d") if end_dt else "NA"
    return f"{prefix}_{s}_{e}.csv"

@require_POST
def export_logs_csv(request):
    password = request.POST.get("password", "")
    if password != DEV_PASSWORD:
        return HttpResponseForbidden("invalid password")

    dataset = request.POST.get("dataset", "")
    if dataset not in ("events", "turns", "cpu_turns"):
        return HttpResponseBadRequest("dataset must be 'events', 'turns', or 'cpu_turns'")

    start_dt = _parse_date(request.POST.get("start"))
    end_dt = _parse_date(request.POST.get("end"))
    if end_dt: end_dt += timedelta(days=1)

    if dataset == "cpu_turns":
        if not os.path.exists(CPU_LOG_FILE_PATH):
            return HttpResponseNotFound("CPU 로그 파일이 존재하지 않습니다.")

        pseudo = _Echo()
        writer = csv.writer(pseudo)
        
        def stream_cpu_logs():
            with open(CPU_LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                yield writer.writerow(reader.fieldnames)
                for row in reader:
                    try:
                        created_at_str = row.get('created_at', '')
                        if '+' in created_at_str:
                            created_at_dt = datetime.fromisoformat(created_at_str)
                        else:
                            created_at_dt = timezone.make_aware(datetime.fromisoformat(created_at_str))

                        if start_dt and created_at_dt < start_dt: continue
                        if end_dt and created_at_dt >= end_dt: continue
                        
                        yield writer.writerow([row.get(h, '') for h in reader.fieldnames])
                    except (ValueError, TypeError):
                        continue
        
        filename = _filename("cpu_turn_logs", start_dt, end_dt)
        response = StreamingHttpResponse(stream_cpu_logs(), content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

    elif dataset == "events":
        qs = GameSession.objects.filter(phone_number__isnull=False).order_by("created_at")
        if start_dt: qs = qs.filter(created_at__gte=start_dt)
        if end_dt: qs = qs.filter(created_at__lt=end_dt)
        headers = ["created_at", "game_id", "player_name", "total_score", "phone_hash", "ip_address"]
        def gen_rows():
            yield headers
            for r in qs.iterator():
                kst = timezone.localtime(r.created_at).strftime("%Y-%m-%d %H:%M:%S")
                yield [kst, r.game_id, r.player_name, r.total_score, r.phone_number or "", r.ip_address or ""]
        filename = _filename("event_logs", start_dt, end_dt)

    else:  # "turns"
        qs = TurnLog.objects.select_related("game_session").order_by("created_at")
        if start_dt: qs = qs.filter(created_at__gte=start_dt)
        if end_dt: qs = qs.filter(created_at__lt=end_dt)
        headers = [
            "id","game_id","player_name","turn_number","score_state_before",
            "dice_roll_1","kept_after_roll_1","dice_roll_2","kept_after_roll_2",
            "final_dice_state","chosen_category","score_obtained","created_at"
        ]
        def gen_rows():
            yield headers
            for r in qs.iterator():
                gid = r.game_session.game_id if r.game_session_id else ""
                score_before_json = json.dumps(r.score_state_before or {}, ensure_ascii=False)
                created_iso = r.created_at.astimezone(dt_timezone.utc).isoformat()
                yield [
                    r.id, gid, r.player_name, r.turn_number, score_before_json,
                    r.dice_roll_1 or "", r.kept_after_roll_1 or "",
                    r.dice_roll_2 or "", r.kept_after_roll_2 or "",
                    r.final_dice_state or "", r.chosen_category or "",
                    r.score_obtained, created_iso,
                ]
        filename = _filename("turn_logs", start_dt, end_dt)

    pseudo = _Echo()
    writer = csv.writer(pseudo)
    def stream():
        for row in gen_rows():
            yield writer.writerow(row)
    resp = StreamingHttpResponse(stream(), content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp

# 1. 이벤트 기능 비활성화
# @require_POST
# def save_event_entry_api(request):
#     try:
#         data = json.loads(request.body)
#         game_id = data.get('game_id')
#         player_name = data.get('player_name')
#         phone_number = data.get('phone_number')
#         recaptcha_token = data.get('recaptcha_token')

#         if not all([game_id, player_name, phone_number, recaptcha_token]):
#             return JsonResponse({'error': '모든 필드를 입력해야 합니다.'}, status=400)

#         verify_payload = {'secret': settings.RECAPTCHA_PRIVATE_KEY, 'response': recaptcha_token}
#         verify_response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=verify_payload)
#         result = verify_response.json()
#         if not result.get('success') or result.get('score', 0) < 0.5:
#             return JsonResponse({'error': 'reCAPTCHA 인증에 실패했습니다. 봇으로 의심됩니다.'}, status=403)

#         cleaned_phone_number = phone_number.replace('-', '').strip()
#         phone_hash = hashlib.sha256(cleaned_phone_number.encode()).hexdigest()

#         now_kst = timezone.localtime(timezone.now())
#         today_start_kst = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
#         if now_kst.hour < 9:
#             today_start_kst -= timedelta(days=1)

#         existing_entry = GameSession.objects.filter(
#             phone_number=phone_hash,
#             created_at__gte=today_start_kst
#         ).order_by('-total_score').first()

#         try:
#             current_game_session = GameSession.objects.get(game_id=game_id, player_name=player_name)
#         except GameSession.DoesNotExist:
#             return JsonResponse({'error': '유효하지 않은 게임 정보입니다.'}, status=404)

#         if existing_entry:
#             if current_game_session.total_score > existing_entry.total_score:
#                 existing_entry.total_score = current_game_session.total_score
#                 existing_entry.save()
#                 return JsonResponse({'message': '최고 기록 갱신 성공!'})
#             else:
#                 return JsonResponse({'error': '기존 점수보다 낮아 갱신되지 않았습니다.'}, status=400)
#         else:
#             current_game_session.phone_number = phone_hash
#             current_game_session.save()
#             return JsonResponse({'message': '이벤트 참여가 완료되었습니다.'})

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=400)
