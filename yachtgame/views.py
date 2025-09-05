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
    HttpResponseNotFound,
)
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .models import GameSession, TurnLog


# -------------------- 상수 --------------------
CATEGORIES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "Four of a Kind", "Full House", "Small Straight", "Large Straight",
    "Yahtzee", "Chance"
]
CPU_TYPES = ["엘리트형", "도박형", "공격형", "안정형", "일반형", "ML형"]
BASE_WEIGHTS = {
    "Ones": 0.3, "Twos": 0.4, "Threes": 0.6, "Fours": 0.8, "Fives": 1.0, "Sixes": 1.2,
    "Four of a Kind": 1.8, "Full House": 2.0, "Small Straight": 1.1,
    "Large Straight": 1.6, "Yahtzee": 3.0, "Chance": 1.0
}
DEV_PASSWORD = "Split5234"

CPU_LOG_FILE_PATH = os.path.join(settings.MEDIA_ROOT, 'cpu_turn_logs.csv')
CPU_LOG_HEADERS = [
    "id","game_id","player_name","cpu_type","turn_number",
    "score_state_before",
    "dice_roll_1","kept_after_roll_1","dice_roll_2","kept_after_roll_2",
    "final_dice_state","chosen_category","score_obtained","created_at"
]
csv_writer_lock = threading.Lock()
# ID 생성을 위한 전역 카운터 및 락 추가
cpu_log_id_counter = 0
cpu_log_id_lock = threading.Lock()

# -------------------- CPU 로그 기록 함수 --------------------
def log_cpu_turn_to_csv(log_data):
    """CPU 턴 로그를 CSV 파일에 기록합니다."""
    global cpu_log_id_counter
    with csv_writer_lock:
        file_exists = os.path.isfile(CPU_LOG_FILE_PATH)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        with cpu_log_id_lock:
            # 서버 시작 시 한번만 CSV의 마지막 ID를 읽어와 카운터 초기화
            if cpu_log_id_counter == 0 and file_exists and os.path.getsize(CPU_LOG_FILE_PATH) > 0:
                try:
                    with open(CPU_LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                        # 파일의 마지막 줄을 효율적으로 읽기
                        last_line = None
                        for last_line in f:
                            pass
                        if last_line and last_line.strip().split(',')[0].isdigit():
                            last_id = int(last_line.split(',')[0])
                            cpu_log_id_counter = last_id
                except (IOError, IndexError, ValueError):
                     cpu_log_id_counter = 0 # 파일 읽기 실패 시 0부터 시작

            cpu_log_id_counter += 1
            log_data['id'] = cpu_log_id_counter
        
        with open(CPU_LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CPU_LOG_HEADERS)
            if not file_exists or os.path.getsize(CPU_LOG_FILE_PATH) == 0:
                writer.writeheader()
            writer.writerow(log_data)


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


# -------------------- CPU 의사결정 --------------------
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


def cpu_select_category_simple(dice, scoreboard):
    possible = [c for c in CATEGORIES if scoreboard.get(c) is None]
    if not possible: return "Chance"
    return max(possible, key=lambda cat: score_category(dice, cat))


def cpu_select_category_dispatcher(dice, scoreboard, cpu_type, turn):
    if cpu_type == "엘리트형":
        return cpu_select_category_elite(dice, scoreboard, turn)
    return cpu_select_category_simple(dice, scoreboard)


def estimate_expected_score(dice, keep_idxs, scoreboard, turn, rolls_left, n_sim=100):
    """[성능 복원] CPU의 기대 점수를 계산합니다. 빠른 시뮬레이션을 위해 단순화된 버전을 사용합니다."""
    total = 0
    for _ in range(n_sim):
        sim_dice = [dice[i] for i in keep_idxs]
        reroll_count = 5 - len(sim_dice)
        # 남은 롤을 모두 소진했을 때의 최종 결과만 한 번 시뮬레이션합니다.
        final_dice = sim_dice + [random.randint(1, 6) for _ in range(reroll_count * rolls_left)]
        final_dice = final_dice[:5]

        best_cat = cpu_select_category_elite(final_dice, scoreboard, turn)
        total += score_category(final_dice, best_cat)
    return total / n_sim


def get_candidate_keeps(dice, scoreboard, turn):
    cands = [list(c) for r in range(6) for c in itertools.combinations(range(5), r)]
    unique, seen = [], set()
    for cand in cands:
        key = tuple(sorted([dice[i] for i in cand]))
        if key not in seen:
            unique.append(cand)
            seen.add(key)
    return unique


def strategic_keep_elite(dice, scoreboard, turn, rolls_left):
    counts = Counter(dice)
    if sorted(counts.values()) == [1, 2, 2] and scoreboard.get("Full House") is None:
        pair_nums = [n for n, c in counts.items() if c == 2]
        return [i for i, d in enumerate(dice) if d in pair_nums]
    best_keep, best_ev = [], -1
    for keep in get_candidate_keeps(dice, scoreboard, turn):
        ev = estimate_expected_score(dice, keep, scoreboard, turn, rolls_left)
        if ev > best_ev:
            best_ev, best_keep = ev, keep
    return best_keep


def get_recommended_target_gambler(dice, scoreboard):
    possible = [c for c, s in scoreboard.items() if s is None]
    if not possible: return "Chance"
    return max(possible, key=lambda c: score_category(dice, c) * BASE_WEIGHTS.get(c, 1.0))

def strategic_keep_gambler(dice, scoreboard):
    counts = Counter(dice)
    if scoreboard.get("Yahtzee") is None and 5 in counts.values(): return list(range(5))
    if scoreboard.get("Full House") is None and sorted(counts.values()) == [2, 3]: return list(range(5))
    if scoreboard.get("Yahtzee") is None and counts and counts.most_common(1)[0][1] >= 4:
        num = counts.most_common(1)[0][0]
        return [i for i, d in enumerate(dice) if d == num]
    
    tgt = get_recommended_target_gambler(dice, scoreboard)
    if not tgt: return []

    keep_indices = []
    if tgt in CATEGORIES[:6]:
        face = CATEGORIES.index(tgt) + 1
        keep_indices = [i for i, d in enumerate(dice) if d == face]
    elif tgt in ("Four of a Kind", "Yahtzee"):
        if counts:
            num = counts.most_common(1)[0][0]
            keep_indices = [i for i, d in enumerate(dice) if d == num]
    elif tgt == "Full House":
        nums_to_keep = [num for num, count in counts.items() if count in [2, 3]]
        if nums_to_keep:
            keep_indices = [i for i, d in enumerate(dice) if d in nums_to_keep]
    elif tgt in ("Small Straight", "Large Straight"):
        dice_set = sorted(list(set(dice)))
        if not dice_set: return []
        best_seq = []
        current_seq = [dice_set[0]]
        for i in range(1, len(dice_set)):
            if dice_set[i] == dice_set[i-1] + 1: current_seq.append(dice_set[i])
            else:
                if len(current_seq) > len(best_seq): best_seq = current_seq
                current_seq = [dice_set[i]]
        if len(current_seq) > len(best_seq): best_seq = current_seq
        if len(best_seq) >= 3:
            temp_dice = list(dice)
            indices = []
            for val in best_seq:
                try:
                    idx = temp_dice.index(val)
                    indices.append(idx)
                    temp_dice[idx] = -1
                except ValueError: pass
            keep_indices = indices

    if not keep_indices and counts:
        num = counts.most_common(1)[0][0]
        keep_indices = [i for i, d in enumerate(dice) if d == num]
        
    return keep_indices


def strategic_keep_attack(dice, scoreboard, turn):
    counts = Counter(dice)
    if scoreboard.get("Yahtzee") is None and counts.most_common(1) and counts.most_common(1)[0][1] >= 3:
        return [i for i, d in enumerate(dice) if d == counts.most_common(1)[0][0]]
    if scoreboard.get("Full House") is None and sorted(counts.values()) == [2, 3]:
        return list(range(5))
    possible = [c for c, s in scoreboard.items() if s is None]
    if not possible: return []
    rec = max(possible, key=lambda c: score_category(dice, c) * BASE_WEIGHTS.get(c, 1.0))
    if rec in CATEGORIES[:6]:
        return [i for i, d in enumerate(dice) if d == CATEGORIES.index(rec) + 1]
    if counts: return [i for i, d in enumerate(dice) if d == counts.most_common(1)[0][0]]
    return []


def strategic_keep_defense(dice, scoreboard, turn):
    counts = Counter(dice)
    upper_score = calculate_upper_score(scoreboard)
    remain_upper = [c for c in CATEGORIES[:6] if scoreboard.get(c) is None]
    if upper_score < 63 and remain_upper:
        rec = max(remain_upper, key=lambda c: score_category(dice, c))
        face = CATEGORIES.index(rec) + 1
        return [i for i, d in enumerate(dice) if d == face]
    possible = [c for c, s in scoreboard.items() if s is None]
    if not possible: return list(range(5))
    rec = max(possible, key=lambda c: score_category(dice, c))
    if rec in CATEGORIES[:6]:
        return [i for i, d in enumerate(dice) if d == CATEGORIES.index(rec) + 1]
    if counts: return [i for i, d in enumerate(dice) if d == counts.most_common(1)[0][0]]
    return []


def strategic_keep_normal(dice, scoreboard, turn):
    counts = Counter(dice)
    if counts:
        num = counts.most_common(1)[0][0]
        return [i for i, d in enumerate(dice) if d == num]
    return []


def cpu_decide_dice_to_keep(dice, scoreboard, cpu_type, turn, rolls_left):
    if cpu_type == "엘리트형": return strategic_keep_elite(dice, scoreboard, turn, rolls_left)
    if cpu_type == "도박형": return strategic_keep_gambler(dice, scoreboard)
    if cpu_type == "공격형": return strategic_keep_attack(dice, scoreboard, turn)
    if cpu_type == "안정형": return strategic_keep_defense(dice, scoreboard, turn)
    return strategic_keep_normal(dice, scoreboard, turn)


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
            'game_id': game_id,
            'players': [],
            'current_turn': 1,
            'current_player_index': 0,
            'rolls_left': 3,
            'dice': [0, 0, 0, 0, 0],
            'kept_indices': [],
            'is_over': False,
            'log': [],
            'turn_buf': _init_turn_buf(),
        }
        for i, p in enumerate(players_info):
            original_name = p.get('name', f'Player {i+1}')
            display_name = f"{original_name}#{unique_tag}" if not p.get('is_cpu') \
                           else f"{p.get('name', 'CPU')} ({p.get('type')})"
            player_state = {
                'id': i,
                'name': original_name,
                'display_name': display_name,
                'is_cpu': p.get('is_cpu', False),
                'type': p.get('type') if p.get('is_cpu') else None,
                'scoreboard': {cat: None for cat in CATEGORIES}
            }
            game_state['players'].append(player_state)

        request.session['game_state'] = game_state
        return JsonResponse(game_state)
    except Exception:
        return JsonResponse({'error': '게임 시작 중 오류가 발생했습니다.'}, status=500)

@require_POST
def select_category_cpu_api(request):
    try:
        game_state = request.session.get('game_state')
        if not game_state or game_state.get('is_over'):
            return JsonResponse({'error': '활성화된 게임이 없습니다.'}, status=400)

        player = game_state['players'][game_state['current_player_index']]
        if not player.get('is_cpu'):
            return JsonResponse({'error': '사람 플레이어는 일반 엔드포인트를 사용하세요.'}, status=400)

        data = json.loads(request.body)
        category = data.get('category')

        if category not in CATEGORIES or player['scoreboard'][category] is not None or game_state['rolls_left'] == 3:
            return JsonResponse({'error': '잘못된 행동입니다.'}, status=400)

        turn_buf = _get_turn_buf(game_state)
        turn_buf['score_state_before'] = deepcopy(player['scoreboard'])

        score = score_category(game_state['dice'], category)
        player['scoreboard'][category] = score
        game_state['log'].append(f"[{player['display_name']}]가 '{category}'를 선택하여 {score}점을 얻었습니다.")

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

        request.session['game_state'] = game_state
        return JsonResponse(game_state)

    except Exception as e:
        return JsonResponse({'error': 'CPU 카테고리 확정 중 오류', 'detail': str(e)}, status=500)

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
        if game_state['rolls_left'] == 2 and turn_buf.get('dice_roll_1') is None:
            turn_buf['dice_roll_1'] = ",".join(map(str, new_dice))
        elif game_state['rolls_left'] == 1 and turn_buf.get('dice_roll_2') is None:
            turn_buf['dice_roll_2'] = ",".join(map(str, new_dice))

        request.session['game_state'] = game_state
        return JsonResponse(game_state)

    except (IndexError, KeyError):
        return JsonResponse({'error': '게임 상태에 오류가 발생했습니다. 새 게임을 시작해주세요.'}, status=500)
    except Exception:
        return JsonResponse({'error': '서버에서 예기치 않은 오류가 발생했습니다.'}, status=500)


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
    except Exception:
        return JsonResponse({'error': '주사위 고정 중 오류가 발생했습니다.'}, status=500)


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
            return JsonResponse({'error': 'CPU는 전용 엔드포인트를 사용합니다.'}, status=400)

        if category not in CATEGORIES or player['scoreboard'][category] is not None or game_state['rolls_left'] == 3:
            return JsonResponse({'error': '잘못된 행동입니다.'}, status=400)

        turn_buf = _get_turn_buf(game_state)
        turn_buf['score_state_before'] = deepcopy(player['scoreboard'])

        score = score_category(game_state['dice'], category)
        player['scoreboard'][category] = score
        game_state['log'].append(f"[{player['display_name']}]가 '{category}'를 선택하여 {score}점을 얻었습니다.")

        with transaction.atomic():
            game_session, _ = GameSession.objects.get_or_create(
                game_id=game_state['game_id'],
                defaults={'player_name': player['display_name'], 'ip_address': request.META.get('REMOTE_ADDR')}
            )
            TurnLog.objects.create(
                game_session=game_session,
                player_name=player['display_name'],
                turn_number=game_state['current_turn'],
                dice_roll_1=turn_buf.get('dice_roll_1') or "",
                kept_after_roll_1=turn_buf.get('kept_after_roll_1') or "",
                dice_roll_2=turn_buf.get('dice_roll_2') or "",
                kept_after_roll_2=turn_buf.get('kept_after_roll_2') or "",
                final_dice_state=",".join(map(str, sorted(game_state['dice']))),
                chosen_category=category,
                score_obtained=score,
                score_state_before=turn_buf.get('score_state_before') or {}
            )

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

        request.session['game_state'] = game_state
        return JsonResponse(game_state)

    except Exception as e:
        return JsonResponse({'error': '족보 선택 중 오류가 발생했습니다.', 'detail': str(e)}, status=500)


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

        # --- Roll 1 ---
        dice = [random.randint(1, 6) for _ in range(5)]
        game_state['dice'] = dice
        game_state['rolls_left'] = 2
        game_state['log'].append(f"[{player['display_name']}] 1차 굴림 결과: {dice}")
        steps.append(deepcopy(game_state))
        turn_buf['dice_roll_1'] = ",".join(map(str, sorted(dice)))

        # --- Keep 1 ---
        kept_indices_1 = cpu_decide_dice_to_keep(dice, player['scoreboard'], player['type'], game_state['current_turn'], 2)
        game_state['kept_indices'] = kept_indices_1
        kept_dice_1 = [dice[i] for i in kept_indices_1]
        if kept_dice_1:
             game_state['log'].append(f"[{player['display_name']}] 고정: {kept_dice_1}")
             steps.append(deepcopy(game_state))
        turn_buf['kept_after_roll_1'] = ",".join(map(str, sorted(kept_dice_1))) if kept_dice_1 else ""
        
        # --- Roll 2 & Keep 2 ---
        final_dice = sorted(kept_dice_1)
        if len(kept_dice_1) < 5:
            reroll_count_2 = 5 - len(kept_dice_1)
            dice_2 = sorted(kept_dice_1 + [random.randint(1, 6) for _ in range(reroll_count_2)])
            game_state['dice'] = dice_2
            game_state['rolls_left'] = 1
            game_state['log'].append(f"[{player['display_name']}] 2차 굴림 결과: {dice_2}")
            steps.append(deepcopy(game_state))
            turn_buf['dice_roll_2'] = ",".join(map(str, dice_2))

            kept_indices_2 = cpu_decide_dice_to_keep(dice_2, player['scoreboard'], player['type'], game_state['current_turn'], 1)
            game_state['kept_indices'] = kept_indices_2
            kept_dice_2 = [dice_2[i] for i in kept_indices_2]
            if kept_dice_2:
                game_state['log'].append(f"[{player['display_name']}] 고정: {kept_dice_2}")
                steps.append(deepcopy(game_state))
            turn_buf['kept_after_roll_2'] = ",".join(map(str, sorted(kept_dice_2))) if kept_dice_2 else ""
            
            # --- Roll 3 (Final) ---
            if len(kept_dice_2) < 5:
                 reroll_count_3 = 5 - len(kept_dice_2)
                 final_dice = sorted(kept_dice_2 + [random.randint(1, 6) for _ in range(reroll_count_3)])
                 game_state['dice'] = final_dice
                 game_state['rolls_left'] = 0
                 game_state['log'].append(f"[{player['display_name']}] 3차 굴림 결과: {final_dice}")
                 steps.append(deepcopy(game_state))
            else:
                 final_dice = sorted(kept_dice_2)
        
        game_state['dice'] = final_dice

        # --- Select Category & Log ---
        turn_buf['score_state_before'] = deepcopy(player['scoreboard'])
        category = cpu_select_category_dispatcher(final_dice, player['scoreboard'], player['type'], game_state['current_turn'])
        score = score_category(final_dice, category)
        player['scoreboard'][category] = score
        game_state['log'].append(f"[{player['display_name']}]가 '{category}'를 선택하여 {score}점을 얻었습니다.")

        if player['type'] in ['엘리트형', '도박형']:
            log_data = {
                "id": None, "game_id": game_state['game_id'], "player_name": player['display_name'],
                "cpu_type": player['type'], "turn_number": game_state['current_turn'],
                "score_state_before": json.dumps(turn_buf['score_state_before'] or {}, ensure_ascii=False),
                "dice_roll_1": turn_buf['dice_roll_1'] or "", "kept_after_roll_1": turn_buf['kept_after_roll_1'] or "",
                "dice_roll_2": turn_buf.get('dice_roll_2', '') or "", "kept_after_roll_2": turn_buf.get('kept_after_roll_2', '') or "",
                "final_dice_state": ",".join(map(str, sorted(final_dice))),
                "chosen_category": category, "score_obtained": score,
                "created_at": timezone.now().isoformat()
            }
            log_cpu_turn_to_csv(log_data)

        game_state['current_player_index'] = (game_state['current_player_index'] + 1) % len(game_state['players'])
        if game_state['current_player_index'] == 0:
            game_state['current_turn'] += 1

        game_state['rolls_left'] = 3
        game_state['dice'] = [0, 0, 0, 0, 0]
        game_state['kept_indices'] = []
        
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

        request.session['game_state'] = game_state
        return JsonResponse({'steps': steps, 'final_state': game_state})

    except Exception as e:
        return JsonResponse({'error': 'CPU 턴 진행 중 오류가 발생했습니다.', 'detail': str(e)}, status=500)

@require_POST
def collect_cpu_logs_api(request):
    try:
        data = json.loads(request.body)
        cpu_type = data.get('cpu_type')
        count = int(data.get('count', 1))

        if cpu_type not in ['엘리트형', '도박형']:
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
                
                # Roll 1
                dice = [random.randint(1, 6) for _ in range(5)]
                turn_buf['dice_roll_1'] = ",".join(map(str, sorted(dice)))
                
                # Keep 1
                kept_indices_1 = cpu_decide_dice_to_keep(dice, scoreboard, cpu_type, turn, 2)
                kept_dice_1 = [dice[i] for i in kept_indices_1]
                turn_buf['kept_after_roll_1'] = ",".join(map(str, sorted(kept_dice_1))) if kept_dice_1 else ""

                # Roll 2 & Keep 2
                dice_2 = sorted(kept_dice_1)
                kept_dice_2 = []
                if len(kept_indices_1) < 5:
                    reroll_count_2 = 5 - len(kept_indices_1)
                    dice_2 = sorted(kept_dice_1 + [random.randint(1, 6) for _ in range(reroll_count_2)])
                    turn_buf['dice_roll_2'] = ",".join(map(str, dice_2))
                    
                    kept_indices_2 = cpu_decide_dice_to_keep(dice_2, scoreboard, cpu_type, turn, 1)
                    kept_dice_2 = [dice_2[i] for i in kept_indices_2]
                    turn_buf['kept_after_roll_2'] = ",".join(map(str, sorted(kept_dice_2))) if kept_dice_2 else ""
                
                # Roll 3 (Final)
                final_dice = []
                if len(kept_dice_2) < 5:
                     reroll_count_3 = 5 - len(kept_dice_2)
                     final_dice = sorted(kept_dice_2 + [random.randint(1, 6) for _ in range(reroll_count_3)])
                else:
                     final_dice = sorted(kept_dice_2)

                turn_buf['score_state_before'] = deepcopy(scoreboard)
                category = cpu_select_category_dispatcher(final_dice, scoreboard, cpu_type, turn)
                score = score_category(final_dice, category)
                scoreboard[category] = score

                log_data = {
                    "id": None, "game_id": game_id, "player_name": player_name,
                    "cpu_type": cpu_type, "turn_number": turn,
                    "score_state_before": json.dumps(turn_buf['score_state_before'] or {}, ensure_ascii=False),
                    "dice_roll_1": turn_buf['dice_roll_1'] or "",
                    "kept_after_roll_1": turn_buf['kept_after_roll_1'] or "",
                    "dice_roll_2": turn_buf.get('dice_roll_2', "") or "",
                    "kept_after_roll_2": turn_buf.get('kept_after_roll_2', "") or "",
                    "final_dice_state": ",".join(map(str, sorted(final_dice))),
                    "chosen_category": category, "score_obtained": score,
                    "created_at": timezone.now().isoformat()
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
    except Exception as e:
        return JsonResponse({'error': 'CPU 로그 수집 중 오류 발생', 'detail': str(e)}, status=500)


# -------------------- 이벤트/랭킹/로그 API --------------------
# 1. 이벤트 기능 비활성화: 관련 API를 주석 처리합니다.
# @require_POST
# def save_event_entry_api(request):
#     ... (기존 코드)

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
                    if len(kept) == 5:
                        break
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
            'count': count,
            'mean': statistics.mean(scores),
            'max': max(scores),
            'min': min(scores),
        })
    except Exception as e:
        return JsonResponse({'error': 'CPU 분석 중 오류가 발생했습니다.', 'detail': str(e)}, status=500)


@require_GET
def get_hall_of_fame(request):
    try:
        now_kst = timezone.localtime(timezone.now())
        today_start_kst = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
        if now_kst.hour < 9:
            today_start_kst -= timedelta(days=1)
        hall = GameSession.objects.filter(
            created_at__gte=today_start_kst, total_score__isnull=False
        ).order_by('-total_score')[:10]
        data = list(hall.values('player_name', 'total_score'))
        return JsonResponse(data, safe=False)
    except Exception:
        return JsonResponse([], safe=False)


@require_GET
def get_weekly_high_scores(request):
    try:
        now_kst = timezone.localtime(timezone.now())
        days_since_wed = (now_kst.weekday() - 2 + 7) % 7
        weekly_start_kst = (now_kst - timedelta(days=days_since_wed)).replace(hour=9, minute=0, second=0, microsecond=0)
        if now_kst < weekly_start_kst:
            weekly_start_kst -= timedelta(days=7)
        weekly_end_kst = weekly_start_kst + timedelta(days=7)
        weekly = GameSession.objects.filter(
            created_at__gte=weekly_start_kst, created_at__lt=weekly_end_kst, total_score__isnull=False
        ).order_by('-total_score')[:10]
        data = list(weekly.values('player_name', 'total_score'))
        return JsonResponse(data, safe=False)
    except Exception:
        return JsonResponse([], safe=False)


@require_GET
def get_notice(request):
    return render(request, 'yachtgame/notice.html')


@require_GET
def get_patch_notes(request):
    return render(request, 'yachtgame/patch_notes.html')


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

    start_dt_str = request.POST.get("start")
    end_dt_str = request.POST.get("end")
    start_dt = _parse_date(start_dt_str)
    end_dt = _parse_date(end_dt_str)
    if end_dt:
        end_dt += timedelta(days=1)

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

    if dataset == "events":
        qs = GameSession.objects.all().order_by("created_at")
        qs = qs.filter(phone_number__isnull=False)
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
        qs = TurnLog.objects.select_related("game_session").all().order_by("created_at")
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

