# yachtgame/urls.py
from django.urls import path
from . import views
from .ml_views import ml_keep_decision, ml_category_decision, ml_health

urlpatterns = [
    path('', views.index, name='index'),

    # --- 게임 API ---
    path('api/start-game/', views.start_game_api, name='start_game_api'),
    path('api/roll-dice/', views.roll_dice_api, name='roll_dice_api'),
    path('api/keep-dice/', views.keep_dice_api, name='keep_dice_api'),
    path('api/select-category/', views.select_category_api, name='select_category_api'),
    path('api/get-game-state/', views.get_game_state_api, name='get_game_state_api'),
    path('api/play-cpu-turn/', views.play_cpu_turn_api, name='play_cpu_turn_api'),
    path('api/analyze-cpu/', views.analyze_cpu_api, name='analyze_cpu_api'),
    path('api/collect-cpu-logs/', views.collect_cpu_logs_api, name='collect_cpu_logs_api'),

    # --- 이벤트/로그/랭킹 ---
    # path('api/save-event-entry/', views.save_event_entry_api, name='save_event_entry_api'), # 1. 이벤트 비활성화: URL 경로도 주석 처리
    path('api/get-all-logs/', views.get_all_logs_api, name='get_all_logs_api'),
    path('api/get-hall-of-fame/', views.get_hall_of_fame, name='get_hall_of_fame'),
    path('api/get-weekly-high-scores/', views.get_weekly_high_scores, name='get_weekly_high_scores'),
    path('notice/', views.get_notice, name='get_notice'),
    path('patch_notes/', views.get_patch_notes, name='get_patch_notes'),

    # --- ML 엔드포인트 ---
    path('api/ml/keep', ml_keep_decision, name='ml_keep_decision'),
    path('api/ml/category', ml_category_decision, name='ml_category_decision'),
    path('api/ml/health', ml_health, name='ml_health'),
    path('api/cpu/select-category/', views.select_category_cpu_api, name='select_category_cpu_api'),

    # --- 개발자 CSV ---
    path('api/dev/export/', views.export_logs_csv, name='export_logs_csv'),
]

