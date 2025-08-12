# yachtgame/urls.py
from django.urls import path
from . import views
from . import views_ml  # ★ 추가: ML 전용 뷰

urlpatterns = [
    # --- 메인/게임 API들 ---
    path('', views.index, name='index'),
    path('api/start-game/', views.start_game_api, name='start_game_api'),
    path('api/roll-dice/', views.roll_dice_api, name='roll_dice_api'),
    path('api/keep-dice/', views.keep_dice_api, name='keep_dice_api'),
    path('api/select-category/', views.select_category_api, name='select_category_api'),
    path('api/get-game-state/', views.get_game_state_api, name='get_game_state_api'),

    path('api/play-cpu-turn/', views.play_cpu_turn_api, name='play_cpu_turn_api'),
    path('api/analyze-cpu/', views.analyze_cpu_api, name='analyze_cpu_api'),

    path('api/save-event-entry/', views.save_event_entry_api, name='save_event_entry_api'),
    path('api/get-all-logs/', views.get_all_logs_api, name='get_all_logs_api'),

    path('api/get-hall-of-fame/', views.get_hall_of_fame, name='get_hall_of_fame'),
    path('api/get-weekly-high-scores/', views.get_weekly_high_scores, name='get_weekly_high_scores'),

    path('notice/', views.get_notice, name='get_notice'),
    path('patch_notes/', views.get_patch_notes, name='get_patch_notes'),

    # ✅ 개발자용 CSV 다운로드
    path('api/dev/export/', views.export_logs_csv, name='export_logs_csv'),

    # ✅ ML 엔드포인트(프론트 index가 호출하는 경로)
    path('api/ml/keep', views_ml.ml_keep_decision, name='ml_keep_decision'),
    path('api/ml/category', views_ml.ml_category_decision, name='ml_category_decision'),
    path('api/ml/health', views_ml.ml_health, name='ml_health'),
]
