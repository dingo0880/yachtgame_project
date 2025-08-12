# yachtgame/models.py

from django.db import models
from django.db.models import UniqueConstraint

class GameSession(models.Model):
    game_id = models.CharField(max_length=255, unique=True)
    player_name = models.CharField(max_length=255)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    total_score = models.IntegerField(null=True, blank=True)
    # 전화번호는 해시된 값(SHA-256)을 저장하므로 길이를 넉넉하게 64자로 설정합니다.
    phone_number = models.CharField(max_length=64, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.player_name} ({self.game_id})"

class TurnLog(models.Model):
    game_session = models.ForeignKey(GameSession, on_delete=models.CASCADE, related_name='turns')
    player_name = models.CharField(max_length=255)
    turn_number = models.IntegerField()
    
    # 기존 필드는 그대로 유지합니다.
    dice_roll_1 = models.CharField(max_length=50, null=True, blank=True)
    kept_after_roll_1 = models.CharField(max_length=50, null=True, blank=True)
    dice_roll_2 = models.CharField(max_length=50, null=True, blank=True)
    kept_after_roll_2 = models.CharField(max_length=50, null=True, blank=True)
    final_dice_state = models.CharField(max_length=50)
    chosen_category = models.CharField(max_length=50)
    score_obtained = models.IntegerField()
    score_state_before = models.JSONField(default=dict) # JSONField가 없다면 TextField 사용
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 데이터 무결성을 위한 DB 제약 조건 추가
        constraints = [
            # 1. 한 게임 세션 내에서 같은 턴 번호는 한 번만 기록될 수 있습니다.
            UniqueConstraint(fields=['game_session', 'turn_number', 'player_name'], name='unique_turn_per_game'),
            
            # 2. 한 게임 세션 내에서 같은 플레이어는 동일한 족보를 한 번만 선택할 수 있습니다.
            UniqueConstraint(fields=['game_session', 'player_name', 'chosen_category'], name='unique_category_per_game')
        ]
        ordering = ['created_at']

    def __str__(self):
        return f"Turn {self.turn_number} for {self.player_name} in {self.game_session.game_id}"
