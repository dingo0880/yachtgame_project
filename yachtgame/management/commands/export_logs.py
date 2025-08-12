import csv

import json

from datetime import date, datetime

from django.core.management.base import BaseCommand

from django.utils import timezone

from yachtgame.models import TurnLog

import pytz


class Command(BaseCommand):

    help = 'Exports turn logs for a specific date to a CSV file.'


    def add_arguments(self, parser):

        parser.add_argument(

            '--date',

            type=str,

            help='Date in YYYY-MM-DD format to export logs for.',

        )


    def handle(self, *args, **kwargs):

        date_str = kwargs.get('date')

        if not date_str:

            self.stdout.write(self.style.ERROR('You must specify a date with --date.'))

            return


        try:

            target_date = date.fromisoformat(date_str)

        except ValueError:

            self.stdout.write(self.style.ERROR('Invalid date format. Use YYYY-MM-DD.'))

            return


        # KST 기준 하루 범위

        tz = timezone.get_current_timezone()  # settings.TIME_ZONE 사용(예: Asia/Seoul)

        start_of_day = timezone.make_aware(datetime.combine(target_date, datetime.min.time()), tz)

        end_of_day = timezone.make_aware(datetime.combine(target_date, datetime.max.time()), tz)


        # game_session FK 미리 로딩

        logs = (

            TurnLog.objects

            .select_related('game_session')

            .filter(created_at__gte=start_of_day, created_at__lte=end_of_day)

            .order_by('created_at')

        )


        if not logs.exists():

            self.stdout.write(self.style.WARNING(f'No logs found for date {date_str}.'))

            return


        def to_csv_field(v):

            """CSV에 안전하게 넣기 위한 일관 문자열화."""

            if v is None:

                return ''

            # 주사위가 리스트/튜플이면 "1,2,3,4,5"로

            if isinstance(v, (list, tuple)):

                return ",".join(map(str, v))

            # dict/JSON은 정식 JSON 문자열로 (따옴표 변환 금지!)

            if isinstance(v, dict):

                return json.dumps(v, ensure_ascii=False, separators=(',', ':'))

            # 이미 문자열인 경우도 그대로 사용(앞뒤 공백만 정리)

            s = str(v).strip()

            return s


        filename = f'yachtgame_logs_{date_str}.csv'

        with open(filename, 'w', newline='', encoding='utf-8') as file:

            writer = csv.writer(

                file,

                quoting=csv.QUOTE_ALL,   # 모든 필드를 따옴표로 감싸기

                escapechar='\\',         # 내부 따옴표/구분자 이스케이프

                lineterminator='\n'

            )

            writer.writerow([

                'id', 'game_id', 'player_name', 'turn', 'score_state_before',

                'dice_roll_1', 'kept_after_roll_1', 'dice_roll_2', 'kept_after_roll_2',

                'final_dice_state', 'chosen_category', 'score_obtained', 'created_at'

            ])


            for log in logs:

                # 모델 필드 타입이 str/JSONField/Array 등 무엇이든 to_csv_field로 일원화

                row = [

                    to_csv_field(log.id),

                    to_csv_field(getattr(log.game_session, 'game_id', '')),

                    to_csv_field(log.player_name),

                    to_csv_field(log.turn_number),

                    to_csv_field(log.score_state_before),   # JSON은 dumps로 안전 출력

                    to_csv_field(log.dice_roll_1),

                    to_csv_field(log.kept_after_roll_1),

                    to_csv_field(log.dice_roll_2),

                    to_csv_field(log.kept_after_roll_2),

                    to_csv_field(log.final_dice_state),

                    to_csv_field(log.chosen_category),

                    to_csv_field(log.score_obtained),

                    to_csv_field(log.created_at.isoformat()),

                ]

                writer.writerow(row)


        self.stdout.write(self.style.SUCCESS(f'Successfully exported logs to {filename}'))

