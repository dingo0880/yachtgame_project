import csv

from datetime import date, datetime

from django.core.management.base import BaseCommand

from django.utils import timezone

from yachtgame.models import GameSession

import pytz


class Command(BaseCommand):

    help = 'Exports event participant logs for a date range to a CSV file.'


    def add_arguments(self, parser):

        parser.add_argument(

            '--start_date',

            type=str,

            help='Start date in YYYY-MM-DD format.',

        )

        parser.add_argument(

            '--end_date',

            type=str,

            help='End date in YYYY-MM-DD format.',

        )


    def handle(self, *args, **kwargs):

        start_date_str = kwargs['start_date']

        end_date_str = kwargs['end_date']


        if not start_date_str or not end_date_str:

            self.stdout.write(self.style.ERROR('You must specify both start and end dates.'))

            return


        try:

            target_start_date = date.fromisoformat(start_date_str)

            target_end_date = date.fromisoformat(end_date_str)

            

            kst = pytz.timezone('Asia/Seoul')

            start_of_period_kst = kst.localize(datetime.combine(target_start_date, datetime.min.time()))

            end_of_period_kst = kst.localize(datetime.combine(target_end_date, datetime.max.time()))


        except ValueError:

            self.stdout.write(self.style.ERROR('Invalid date format. Use YYYY-MM-DD.'))

            return


        logs = GameSession.objects.filter(

            phone_number__isnull=False,  # 전화번호가 있는 이벤트 참여자만 필터링

            created_at__gte=start_of_period_kst,

            created_at__lte=end_of_period_kst

        ).order_by('-total_score')


        if not logs:

            self.stdout.write(self.style.WARNING(f'No event participants found for the period {start_date_str} to {end_date_str}.'))

            return


        filename = f'yachtgame_event_logs_{start_date_str}_to_{end_date_str}.csv'

        with open(filename, 'w', newline='', encoding='utf-8') as file:

            writer = csv.writer(file, quoting=csv.QUOTE_ALL)

            writer.writerow([

                'id', 'game_id', 'player_name', 'total_score', 'phone_number', 'created_at', 'ip_address'

            ])

            for log in logs:

                writer.writerow([

                    log.id,

                    log.game_id,

                    log.player_name,

                    log.total_score,

                    log.phone_number,

                    log.created_at.isoformat(),

                    log.ip_address,

                ])

        self.stdout.write(self.style.SUCCESS(f'Successfully exported event logs to {filename}'))


