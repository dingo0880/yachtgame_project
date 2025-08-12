import csv

import json

from datetime import date, datetime

from django.core.management.base import BaseCommand

from django.utils import timezone

from yachtgame.models import TurnLog, GameSession

import pytz


class Command(BaseCommand):

    help = 'Exports all turn logs or turn logs for a specific date range.'


    def add_arguments(self, parser):

        parser.add_argument(

            '--start_date',

            type=str,

            help='Start date in YYYY-MM-DD format. Optional.',

        )

        parser.add_argument(

            '--end_date',

            type=str,

            help='End date in YYYY-MM-DD format. Optional.',

        )


    def handle(self, *args, **kwargs):

        start_date_str = kwargs['start_date']

        end_date_str = kwargs['end_date']


        logs = TurnLog.objects.select_related('game_session')


        if start_date_str and end_date_str:

            try:

                target_start_date = date.fromisoformat(start_date_str)

                target_end_date = date.fromisoformat(end_date_str)

                

                kst = pytz.timezone('Asia/Seoul')

                start_of_period_kst = kst.localize(datetime.combine(target_start_date, datetime.min.time()))

                end_of_period_kst = kst.localize(datetime.combine(target_end_date, datetime.max.time()))


                logs = logs.filter(created_at__gte=start_of_period_kst, created_at__lte=end_of_period_kst)


            except ValueError:

                self.stdout.write(self.style.ERROR('Invalid date format. Use YYYY-MM-DD.'))

                return

        

        logs = logs.order_by('created_at')

        

        if not logs:

            self.stdout.write(self.style.WARNING('No logs found for the specified period.'))

            return


        filename_prefix = 'yachtgame_turn_logs'

        if start_date_str and end_date_str:

            filename = f'{filename_prefix}_{start_date_str}_to_{end_date_str}.csv'

        else:

            filename = f'{filename_prefix}_all.csv'


        with open(filename, 'w', newline='', encoding='utf-8') as file:

            writer = csv.writer(file, quoting=csv.QUOTE_ALL)

            writer.writerow([

                'id', 'game_id', 'player_name', 'turn', 'score_state_before',

                'dice_roll_1', 'kept_after_roll_1', 'dice_roll_2', 'kept_after_roll_2',

                'final_dice_state', 'chosen_category', 'score_obtained', 'created_at'

            ])

            for log in logs:

                score_state_json = json.dumps(log.score_state_before).replace('"', "'")

                writer.writerow([

                    log.id,

                    log.game_session.game_id,

                    log.player_name,

                    log.turn_number,

                    score_state_json,

                    log.dice_roll_1,

                    log.kept_after_roll_1,

                    log.dice_roll_2,

                    log.kept_after_roll_2,

                    log.final_dice_state,

                    log.chosen_category,

                    log.score_obtained,

                    log.created_at.isoformat(),

                ])

        self.stdout.write(self.style.SUCCESS(f'Successfully exported logs to {filename}'))

