import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from pymongo import MongoClient

from predictions.models import PredictionResult


class Command(BaseCommand):
    help = 'Generate synthetic football prediction dataset (default 20000 rows) and save CSV and optionally to MongoDB.'

    def add_arguments(self, parser):
        parser.add_argument('--rows', type=int, default=20000)
        parser.add_argument('--csv', type=str, default='synthetic_predictions.csv')
        parser.add_argument('--to-mongo', action='store_true')
        parser.add_argument('--user-id', type=int, default=None, help='Associate records to this user id; auto-create if absent')

    def handle(self, *args, **options):
        num_rows = options['rows']
        csv_path = Path(options['csv']).resolve()
        to_mongo = options['to_mongo']
        user_id = options['user_id']

        User = get_user_model()
        user = None
        if user_id is not None:
            user = User.objects.filter(id=user_id).first()
        if user is None:
            # Create a synthetic user for data ownership
            username = 'synthetic_user'
            user, _ = User.objects.get_or_create(username=username, defaults={'email': 'synthetic@example.com'})
            if not user.password:
                user.set_password('synthetic_pass_123')
                user.save()

        labels = ['Win', 'Draw', 'Loss']
        header = [
            'home_team_strength', 'away_team_strength', 'home_advantage', 'recent_form_diff',
            'shots_on_target_diff', 'possession_diff', 'predicted_class', 'actual_class', 'created_at'
        ]

        rows = []
        base_time = datetime.utcnow() - timedelta(days=60)
        for i in range(num_rows):
            home = round(random.uniform(0, 1), 3)
            away = round(random.uniform(0, 1), 3)
            home_adv = random.choice([0, 1])
            form_diff = round(random.uniform(-1, 1), 3)
            shots_diff = random.randint(-10, 10)
            poss_diff = random.randint(-30, 30)

            # simple probabilistic rule for generating labels
            win_score = home + 0.2 * home_adv + 0.1 * max(form_diff, 0)
            loss_score = away + 0.1 * max(-form_diff, 0)
            draw_score = 0.3 + 0.1 * (1 - abs(form_diff))
            probs = [win_score, draw_score, loss_score]
            s = sum(probs)
            probs = [p / s for p in probs]
            predicted = random.choices(labels, weights=probs, k=1)[0]
            # Add noise so actual can differ
            actual = random.choices(labels, weights=[0.7 if l == predicted else 0.15 for l in labels], k=1)[0]

            created_at = (base_time + timedelta(minutes=i % (60 * 24))).isoformat()
            rows.append([
                home, away, home_adv, form_diff, shots_diff, poss_diff, predicted, actual, created_at
            ])

        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        self.stdout.write(self.style.SUCCESS(f'CSV written: {csv_path}'))

        # Save a subset into relational DB and optionally to Mongo for demo purposes
        objs = []
        mongo_docs = []
        for r in rows[:min(5000, num_rows)]:
            features = {
                'home_team_strength': r[0],
                'away_team_strength': r[1],
                'home_advantage': r[2],
                'recent_form_diff': r[3],
                'shots_on_target_diff': r[4],
                'possession_diff': r[5],
            }
            objs.append(PredictionResult(
                user=user,
                input_features=features,
                predicted_class=r[6],
                predicted_proba=None,
                actual_class=r[7],
            ))
            mongo_docs.append({
                'user_id': user.id,
                'input_features': features,
                'predicted_class': r[6],
                'predicted_proba': None,
                'actual_class': r[7],
                'created_at': r[8],
            })

        PredictionResult.objects.bulk_create(objs, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(f'Inserted {len(objs)} rows into SQL DB'))

        if to_mongo:
            try:
                client = MongoClient(settings.MONGODB_URI)
                db = client[settings.MONGODB_DB_NAME]
                coll = db[settings.MONGODB_COLLECTION]
                coll.insert_many(mongo_docs, ordered=False)
                self.stdout.write(self.style.SUCCESS(f'Inserted {len(mongo_docs)} docs into MongoDB'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Mongo insert skipped/failed: {e}'))

