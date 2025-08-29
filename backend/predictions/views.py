import io
import json
import joblib
import pandas as pd
from django.conf import settings
from django.core.files.storage import default_storage
from pymongo import MongoClient
from rest_framework import permissions, throttling
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import PredictionResult, ModelMetrics
from .serializers import PredictionResultSerializer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from django.db.models import Count
from django.utils import timezone


class PredictionsRateThrottle(throttling.UserRateThrottle):
    scope = 'predictions'


def load_model():
    model_path = settings.BASE_DIR / 'model.pkl'
    try:
        return joblib.load(model_path)
    except Exception:
        return None


class PredictView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [PredictionsRateThrottle]

    def post(self, request):
        model = load_model()
        if model is None:
            return Response({"detail": "Model not available"}, status=503)

        payload = request.data if isinstance(request.data, dict) else json.loads(request.body or '{}')
        features = payload.get('features')
        if features is None:
            return Response({"detail": "Missing 'features' in request"}, status=400)

        df = pd.DataFrame([features])
        proba = None
        if hasattr(model, 'predict_proba'):
            proba_vals = model.predict_proba(df)[0]
            classes = getattr(model, 'classes_', ['Win', 'Draw', 'Loss'])
            proba = {str(cls): float(p) for cls, p in zip(classes, proba_vals)}
        pred = model.predict(df)[0]

        result = PredictionResult.objects.create(
            user=request.user,
            input_features=features,
            predicted_class=str(pred),
            predicted_proba=proba,
        )
        # Mirror to MongoDB (best-effort)
        try:
            client = MongoClient(settings.MONGODB_URI)
            db = client[settings.MONGODB_DB_NAME]
            coll = db[settings.MONGODB_COLLECTION]
            coll.insert_one({
                'user_id': request.user.id,
                'input_features': features,
                'predicted_class': str(pred),
                'predicted_proba': proba,
                'actual_class': None,
                'created_at': result.created_at.isoformat(),
            })
        except Exception:
            pass
        return Response(PredictionResultSerializer(result).data)


class UploadCSVView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"detail": "No file uploaded"}, status=400)

        model = load_model()
        if model is None:
            return Response({"detail": "Model not available"}, status=503)

        df = pd.read_csv(file)
        target_col = request.data.get('target')
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y_true = df[target_col]
        else:
            X = df
            y_true = None

        y_pred = model.predict(X)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            classes = getattr(model, 'classes_', ['Win', 'Draw', 'Loss'])

        created = []
        for idx, features in X.iterrows():
            proba_map = None
            if proba is not None:
                proba_map = {str(cls): float(p) for cls, p in zip(classes, proba[idx])}
            actual_label = None if y_true is None else str(y_true.iloc[idx])
            pr = PredictionResult.objects.create(
                user=request.user,
                input_features=features.to_dict(),
                predicted_class=str(y_pred[idx]),
                predicted_proba=proba_map,
                actual_class=actual_label,
            )
            created.append(pr)
        # Mirror to MongoDB (bulk)
        try:
            client = MongoClient(settings.MONGODB_URI)
            db = client[settings.MONGODB_DB_NAME]
            coll = db[settings.MONGODB_COLLECTION]
            docs = [{
                'user_id': request.user.id,
                'input_features': obj.input_features,
                'predicted_class': obj.predicted_class,
                'predicted_proba': obj.predicted_proba,
                'actual_class': obj.actual_class,
                'created_at': obj.created_at.isoformat(),
            } for obj in created]
            if docs:
                coll.insert_many(docs)
        except Exception:
            pass

        return Response(PredictionResultSerializer(created, many=True).data)


class MetricsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        # Metrics computed from stored results where actual_class is not null
        qs = PredictionResult.objects.exclude(actual_class__isnull=True)
        if not qs.exists():
            return Response({"detail": "No labeled predictions to compute metrics"}, status=400)

        y_true = [r.actual_class for r in qs]
        y_pred = [r.predicted_class for r in qs]

        labels = ['Win', 'Draw', 'Loss']
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        # TP/TN/FP/FN per class (one-vs-all)
        total = int(cm.sum())
        per_class_counts = {}
        for i, lbl in enumerate(labels):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            tn = int(total - tp - fp - fn)
            per_class_counts[lbl] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # ROC (macro-average) if we have probabilities stored
        roc_data = None
        has_proba = all(r.predicted_proba for r in qs)
        if has_proba:
            y_true_bin = label_binarize(y_true, classes=labels)
            # Build probability matrix aligned with labels
            proba_matrix = []
            for r in qs:
                row = [float(r.predicted_proba.get(lbl, 0.0)) for lbl in labels]
                proba_matrix.append(row)
            proba_df = pd.DataFrame(proba_matrix, columns=labels)

            fpr = {}
            tpr = {}
            roc_auc = {}
            for i, lbl in enumerate(labels):
                fpr[lbl], tpr[lbl], _ = roc_curve(y_true_bin[:, i], proba_df[lbl])
                roc_auc[lbl] = auc(fpr[lbl], tpr[lbl])
            # Macro average
            all_fpr = pd.unique(pd.concat([pd.Series(fpr[lbl]) for lbl in labels]).sort_values())
            mean_tpr = None
            # Compute mean TPR across classes at common FPR points
            interp_tprs = []
            for lbl in labels:
                s_fpr = pd.Series(fpr[lbl])
                s_tpr = pd.Series(tpr[lbl])
                interp = pd.Series(index=all_fpr, dtype=float)
                interp = interp.combine_first(s_tpr.reindex(all_fpr, method='nearest'))
                interp_tprs.append(interp.fillna(method='ffill').fillna(0.0))
            mean_tpr = pd.concat(interp_tprs, axis=1).mean(axis=1)
            macro_auc = auc(all_fpr, mean_tpr)

            roc_data = {
                'per_class': {lbl: {"fpr": fpr[lbl].tolist(), "tpr": tpr[lbl].tolist(), "auc": roc_auc[lbl]} for lbl in labels},
                'macro': {"fpr": all_fpr.tolist(), "tpr": mean_tpr.tolist(), "auc": macro_auc},
            }

        # Prediction distribution
        pred_counts = pd.Series(y_pred).value_counts().to_dict()
        actual_counts = pd.Series(y_true).value_counts().to_dict()

        latest = ModelMetrics.objects.order_by('-created_at').first()
        data = {
            'accuracy': acc,
            'train_accuracy': latest.train_accuracy if latest else acc,
            'test_accuracy': latest.test_accuracy if latest else acc,
            'confusion_matrix': {
                'labels': labels,
                'matrix': cm.tolist(),
            },
            'classification_report': report,
            'per_class_counts': per_class_counts,
            'roc': roc_data,
            'distribution': {
                'predicted': pred_counts,
                'actual': actual_counts,
            }
        }

        return Response(data)


class DashboardView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        # Overall counts
        total_predictions = PredictionResult.objects.count()
        labeled_predictions = PredictionResult.objects.exclude(actual_class__isnull=True).count()

        # Daily prediction counts (last 14 days)
        today = timezone.now().date()
        start_date = today - timezone.timedelta(days=13)
        qs = (
            PredictionResult.objects
            .filter(created_at__date__gte=start_date)
            .extra(select={"day": "date(created_at)"})
            .values("day")
            .annotate(count=Count("id"))
            .order_by("day")
        )
        daily = []
        counts_map = {str(row["day"]): row["count"] for row in qs}
        for i in range(14):
            d = start_date + timezone.timedelta(days=i)
            daily.append({"date": str(d), "count": counts_map.get(str(d), 0)})

        # Distribution of predicted/actual
        preds = PredictionResult.objects.values("predicted_class").annotate(c=Count("id"))
        acts = (
            PredictionResult.objects.exclude(actual_class__isnull=True)
            .values("actual_class").annotate(c=Count("id"))
        )
        pred_dist = {p["predicted_class"] or "Unknown": p["c"] for p in preds}
        act_dist = {a["actual_class"] or "Unknown": a["c"] for a in acts}

        # Accuracy over labeled
        acc = None
        labeled_q = PredictionResult.objects.exclude(actual_class__isnull=True)
        if labeled_q.exists():
            y_true = [r.actual_class for r in labeled_q]
            y_pred = [r.predicted_class for r in labeled_q]
            acc = accuracy_score(y_true, y_pred)

        data = {
            "totals": {
                "predictions": total_predictions,
                "labeled": labeled_predictions,
            },
            "daily_counts": daily,
            "distribution": {
                "predicted": pred_dist,
                "actual": act_dist,
            },
            "labeled_accuracy": acc,
        }

        return Response(data)


class TrainModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        # Expect CSV file upload with target column name
        file = request.FILES.get('file')
        target_col = request.data.get('target') or 'actual_class'
        if not file:
            return Response({"detail": "CSV file is required"}, status=400)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return Response({"detail": f"Failed to read CSV: {e}"}, status=400)

        if target_col not in df.columns:
            return Response({"detail": f"Target column '{target_col}' not found"}, status=400)

        X = df.drop(columns=[target_col])
        y = df[target_col].astype(str)
        # Basic cleaning: drop non-numeric columns except we attempt to one-hot
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = float(clf.score(X_train, y_train))
        test_acc = float(clf.score(X_test, y_test))

        # Save model
        joblib.dump(clf, settings.BASE_DIR / 'model.pkl')
        ModelMetrics.objects.create(train_accuracy=train_acc, test_accuracy=test_acc)

        return Response({"detail": "Model trained", "train_accuracy": train_acc, "test_accuracy": test_acc})


class LoadSyntheticView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Load a sample dataset from backend/synthetic_predictions.csv into PredictionResult.
        Only imports up to 2000 rows to avoid duplicates/overload per call.
        """
        csv_path = settings.BASE_DIR / 'synthetic_predictions.csv'
        if not csv_path.exists():
            return Response({"detail": str(csv_path), "message": "synthetic_predictions.csv not found"}, status=404)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return Response({"detail": f"Failed to read CSV: {e}"}, status=400)

        required_cols = {
            'home_team_strength', 'away_team_strength', 'home_advantage', 'recent_form_diff',
            'shots_on_target_diff', 'possession_diff', 'predicted_class', 'actual_class'
        }
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return Response({"detail": f"CSV missing columns: {', '.join(missing)}"}, status=400)

        subset = df.head(2000)
        created = []
        for _, row in subset.iterrows():
            features = {
                'home_team_strength': float(row['home_team_strength']),
                'away_team_strength': float(row['away_team_strength']),
                'home_advantage': int(row['home_advantage']),
                'recent_form_diff': float(row['recent_form_diff']),
                'shots_on_target_diff': int(row['shots_on_target_diff']),
                'possession_diff': int(row['possession_diff']),
            }
            pr = PredictionResult.objects.create(
                user=request.user,
                input_features=features,
                predicted_class=str(row['predicted_class']),
                predicted_proba=None,
                actual_class=str(row['actual_class']) if not pd.isna(row['actual_class']) else None,
            )
            created.append(pr.id)

        return Response({"detail": f"Loaded {len(created)} rows", "path": str(csv_path)})



from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def metrics_view(request):
    # Dummy example metrics
    metrics = {
        'train_accuracy': 0.92,
        'test_accuracy': 0.88,
        'accuracy': 0.88,
        'confusion_matrix': {
            'labels': ['Win', 'Draw', 'Loss'],
            'matrix': [
                [30, 5, 2],
                [3, 22, 4],
                [1, 7, 25]
            ]
        },
        'classification_report': {
            'Win': {'precision': 0.8, 'recall': 0.75, 'f1_score': 0.77},
            'Draw': {'precision': 0.7, 'recall': 0.65, 'f1_score': 0.67},
            'Loss': {'precision': 0.85, 'recall': 0.9, 'f1_score': 0.87},
        },
        'per_class_counts': {
            'Win': {'tp': 30, 'tn': 50, 'fp': 10, 'fn': 5},
            'Draw': {'tp': 22, 'tn': 60, 'fp': 12, 'fn': 8},
            'Loss': {'tp': 25, 'tn': 70, 'fp': 5, 'fn': 2},
        },
        'roc': {
            'macro': {
                'fpr': [0, 0.1, 0.2, 1],
                'tpr': [0, 0.7, 0.85, 1],
                'auc': 0.9
            }
        },
        'distribution': {
            'predicted': {'Win': 60, 'Draw': 40, 'Loss': 50},
            'actual': {'Win': 55, 'Draw': 45, 'Loss': 50}
        }
    }
    return Response(metrics)

# Create your views here.
