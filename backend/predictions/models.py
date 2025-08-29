from django.db import models
from django.conf import settings
from django.utils import timezone


class PredictionResult(models.Model):
    CLASS_WIN = 'Win'
    CLASS_DRAW = 'Draw'
    CLASS_LOSS = 'Loss'

    CLASS_CHOICES = [
        (CLASS_WIN, 'Win'),
        (CLASS_DRAW, 'Draw'),
        (CLASS_LOSS, 'Loss'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='predictions')
    input_features = models.JSONField()
    predicted_class = models.CharField(max_length=8, choices=CLASS_CHOICES)
    predicted_proba = models.JSONField(null=True, blank=True)
    actual_class = models.CharField(max_length=8, choices=CLASS_CHOICES, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self) -> str:
        return f"Prediction {self.id} by {self.user_id}: {self.predicted_class}"


class ModelMetrics(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    train_accuracy = models.FloatField(null=True, blank=True)
    test_accuracy = models.FloatField(null=True, blank=True)

    def __str__(self) -> str:
        return f"Metrics at {self.created_at}: train={self.train_accuracy}, test={self.test_accuracy}"

# Create your models here.
