from rest_framework import serializers
from .models import PredictionResult


class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = (
            'id', 'user', 'input_features', 'predicted_class', 'predicted_proba',
            'actual_class', 'created_at'
        )
        read_only_fields = ('id', 'user', 'created_at')
