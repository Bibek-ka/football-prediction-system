from django.contrib import admin
from .models import PredictionResult


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "predicted_class", "actual_class", "created_at")
    list_filter = ("predicted_class", "actual_class", "created_at")
    search_fields = ("user__username",)

# Register your models here.
