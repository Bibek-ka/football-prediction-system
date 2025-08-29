from django.urls import path
from .views import PredictView, UploadCSVView, MetricsView, DashboardView, TrainModelView, LoadSyntheticView


urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('upload-csv/', UploadCSVView.as_view(), name='upload_csv'),
    path('metrics/', MetricsView.as_view(), name='metrics'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('train/', TrainModelView.as_view(), name='train_model'),
    path('load-synthetic/', LoadSyntheticView.as_view(), name='load_synthetic'),
]
