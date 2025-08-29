from django.urls import path
from .views import RegisterView, MeView, SendVerificationCodeView, VerifyCodeView


urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('me/', MeView.as_view(), name='me'),
    path('send-code/', SendVerificationCodeView.as_view(), name='send_code'),
    path('verify-code/', VerifyCodeView.as_view(), name='verify_code'),
]
from django.urls import path
from .views import load_synthetic_dataset

urlpatterns = [
    path('api/predictions/load-synthetic/', load_synthetic_dataset),
]

