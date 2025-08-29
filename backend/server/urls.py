from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),

    # User registration and authentication endpoints in accounts app
    path('api/auth/', include('accounts.urls')),

    # JWT token obtain and refresh endpoints
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Prediction app endpoints
    path('api/predictions/', include('predictions.urls')),

    # Root endpoint showing service status and available endpoints
    path('', lambda request: JsonResponse({
        'service': 'football-prediction-backend',
        'status': 'ok',
        'auth_endpoints': [
            '/api/auth/register/',
            '/api/token/',
            '/api/token/refresh/',
        ],
        'prediction_endpoints': [
            '/api/predictions/predict/',
            '/api/predictions/upload-csv/',
            '/api/predictions/metrics/',
        ],
    })),
]
