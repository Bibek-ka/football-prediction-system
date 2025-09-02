from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),

    # Frontend pages
    path('', include('frontend.urls')),

    # User registration and authentication endpoints in accounts app
    path('api/auth/', include('accounts.urls')),

    # JWT token obtain and refresh endpoints
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Prediction app endpoints
    path('api/predictions/', include('predictions.urls')),

    # API status endpoint
    path('api/', lambda request: JsonResponse({
        'service': 'football-prediction-backend',
        'status': 'ok',
        'auth_endpoints': [
            '/api/auth/register/',
            '/api/auth/login/',
            '/api/auth/me/',
            '/api/auth/send-code/',
            '/api/auth/verify-code/',
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

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
