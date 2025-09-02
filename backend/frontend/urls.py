from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('profile/', views.profile_view, name='profile'),
    
    # Football related pages
    path('fixtures/', views.fixtures_view, name='fixtures'),
    path('table/', views.table_view, name='table'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('prediction/', views.prediction_view, name='prediction'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # Other pages
    path('forgot-password/', views.forgot_password_view, name='forgot-password'),
    path('debug/', views.debug_view, name='debug'),
]
