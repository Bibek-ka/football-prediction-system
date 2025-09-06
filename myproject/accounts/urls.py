from .views import signup_view, login_view, logout_view, edit_profile, prediction_view, fixtures_view, table_view, analysis_view, dashboard, performance_view
from django.urls import path
from . import views

urlpatterns = [
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('prediction/', prediction_view, name='prediction'),
    path('edit-profile/', edit_profile, name='edit_profile'),
     path('fixtures/', views.fixtures_view, name='fixtures'),
     path('table/', views.table_view, name='table'),
     path('analysis/', analysis_view, name='analysis'),
     path("performance/", performance_view, name="performance"),
]
