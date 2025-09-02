from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth import get_user_model

User = get_user_model()

def index(request):
    """Home page view"""
    return render(request, 'frontend/index.html')

def login_view(request):
    """Login page view"""
    return render(request, 'frontend/login.html')

def signup_view(request):
    """Signup page view"""
    return render(request, 'frontend/signup.html')

@login_required
def profile_view(request):
    """Profile page view"""
    return render(request, 'frontend/profile.html')

def fixtures_view(request):
    """Fixtures page view"""
    return render(request, 'frontend/fixtures.html')

def table_view(request):
    """Table page view"""
    return render(request, 'frontend/table.html')

def analysis_view(request):
    """Analysis page view"""
    return render(request, 'frontend/analysis.html')

@login_required
def prediction_view(request):
    """Prediction page view"""
    return render(request, 'frontend/prediction.html')

@login_required
def dashboard_view(request):
    """Dashboard page view"""
    return render(request, 'frontend/dashboard.html')

def forgot_password_view(request):
    """Forgot password page view"""
    return render(request, 'frontend/forgot-password.html')

def debug_view(request):
    """Debug page view"""
    return render(request, 'frontend/debug.html')