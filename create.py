import os

# Folder structure
folders = [
    "myproject",
    "myproject/accounts",
    "myproject/templates",
    "myproject/templates/accounts",
    "myproject/myproject"
]

# Files content
files = {
    "myproject/manage.py": """#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Couldn't import Django.") from exc
    execute_from_command_line(sys.argv)
""",

    "myproject/myproject/__init__.py": "",
    "myproject/myproject/settings.py": """import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET_KEY = 'django-insecure-123456'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'accounts',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR,'templates')],
        'APP_DIRS': True,
        'OPTIONS': {'context_processors': [
            'django.template.context_processors.debug',
            'django.template.context_processors.request',
            'django.contrib.auth.context_processors.auth',
            'django.contrib.messages.context_processors.messages',
        ],},
    },
]

WSGI_APPLICATION = 'myproject.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True
STATIC_URL = '/static/'
""",

    "myproject/myproject/urls.py": """from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda request: render(request,'index.html'), name='home'),
    path('accounts/', include('accounts.urls')),
]
""",

    "myproject/myproject/wsgi.py": """import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
application = get_wsgi_application()
""",

    "myproject/accounts/__init__.py": "",
    "myproject/accounts/admin.py": "",
    "myproject/accounts/apps.py": """from django.apps import AppConfig
class AccountsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'accounts'
""",
    "myproject/accounts/models.py": "",
    "myproject/accounts/forms.py": """from django import forms
from django.contrib.auth.models import User

class SignUpForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username','email','password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")
        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match")
""",
    "myproject/accounts/views.py": """from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignUpForm

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request,'Account created successfully!')
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request,'accounts/signup.html',{'form':form})

def login_view(request):
    if request.method=='POST':
        form = AuthenticationForm(request,data=request.POST)
        if form.is_valid():
            username=form.cleaned_data.get('username')
            password=form.cleaned_data.get('password')
            user=authenticate(username=username,password=password)
            if user:
                login(request,user)
                return redirect('dashboard')
        messages.error(request,'Invalid username or password')
    else:
        form = AuthenticationForm()
    return render(request,'accounts/login.html',{'form':form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def dashboard_view(request):
    return render(request,'dashboard.html',{'user':request.user})
""",

    "myproject/accounts/urls.py": """from django.urls import path
from .views import signup_view, login_view, logout_view, dashboard_view

urlpatterns = [
    path('signup/',signup_view,name='signup'),
    path('login/',login_view,name='login'),
    path('logout/',logout_view,name='logout'),
    path('dashboard/',dashboard_view,name='dashboard'),
]
""",

    "myproject/templates/base.html": """<!DOCTYPE html>
<html>
<head>
<title>{% block title %}My Project{% endblock %}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
.circle-avatar {
    width:40px;height:40px;border-radius:50%;background:#007bff;color:white;
    display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:14px;
}
</style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-light px-3">
<a class="navbar-brand" href="/">My Project</a>
<div class="ms-auto">
{% if user.is_authenticated %}
<div class="d-flex align-items-center">
<div class="circle-avatar me-2">{{ user.email|slice:":1"|upper }}</div>
<span>{{ user.email }}</span>
<a href="{% url 'logout' %}" class="btn btn-sm btn-outline-danger ms-3">Logout</a>
</div>
{% else %}
<a href="{% url 'login' %}" class="btn btn-outline-primary me-2">Login</a>
<a href="{% url 'signup' %}" class="btn btn-primary">Sign Up</a>
{% endif %}
</div>
</nav>
<div class="container mt-4">{% block content %}{% endblock %}</div>
</body>
</html>
""",

    "myproject/templates/index.html": """{% extends "base.html" %}
{% block title %}Home{% endblock %}
{% block content %}
<h2>Welcome to My Project</h2>
<p>This is the home page. Please login or sign up.</p>
{% endblock %}
""",

    "myproject/templates/dashboard.html": """{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<h2>Welcome, {{ user.username }}!</h2>
<p>Email: {{ user.email }}</p>
<p>This is your dashboard page.</p>
{% endblock %}
""",

    "myproject/templates/accounts/signup.html": """{% extends "base.html" %}
{% block title %}Sign Up{% endblock %}
{% block content %}
<h2>Sign Up</h2>
<form method="post">{% csrf_token %}{{ form.as_p }}
<button type="submit" class="btn btn-primary">Sign Up</button>
</form>
<p class="mt-2">Already have an account? <a href="{% url 'login' %}">Login here</a></p>
{% endblock %}
""",

    "myproject/templates/accounts/login.html": """{% extends "base.html" %}
{% block title %}Login{% endblock %}
{% block content %}
<h2>Login</h2>
<form method="post">{% csrf_token %}{{ form.as_p }}
<button type="submit" class="btn btn-primary">Login</button>
</form>
<p class="mt-2">Don't have an account? <a href="{% url 'signup' %}">Sign Up here</a></p>
{% endblock %}
"""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("Django project files created successfully! You can now run migrations and start the server.")
