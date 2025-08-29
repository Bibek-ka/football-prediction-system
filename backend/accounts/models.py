from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    ROLE_ADMIN = 'admin'
    ROLE_ANALYST = 'analyst'
    ROLE_USER = 'user'

    ROLE_CHOICES = [
        (ROLE_ADMIN, 'Admin'),
        (ROLE_ANALYST, 'Analyst'),
        (ROLE_USER, 'User'),
    ]

    role = models.CharField(max_length=16, choices=ROLE_CHOICES, default=ROLE_USER)
    is_email_verified = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.username} ({self.role})"


class EmailVerificationCode(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='email_codes')
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    used = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.user_id}:{self.code} (used={self.used})"

# Create your models here.
