from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.mail import send_mail
from .serializers import RegisterSerializer, UserSerializer
from .models import EmailVerificationCode
import random


from django.shortcuts import render
from django.http import JsonResponse

def load_synthetic_dataset(request):
    
    data = {"message": "Synthetic dataset loaded successfully."}
    return JsonResponse(data)


User = get_user_model()


class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

    def perform_create(self, serializer):
        user = serializer.save()
        # Auto-send verification code if email present
        if user.email:
            code = f"{random.randint(0, 999999):06d}"
            expires_at = timezone.now() + timezone.timedelta(minutes=15)
            EmailVerificationCode.objects.create(user=user, code=code, expires_at=expires_at)
            try:
                send_mail(
                    subject="Your Verification Code",
                    message=f"Your code is: {code}. It expires in 15 minutes.",
                    from_email=None,
                    recipient_list=[user.email],
                    fail_silently=True,
                )
            except Exception:
                pass


class MeView(APIView):
    def get(self, request):
        return Response(UserSerializer(request.user).data)


class SendVerificationCodeView(APIView):
    def post(self, request):
        user = request.user
        if not user.email:
            return Response({"detail": "Email not set on profile"}, status=400)
        code = f"{random.randint(0, 999999):06d}"
        expires_at = timezone.now() + timezone.timedelta(minutes=15)
        EmailVerificationCode.objects.create(user=user, code=code, expires_at=expires_at)
        send_mail(
            subject="Your Verification Code",
            message=f"Your code is: {code}. It expires in 15 minutes.",
            from_email=None,
            recipient_list=[user.email],
            fail_silently=False,
        )
        return Response({"detail": "Verification code sent"})


class VerifyCodeView(APIView):
    def post(self, request):
        code = request.data.get('code')
        if not code:
            return Response({"detail": "Code required"}, status=400)
        q = EmailVerificationCode.objects.filter(user=request.user, code=code, used=False).order_by('-created_at').first()
        if not q:
            return Response({"detail": "Invalid code"}, status=400)
        if timezone.now() > q.expires_at:
            return Response({"detail": "Code expired"}, status=400)
        q.used = True
        q.save()
        user = request.user
        user.is_email_verified = True
        user.save(update_fields=["is_email_verified"])
        return Response({"detail": "Email verified"})

# Create your views here.
