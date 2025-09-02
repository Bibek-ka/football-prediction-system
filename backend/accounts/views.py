from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model, authenticate
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from .serializers import RegisterSerializer, UserSerializer, LoginSerializer
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
            self.send_verification_code(user)

    def send_verification_code(self, user):
        """Send verification code to user's email"""
        code = f"{random.randint(0, 999999):06d}"
        expires_at = timezone.now() + timezone.timedelta(minutes=15)
        
        # Create new verification code
        EmailVerificationCode.objects.create(user=user, code=code, expires_at=expires_at)
        
        try:
            send_mail(
                subject="Football Prediction - Email Verification",
                message=f"""
Hello {user.username},

Thank you for registering with Football Match Prediction!

Your verification code is: {code}

This code will expire in 15 minutes.

If you didn't request this code, please ignore this email.

Best regards,
Football Prediction Team
                """.strip(),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )
        except Exception as e:
            print(f"Failed to send email: {e}")


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response(
                {"detail": "Username and password are required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Authenticate user
        user = authenticate(username=username, password=password)
        
        if not user:
            return Response(
                {"detail": "Invalid credentials"}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        # Generate JWT tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': UserSerializer(user).data,
            'email_verified': user.is_email_verified
        })


class MeView(APIView):
    def get(self, request):
        return Response(UserSerializer(request.user).data)


class SendVerificationCodeView(APIView):
    def post(self, request):
        user = request.user
        if not user.email:
            return Response({"detail": "Email not set on profile"}, status=400)
        
        # Use the same method as registration
        register_view = RegisterView()
        register_view.send_verification_code(user)
        
        return Response({"detail": "Verification code sent to your email"})


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
