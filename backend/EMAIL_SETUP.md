# Email Configuration Guide

## Setting up Email Service for Verification Codes

To enable email verification functionality, you need to configure email settings in your Django application.

### Option 1: Gmail SMTP (Recommended for Development)

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate an App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate a new app password for "Mail"
3. **Set Environment Variables**:
   ```bash
   export EMAIL_HOST_USER="your-email@gmail.com"
   export EMAIL_HOST_PASSWORD="your-16-character-app-password"
   ```

### Option 2: Console Backend (Development Only)

For development/testing, emails will be printed to the console instead of being sent:

```bash
# No environment variables needed - emails will show in terminal
python manage.py runserver
```

### Option 3: Other Email Providers

You can use other SMTP providers by setting these environment variables:

```bash
export EMAIL_HOST="smtp.your-provider.com"
export EMAIL_PORT="587"
export EMAIL_HOST_USER="your-email@domain.com"
export EMAIL_HOST_PASSWORD="your-password"
export EMAIL_USE_TLS="True"
```

### Testing Email Configuration

1. Start your Django server: `python manage.py runserver`
2. Register a new user through the signup form
3. Check your email inbox (or console output) for the verification code

### Troubleshooting

- **"Authentication failed"**: Check your email credentials and app password
- **"Connection refused"**: Verify SMTP host and port settings
- **No emails received**: Check spam folder or use console backend for testing

### Production Deployment

For production, consider using:
- **SendGrid**
- **Mailgun**
- **Amazon SES**
- **Postmark**

Update the `EMAIL_BACKEND` setting accordingly for these services.
