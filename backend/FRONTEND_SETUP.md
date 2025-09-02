# Django Frontend Setup

## Overview

The frontend has been converted from static HTML files to Django templates for better integration with the backend authentication system.

## Structure

```
backend/
├── templates/
│   └── frontend/
│       ├── base.html          # Base template with common elements
│       ├── index.html         # Home page
│       ├── login.html         # Login page
│       ├── signup.html        # Signup page
│       ├── profile.html       # User profile page
│       ├── fixtures.html      # Fixtures page
│       ├── table.html         # League table page
│       ├── analysis.html      # Analytics page
│       ├── prediction.html    # Prediction page
│       ├── dashboard.html     # User dashboard
│       └── forgot-password.html # Password reset page
├── static/
│   └── frontend/
│       ├── css/
│       │   └── style.css      # Main stylesheet
│       └── js/
│           └── main.js        # Main JavaScript file
└── frontend/
    ├── views.py               # Django views
    ├── urls.py                # URL patterns
    └── apps.py                # App configuration
```

## Features

### ✅ **Django Template System**
- Base template with common elements (Bootstrap, Font Awesome, etc.)
- Template inheritance for consistent styling
- Django template tags and filters
- CSRF protection for forms

### ✅ **Authentication Integration**
- Login/signup forms integrated with Django backend
- JWT token handling in JavaScript
- Protected routes with `@login_required` decorator
- User session management

### ✅ **Static Files**
- CSS and JavaScript files served through Django
- Bootstrap and Font Awesome CDN integration
- Custom styling and functionality

### ✅ **Navigation**
- Dynamic navigation based on authentication status
- User profile integration
- Responsive design

## URL Patterns

| URL | View | Description |
|-----|------|-------------|
| `/` | `index` | Home page |
| `/login/` | `login_view` | Login page |
| `/signup/` | `signup_view` | Signup page |
| `/profile/` | `profile_view` | User profile (login required) |
| `/fixtures/` | `fixtures_view` | Fixtures page |
| `/table/` | `table_view` | League table |
| `/analysis/` | `analysis_view` | Analytics page |
| `/prediction/` | `prediction_view` | Predictions (login required) |
| `/dashboard/` | `dashboard_view` | User dashboard (login required) |
| `/forgot-password/` | `forgot_password_view` | Password reset |

## JavaScript Integration

### Main.js Features
- API request helper functions
- Authentication token management
- Alert message system
- Button loading states
- Navigation initialization

### Usage Example
```javascript
// Make API request
const data = await FootballApp.apiRequest('/auth/me/');

// Show alert
FootballApp.showAlert('Success!', 'success');

// Check authentication
if (FootballApp.isAuthenticated()) {
    // User is logged in
}
```

## Running the Application

1. **Start Django server:**
   ```bash
   cd backend
   python manage.py runserver
   ```

2. **Access the application:**
   - Home page: `http://127.0.0.1:8000/`
   - Login: `http://127.0.0.1:8000/login/`
   - Signup: `http://127.0.0.1:8000/signup/`
   - Profile: `http://127.0.0.1:8000/profile/`

## Benefits of Django Templates

1. **Server-side rendering** - Better SEO and initial load performance
2. **Template inheritance** - Consistent layout and styling
3. **CSRF protection** - Built-in security for forms
4. **User context** - Access to authenticated user data
5. **URL management** - Centralized URL patterns
6. **Static file handling** - Proper asset management
7. **Internationalization** - Built-in i18n support

## Migration from Static HTML

The original static HTML files in the `frontend/` directory have been converted to Django templates with the following improvements:

- ✅ Django template syntax
- ✅ CSRF tokens for forms
- ✅ URL name references
- ✅ Static file integration
- ✅ Authentication integration
- ✅ Server-side rendering

## Next Steps

1. **Add more dynamic content** - Connect templates to database models
2. **Implement real-time features** - WebSocket integration
3. **Add caching** - Template and static file caching
4. **Internationalization** - Multi-language support
5. **Testing** - Template and view testing
