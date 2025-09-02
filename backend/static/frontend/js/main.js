// Main JavaScript file for frontend functionality

// API Configuration
const API_BASE_URL = window.location.origin + '/api';

// Get CSRF token from Django
function getCSRFToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    return token ? token.value : null;
}

// Authentication helpers
function getAuthToken() {
    return localStorage.getItem('access');
}

function setAuthToken(token) {
    localStorage.setItem('access', token);
}

function clearAuthTokens() {
    localStorage.removeItem('access');
    localStorage.removeItem('refresh');
    localStorage.removeItem('user');
}

// Check if user is authenticated
function isAuthenticated() {
    return !!getAuthToken();
}

// Redirect to login if not authenticated
function requireAuth() {
    if (!isAuthenticated()) {
        window.location.href = '/login/';
    }
}

// API request helper
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const token = getAuthToken();
    
    const csrfToken = getCSRFToken();
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` }),
            ...(csrfToken && { 'X-CSRFToken': csrfToken })
        }
    };
    
    const finalOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };
    
    try {
        const response = await fetch(url, finalOptions);
        
        // Handle different response types
        let data;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
        } else {
            data = { detail: await response.text() };
        }
        
        if (!response.ok) {
            // Handle different error types
            if (response.status === 401) {
                // Unauthorized - clear tokens and redirect to login
                clearAuthTokens();
                window.location.href = '/login/';
                throw new Error('Session expired. Please login again.');
            } else if (response.status === 403) {
                throw new Error('Access denied. You do not have permission to perform this action.');
            } else if (response.status === 404) {
                throw new Error('The requested resource was not found.');
            } else if (response.status >= 500) {
                throw new Error('Server error. Please try again later.');
            } else {
                // Handle validation errors
                if (data && typeof data === 'object') {
                    if (data.detail) {
                        throw new Error(data.detail);
                    } else if (data.errors) {
                        const errorMessages = Object.values(data.errors).flat();
                        throw new Error(errorMessages.join(', '));
                    } else if (data.non_field_errors) {
                        throw new Error(data.non_field_errors.join(', '));
                    } else {
                        // Handle field-specific validation errors
                        const fieldErrors = [];
                        Object.keys(data).forEach(field => {
                            if (Array.isArray(data[field])) {
                                fieldErrors.push(`${field}: ${data[field].join(', ')}`);
                            }
                        });
                        if (fieldErrors.length > 0) {
                            throw new Error(fieldErrors.join('; '));
                        }
                    }
                }
                throw new Error(data.detail || `Request failed with status ${response.status}`);
            }
        }
        
        return data;
    } catch (error) {
        console.error('API request error:', error);
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error. Please check your internet connection.');
        }
        throw error;
    }
}

// Show alert message
function showAlert(message, type = 'info', containerId = 'alert-container') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
    }
}

// Clear alert
function clearAlert(containerId = 'alert-container') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
    }
}

// Set loading state for button
function setButtonLoading(button, loading = true) {
    const text = button.querySelector('.button-text');
    const spinner = button.querySelector('.button-spinner');
    
    if (loading) {
        if (text) text.style.display = 'none';
        if (spinner) spinner.style.display = 'block';
        button.disabled = true;
    } else {
        if (text) text.style.display = 'block';
        if (spinner) spinner.style.display = 'none';
        button.disabled = false;
    }
}

// Initialize navigation based on authentication status
function initNavigation() {
    const token = getAuthToken();
    const loginLink = document.getElementById('nav-login');
    const signupLink = document.getElementById('nav-signup');
    const profileLink = document.getElementById('nav-profile');
    const dashboardLink = document.getElementById('nav-dashboard');
    const predictionLink = document.getElementById('nav-prediction');
    
    if (token) {
        // User is logged in
        if (loginLink) loginLink.style.display = 'none';
        if (signupLink) signupLink.style.display = 'none';
        if (profileLink) profileLink.style.display = 'block';
        if (dashboardLink) dashboardLink.style.display = 'block';
        if (predictionLink) predictionLink.style.display = 'block';
        
        // Load user info for profile link
        loadUserInfo();
    } else {
        // User is not logged in
        if (loginLink) loginLink.style.display = 'block';
        if (signupLink) signupLink.style.display = 'block';
        if (profileLink) profileLink.style.display = 'none';
        if (dashboardLink) dashboardLink.style.display = 'none';
        if (predictionLink) predictionLink.style.display = 'none';
    }
}

// Load user information
async function loadUserInfo() {
    try {
        const userData = await apiRequest('/auth/me/');
        const initialsElement = document.getElementById('nav-initials');
        if (initialsElement && userData.username) {
            initialsElement.textContent = userData.username[0].toUpperCase();
        }
    } catch (error) {
        console.error('Failed to load user info:', error);
    }
}

// Logout function
function logout() {
    clearAuthTokens();
    window.location.href = '/';
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
});

// Export functions for use in other scripts
window.FootballApp = {
    apiRequest,
    showAlert,
    clearAlert,
    setButtonLoading,
    getAuthToken,
    setAuthToken,
    clearAuthTokens,
    isAuthenticated,
    requireAuth,
    logout,
    initNavigation,
    getCSRFToken
};
