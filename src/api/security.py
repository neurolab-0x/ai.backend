import time
import logging
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Union, Any

from src.utils.security import JWTAuth, validate_client_id, check_rate_limit
from src.config.settings import SECURITY_CONFIG

logger = logging.getLogger(__name__)

# Create JWT authentication handler
jwt_auth = JWTAuth(token_expiry=SECURITY_CONFIG['token_expiry_hours'])

# Create security scheme that's optional when authentication is disabled
class OptionalHTTPBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request):
        if not SECURITY_CONFIG['require_authentication']:
            return None
        return await super().__call__(request)

security = OptionalHTTPBearer(auto_error=False)

# ========== AUTHENTICATION DEPENDENCIES ==========

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Dependency to get current authenticated user from token
    
    Returns:
    --------
    dict: User information from token payload
    """
    if not SECURITY_CONFIG['require_authentication']:
        # If authentication is disabled, return a default anonymous user
        return {"sub": "anonymous", "roles": ["user", "admin"]}
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token = credentials.credentials
        payload = jwt_auth.verify_token(token)
        
        # Log successful authentication
        logger.info(f"User {payload['sub']} authenticated successfully")
        
        return payload
    except Exception as e:
        logger.warning(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def require_role(role: str, user: Dict = Depends(get_current_user)):
    """
    Dependency to check if user has required role
    
    Parameters:
    -----------
    role: str
        Role to check for
    user: Dict
        User information from get_current_user dependency
    
    Returns:
    --------
    dict: User information if authorized
    """
    if not SECURITY_CONFIG['require_authentication']:
        # If authentication is disabled, return user with all roles
        return {"sub": "anonymous", "roles": ["user", "admin"]}
    
    roles = user.get("roles", [])
    
    if role not in roles:
        logger.warning(f"Authorization failed for user {user['sub']}: Missing role {role}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required role: {role}"
        )
    
    return user

# Role-specific dependencies
async def require_user_role(user: Dict = Depends(get_current_user)):
    """Require user role"""
    return await require_role(SECURITY_CONFIG['required_role_realtime'], user)

async def require_admin_role(user: Dict = Depends(get_current_user)):
    """Require admin role"""
    return await require_role(SECURITY_CONFIG['required_role_admin'], user)

# ========== RATE LIMITING MIDDLEWARE ==========

async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware for rate limiting
    """
    if not SECURITY_CONFIG['enable_rate_limiting']:
        # Skip rate limiting if disabled
        return await call_next(request)
    
    # Get client identifier (IP address or client ID from header)
    client_id = request.headers.get("X-Client-ID", request.client.host)
    endpoint = request.url.path
    
    # Check rate limit
    if not check_rate_limit(
        client_id, 
        endpoint, 
        SECURITY_CONFIG['rate_limit_requests'],
        SECURITY_CONFIG['rate_limit_window_seconds']
    ):
        logger.warning(f"Rate limit exceeded for client {client_id} on {endpoint}")
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later."
        )
    
    # Continue with the request
    return await call_next(request)

# ========== VALIDATION DEPENDENCIES ==========

async def validate_client_identifier(x_client_id: Optional[str] = Header(None)):
    """
    Validate client identifier from header
    
    Parameters:
    -----------
    x_client_id: Optional[str]
        Client identifier from X-Client-ID header
    
    Returns:
    --------
    str: Validated client ID
    """
    if not x_client_id:
        return None
    
    if not validate_client_id(x_client_id):
        logger.warning(f"Invalid client ID format: {x_client_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client ID format"
        )
    
    return x_client_id 