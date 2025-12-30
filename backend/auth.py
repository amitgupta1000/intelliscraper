# auth.py - Simple API Key Authentication for INTELLISEARCH
# Provides secure access control without complex user management

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Dict, List
from fastapi import HTTPException, Header, Depends
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

# Default API keys for development (override with environment variables)
DEFAULT_API_KEYS = {
    "demo-key-research-123": "demo_user",
    "demo-key-admin-456": "admin_user",
}

class APIKeyManager:
    """Manages API key validation and user identification"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        logger.info(f"Loaded {len(self.api_keys)} API keys for authentication")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables with fallback to defaults"""
        keys = {}
        
        # Load from environment variables (format: INTELLISEARCH_API_KEY_1=key:user_id)
        for i in range(1, 11):  # Support up to 10 API keys
            env_key = f"INTELLISEARCH_API_KEY_{i}"
            env_value = os.getenv(env_key)
            
            if env_value and ":" in env_value:
                api_key, user_id = env_value.split(":", 1)
                keys[api_key.strip()] = user_id.strip()
                logger.info(f"Loaded API key for user: {user_id}")
        
        # Use defaults if no environment keys found
        if not keys:
            keys = DEFAULT_API_KEYS.copy()
            logger.warning("Using default API keys - set INTELLISEARCH_API_KEY_* environment variables for production")
        
        return keys
    
    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID"""
        if not api_key:
            return None
        
        user_id = self.api_keys.get(api_key.strip())
        if user_id:
            logger.debug(f"Valid API key used by user: {user_id}")
        else:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        
        return user_id
    
    def generate_new_key(self, user_id: str) -> str:
        """Generate a new secure API key"""
        prefix = "intellisearch"
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}-{random_part}"

# Global API key manager instance
api_key_manager = APIKeyManager()

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter per API key"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.limits = {
            "requests_per_hour": int(os.getenv("RATE_LIMIT_HOURLY", "20")),
            "requests_per_minute": int(os.getenv("RATE_LIMIT_MINUTE", "5")),
        }
        logger.info(f"Rate limiting enabled: {self.limits}")
    
    def check_rate_limit(self, user_id: str) -> None:
        """Check if user has exceeded rate limits"""
        # Admin users bypass all rate limits
        if is_admin_user(user_id):
            logger.debug(f"Admin user {user_id} bypassing rate limits")
            return
        
        now = datetime.now()
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < timedelta(hours=1)  # Keep last hour
        ]
        
        # Count recent requests
        hour_requests = len([
            req for req in self.requests[user_id]
            if now - req < timedelta(hours=1)
        ])
        
        minute_requests = len([
            req for req in self.requests[user_id]
            if now - req < timedelta(minutes=1)
        ])
        
        # Check limits
        if hour_requests >= self.limits["requests_per_hour"]:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.limits['requests_per_hour']} requests per hour maximum"
            )
        
        if minute_requests >= self.limits["requests_per_minute"]:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.limits['requests_per_minute']} requests per minute maximum"
            )
        
        # Record this request
        self.requests[user_id].append(now)
        logger.debug(f"Rate limit check passed for {user_id}: {hour_requests}/hour, {minute_requests}/minute")

# Global rate limiter instance
rate_limiter = RateLimiter()

# =============================================================================
# FASTAPI DEPENDENCIES
# =============================================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None, description="API key for authentication")):
    """FastAPI dependency to verify API key from header"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header with your request.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    user_id = api_key_manager.validate_api_key(x_api_key)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please check your credentials.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return user_id

async def verify_api_key_with_rate_limit(user_id: str = Depends(verify_api_key)):
    """FastAPI dependency that combines API key verification with rate limiting"""
    try:
        rate_limiter.check_rate_limit(user_id)
        return user_id
    except HTTPException as e:
        logger.warning(f"Rate limit exceeded for user {user_id}")
        raise e

# =============================================================================
# OPTIONAL: ADMIN FUNCTIONS
# =============================================================================

def is_admin_user(user_id: str) -> bool:
    """Check if user has admin privileges"""
    admin_users = os.getenv("ADMIN_USERS", "admin_user").split(",")
    return user_id.strip() in [u.strip() for u in admin_users]

async def verify_admin_access(user_id: str = Depends(verify_api_key)):
    """FastAPI dependency for admin-only endpoints"""
    if not is_admin_user(user_id):
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return user_id

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_user_info(user_id: str) -> Dict:
    """Get user information for API responses"""
    admin_status = is_admin_user(user_id)
    
    return {
        "user_id": user_id,
        "is_admin": admin_status,
        "rate_limits": "unlimited" if admin_status else rate_limiter.limits,
        "authenticated_at": datetime.now().isoformat()
    }

def hash_api_key(api_key: str) -> str:
    """Hash API key for logging (security)"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

def log_api_usage(user_id: str, endpoint: str, success: bool = True):
    """Log API usage for monitoring and analytics"""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"API Usage - User: {user_id}, Endpoint: {endpoint}, Status: {status}")

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Test API key validation
    print("Testing API Key Authentication...")
    
    # Test valid key
    valid_user = api_key_manager.validate_api_key("demo-key-research-123")
    print(f"Valid key test: {valid_user}")
    
    # Test invalid key
    invalid_user = api_key_manager.validate_api_key("invalid-key")
    print(f"Invalid key test: {invalid_user}")
    
    # Test rate limiting
    print("\nTesting Rate Limiting...")
    try:
        for i in range(6):  # Exceed minute limit
            rate_limiter.check_rate_limit("test_user")
            print(f"Request {i+1}: OK")
    except Exception as e:
        print(f"Rate limit triggered: {e}")
    
    print("\nAuthentication module ready!")