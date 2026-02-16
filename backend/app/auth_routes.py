"""
Authentication Routes for Resource Management System
Provides basic authentication endpoints for the React frontend
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router with /api/auth prefix
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ============================================================================
# DATA MODELS
# ============================================================================

class User(BaseModel):
    id: str
    email: str
    name: str
    role: str  # 'admin' or 'user'


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    user: User
    token: Optional[str] = None  # For future JWT implementation


# ============================================================================
# IN-MEMORY USER STORAGE (For Development)
# TODO: Replace with proper database and password hashing in production
# ============================================================================

# Sample users for development
SAMPLE_USERS = {
    "admin@company.com": {
        "id": "1",
        "email": "admin@company.com",
        "password": "admin",  # In production, this should be hashed
        "name": "Admin User",
        "role": "admin"
    },
    "user@company.com": {
        "id": "2",
        "email": "user@company.com",
        "password": "user",  # In production, this should be hashed
        "name": "Regular User",
        "role": "user"
    },
    "john@company.com": {
        "id": "3",
        "email": "john@company.com",
        "password": "password",
        "name": "John Smith",
        "role": "user"
    }
}


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """
    Authenticate user and return user information

    Sample credentials for testing:
    - Admin: admin@company.com / admin
    - User: user@company.com / user
    - John: john@company.com / password
    """
    logger.info(f"Login attempt for email: {credentials.email}")

    # Check if user exists
    user_data = SAMPLE_USERS.get(credentials.email)

    if not user_data:
        logger.warning(f"Login failed: User not found - {credentials.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Verify password (in production, use proper password hashing)
    if user_data["password"] != credentials.password:
        logger.warning(f"Login failed: Invalid password for - {credentials.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create user object (exclude password)
    user = User(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        role=user_data["role"]
    )

    logger.info(f"Login successful for user: {credentials.email}")

    # Return user data (token can be added later for JWT implementation)
    return LoginResponse(user=user, token=None)


@router.post("/logout")
async def logout():
    """
    Logout user (placeholder for future session/token invalidation)
    """
    logger.info("Logout called")
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_current_user():
    """
    Get current authenticated user
    TODO: Implement JWT token verification and return current user
    For now, returns a placeholder response
    """
    # This would normally verify the JWT token and return the user
    # For development, we'll return a placeholder
    return {
        "message": "Authentication required",
        "note": "This endpoint requires JWT token implementation"
    }


@router.post("/register")
async def register_user(user_data: dict):
    """
    Register a new user (placeholder for future implementation)
    TODO: Implement user registration with password hashing
    """
    return {
        "message": "User registration endpoint",
        "note": "This endpoint is not yet implemented. Use sample credentials for testing."
    }
