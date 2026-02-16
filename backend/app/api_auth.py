import os
import secrets
import hashlib
from typing import Optional, Dict
from datetime import datetime, timedelta
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.models import APICredential

_api_keys: Dict[str, APICredential] = {}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def generate_api_key() -> str:
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

def create_api_credential(
    organization: str,
    email: str,
    role: str = "user",
    expires_days: Optional[int] = None
) -> APICredential:
    
    api_key = generate_api_key()
    hashed_key = hash_api_key(api_key)

    expires_at = None
    if expires_days:
        expires_at = datetime.now() + timedelta(days=expires_days)

    credential = APICredential(
        api_key=hashed_key,
        organization=organization,
        email=email,
        role=role,
        expires_at=expires_at
    )

    _api_keys[api_key] = credential

    credential.api_key = api_key
    return credential

def validate_api_key(api_key: str) -> Optional[APICredential]:
    if not api_key:
        return None

    credential = _api_keys.get(api_key)

    if not credential:
        return None

    if not credential.is_active:
        return None

    if credential.expires_at and credential.expires_at < datetime.now():
        return None

    return credential

async def verify_api_key(api_key: str = Security(api_key_header)) -> APICredential:
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )

    credential = validate_api_key(api_key)

    if not credential:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key"
        )

    return credential

def require_role(required_role: str):
    async def role_checker(credential: APICredential = Security(verify_api_key)):
        if required_role == "admin" and credential.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Admin role required"
            )
        return credential

    return role_checker

def get_api_credential(api_key: str) -> Optional[APICredential]:
    return _api_keys.get(api_key)
