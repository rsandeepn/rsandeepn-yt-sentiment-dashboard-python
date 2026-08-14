import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import get_db
from models import User


JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
password_hash = PasswordHash.recommended()
dummy_hash = password_hash.hash("not-a-real-password")
bearer_scheme = HTTPBearer(auto_error=False)


def jwt_secret() -> str:
    value = os.getenv("JWT_SECRET_KEY")
    if not value or len(value) < 32:
        raise RuntimeError("JWT_SECRET_KEY must be configured with at least 32 characters")
    return value


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def unusable_password_hash() -> str:
    """Create a password hash that an OAuth-only user can never know."""
    return hash_password(secrets.token_urlsafe(48))


def verify_google_credential(credential: str) -> dict:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise RuntimeError("Google sign-in is not configured.")

    try:
        signing_key = PyJWKClient("https://www.googleapis.com/oauth2/v3/certs").get_signing_key_from_jwt(
            credential
        )
        payload = jwt.decode(
            credential,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=["accounts.google.com", "https://accounts.google.com"],
        )
    except (InvalidTokenError, PyJWKClientError, ValueError) as exc:
        raise ValueError("Google could not verify this sign-in.") from exc

    if not payload.get("email_verified") or not payload.get("email"):
        raise ValueError("Google did not return a verified email address.")
    return payload


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is None:
        password_hash.verify(password, dummy_hash)
        return None
    if not password_hash.verify(password, user.password_hash):
        return None
    return user


def create_access_token(user: User) -> tuple[str, int]:
    expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    token = jwt.encode(
        {"sub": user.id, "ver": user.auth_version, "exp": expires_at},
        jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )
    return token, expires_in


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Your session is invalid or has expired. Please sign in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise credentials_error
    try:
        payload = jwt.decode(
            credentials.credentials,
            jwt_secret(),
            algorithms=[JWT_ALGORITHM],
        )
        user_id = payload.get("sub")
        auth_version = payload.get("ver", 0)
        if not isinstance(user_id, str) or not isinstance(auth_version, int):
            raise credentials_error
    except (InvalidTokenError, RuntimeError):
        raise credentials_error from None

    user = db.get(User, user_id)
    if user is None or user.auth_version != auth_version:
        raise credentials_error
    return user
