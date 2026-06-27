from datetime import datetime, timedelta
from enum import StrEnum

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from astronova_core.config import get_settings
from astronova_core.exceptions import AuthenticationError, AuthorizationError


class UserRole(StrEnum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SERVICE = "service"

class TokenData(BaseModel):
    username: str | None = None
    role: UserRole | None = None

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.jwt.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt.jwt_secret_key, algorithm=settings.jwt.jwt_algorithm)

def verify_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, settings.jwt.jwt_secret_key, algorithms=[settings.jwt.jwt_algorithm])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise AuthenticationError("Invalid token payload")
        return TokenData(username=username, role=UserRole(role))
    except JWTError:
        raise AuthenticationError("Could not validate credentials")

def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    if not token:
        raise AuthenticationError("Token missing")
    return verify_token(token)

class RoleChecker:
    def __init__(self, allowed_roles: list[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role not in self.allowed_roles:
            raise AuthorizationError("Access forbidden: Insufficient permissions")
        return current_user
