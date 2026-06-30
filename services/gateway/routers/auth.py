from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from astronova_core.security import (
    UserRole,
    create_access_token,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str

@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Admin mock login
    if form_data.username == "admin" and form_data.password == "admin123":
        access_token = create_access_token(data={"sub": "admin", "role": UserRole.ADMIN})
        return LoginResponse(access_token=access_token, token_type="bearer", role=UserRole.ADMIN)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
