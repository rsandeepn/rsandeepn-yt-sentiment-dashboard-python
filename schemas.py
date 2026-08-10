from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class AuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class GoogleAuthRequest(BaseModel):
    credential: str = Field(min_length=1, max_length=5000)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    first_name: str | None
    last_name: str | None
    email: EmailStr
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class AnalysisHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    video_id: str
    video_url: str
    status: str
    progress: int
    status_message: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class AnalysisHistoryDetail(AnalysisHistoryItem):
    result: dict


class AnalysisHistoryPage(BaseModel):
    items: list[AnalysisHistoryItem]
    page: int
    page_size: int
    total: int
    total_pages: int
