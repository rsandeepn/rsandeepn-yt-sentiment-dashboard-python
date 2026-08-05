from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class AuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
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
