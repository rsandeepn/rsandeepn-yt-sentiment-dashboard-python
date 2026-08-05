import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent import analyze_comments
from auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
    jwt_secret,
    normalize_email,
)
from database import Base, engine, get_db
from models import Analysis, User
from schemas import AnalysisHistoryDetail, AnalysisHistoryItem, AuthRequest, AuthResponse, UserResponse
from youtube_client import (
    YouTubeClientError,
    YouTubeConfigurationError,
    extract_video_id,
)

logger = logging.getLogger(__name__)


def configured_origins():
    value = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174",
    )
    return [origin.strip() for origin in value.split(",") if origin.strip()]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    jwt_secret()
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="YouTube Sentiment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class AnalyzeRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


def auth_response(user: User) -> AuthResponse:
    token, expires_in = create_access_token(user.id)
    return AuthResponse(access_token=token, expires_in=expires_in, user=user)


@app.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(req: AuthRequest, db: Session = Depends(get_db)):
    user = User(email=normalize_email(req.email), password_hash=hash_password(req.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from None
    db.refresh(user)
    return auth_response(user)


@app.post("/auth/login", response_model=AuthResponse)
def login(req: AuthRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.email, req.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_response(user)


@app.get("/auth/me", response_model=UserResponse)
def current_user(user: User = Depends(get_current_user)):
    return user


@app.post("/analyze")
def analyze(
    req: AnalyzeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Enter a valid YouTube video URL.")

    try:
        result = jsonable_encoder(analyze_comments(req.url))
        db.add(
            Analysis(
                user_id=user.id,
                video_id=video_id,
                video_url=req.url,
                result=result,
            )
        )
        db.commit()
        return result
    except YouTubeConfigurationError:
        logger.exception("YouTube API key is not configured")
        raise HTTPException(
            status_code=503,
            detail="The analysis service is not configured.",
        ) from None
    except YouTubeClientError:
        logger.exception("YouTube API request failed")
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve comments from YouTube.",
        ) from None
    except Exception:
        db.rollback()
        logger.exception("Unexpected analysis failure")
        raise HTTPException(status_code=500, detail="Failed to analyze comments.") from None


@app.get("/analyses", response_model=list[AnalysisHistoryItem])
def analysis_history(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return db.scalars(
        select(Analysis)
        .where(Analysis.user_id == user.id)
        .order_by(Analysis.created_at.desc())
        .limit(50)
    ).all()


@app.get("/analyses/{analysis_id}", response_model=AnalysisHistoryDetail)
def analysis_history_detail(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = db.scalar(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == user.id,
        )
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return analysis
