import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import func, or_, select, update
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
    unusable_password_hash,
    verify_google_credential,
)
from database import (
    Base,
    SessionLocal,
    engine,
    ensure_analysis_job_columns,
    ensure_user_profile_columns,
    get_db,
)
from models import Analysis, User
from schemas import (
    AnalysisHistoryDetail,
    AnalysisHistoryPage,
    AuthRequest,
    AuthResponse,
    GoogleAuthRequest,
    RegisterRequest,
    UserResponse,
)
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
    ensure_analysis_job_columns()
    ensure_user_profile_columns()
    with SessionLocal() as db:
        db.execute(
            update(Analysis)
            .where(Analysis.status.in_(["queued", "running"]))
            .values(
                status="failed",
                error_message="The server restarted before this analysis completed. Please retry it.",
                status_message="Interrupted",
                updated_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
    yield


app = FastAPI(title="YouTube Sentiment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


class AnalyzeRequest(BaseModel):
    url: str
    force: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


def auth_response(user: User) -> AuthResponse:
    token, expires_in = create_access_token(user.id)
    return AuthResponse(access_token=token, expires_in=expires_in, user=user)


@app.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    user = User(
        first_name=req.first_name,
        last_name=req.last_name,
        email=normalize_email(req.email),
        password_hash=hash_password(req.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from None
    db.refresh(user)
    return auth_response(user)


@app.post("/auth/google", response_model=AuthResponse)
def google_login(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    try:
        profile = verify_google_credential(req.credential)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None

    email = normalize_email(profile["email"])
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(
            first_name=(profile.get("given_name") or "").strip() or None,
            last_name=(profile.get("family_name") or "").strip() or None,
            email=email,
            password_hash=unusable_password_hash(),
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            user = db.scalar(select(User).where(User.email == email))
        else:
            db.refresh(user)
    else:
        changed = False
        if not user.first_name and profile.get("given_name"):
            user.first_name = profile["given_name"].strip()
            changed = True
        if not user.last_name and profile.get("family_name"):
            user.last_name = profile["family_name"].strip()
            changed = True
        if changed:
            db.commit()

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


def update_job(analysis_id: str, **values):
    values["updated_at"] = datetime.now(timezone.utc)
    with SessionLocal() as db:
        db.execute(update(Analysis).where(Analysis.id == analysis_id).values(**values))
        db.commit()


def friendly_analysis_error(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, YouTubeConfigurationError):
        return "The analysis service is not configured."
    if "quota" in message:
        return "The YouTube API quota is exhausted. Please try again after the quota resets."
    if "disabled" in message or "no comments" in message or "not available" in message:
        return "No comments are available. They may be disabled or the video may have no comments."
    if "timeout" in message or "timed out" in message or "network" in message:
        return "YouTube could not be reached. Please retry in a moment."
    if isinstance(exc, YouTubeClientError):
        return "Unable to retrieve comments. The video may be private, deleted, or unavailable."
    return "The analysis failed unexpectedly. Please retry."


def run_analysis_job(analysis_id: str):
    try:
        with SessionLocal() as db:
            analysis = db.get(Analysis, analysis_id)
            if analysis is None:
                return
            video_url = analysis.video_url

        update_job(
            analysis_id,
            status="running",
            progress=5,
            status_message="Starting analysis",
            error_message=None,
        )

        def report(progress: int, message: str):
            update_job(
                analysis_id,
                status="running",
                progress=progress,
                status_message=message,
            )

        result = jsonable_encoder(analyze_comments(video_url, progress_callback=report))
        now = datetime.now(timezone.utc)
        update_job(
            analysis_id,
            status="completed",
            progress=100,
            status_message="Completed",
            result=result,
            error_message=None,
            completed_at=now,
        )
    except Exception as exc:
        logger.exception("Analysis job %s failed", analysis_id)
        update_job(
            analysis_id,
            status="failed",
            status_message="Failed",
            error_message=friendly_analysis_error(exc),
        )


def create_or_reuse_analysis(
    req: AnalyzeRequest,
    user: User,
    db: Session,
    background_tasks: BackgroundTasks,
) -> Analysis:
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Enter a valid YouTube video URL.")

    if not req.force:
        existing = db.scalar(
            select(Analysis)
            .where(
                Analysis.user_id == user.id,
                Analysis.video_id == video_id,
                Analysis.status.in_(["queued", "running", "completed"]),
            )
            .order_by(Analysis.created_at.desc())
        )
        if existing is not None:
            return existing

    analysis = Analysis(
        user_id=user.id,
        video_id=video_id,
        video_url=req.url.strip(),
        result={},
        status="queued",
        progress=0,
        status_message="Queued",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    background_tasks.add_task(run_analysis_job, analysis.id)
    return analysis


@app.post(
    "/analyses",
    response_model=AnalysisHistoryDetail,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_analysis(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return create_or_reuse_analysis(req, user, db, background_tasks)


@app.post(
    "/analyze",
    response_model=AnalysisHistoryDetail,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze_compatibility(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return create_or_reuse_analysis(req, user, db, background_tasks)


@app.get("/analyses", response_model=AnalysisHistoryPage)
def analysis_history(
    search: str = "",
    job_status: str = Query("all", alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if job_status not in {"all", "queued", "running", "completed", "failed"}:
        raise HTTPException(status_code=400, detail="Invalid analysis status filter.")

    filters = [Analysis.user_id == user.id]
    if search.strip():
        term = f"%{search.strip()}%"
        filters.append(or_(Analysis.video_id.ilike(term), Analysis.video_url.ilike(term)))
    if job_status != "all":
        filters.append(Analysis.status == job_status)

    total = db.scalar(select(func.count()).select_from(Analysis).where(*filters)) or 0
    items = db.scalars(
        select(Analysis)
        .where(*filters)
        .order_by(Analysis.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return AnalysisHistoryPage(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=math.ceil(total / page_size) if total else 0,
    )


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


@app.post("/analyses/{analysis_id}/retry", response_model=AnalysisHistoryDetail)
def retry_analysis(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id)
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if analysis.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed analyses can be retried.")
    analysis.status = "queued"
    analysis.progress = 0
    analysis.status_message = "Queued for retry"
    analysis.error_message = None
    analysis.completed_at = None
    db.commit()
    db.refresh(analysis)
    background_tasks.add_task(run_analysis_job, analysis.id)
    return analysis


@app.post("/analyses/{analysis_id}/reanalyze", response_model=AnalysisHistoryDetail)
def reanalyze(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id)
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return create_or_reuse_analysis(
        AnalyzeRequest(url=analysis.video_url, force=True), user, db, background_tasks
    )


@app.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id)
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if analysis.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="A running analysis cannot be deleted.")
    db.delete(analysis)
    db.commit()
