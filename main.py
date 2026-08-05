import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from agent import analyze_comments
from youtube_client import (
    YouTubeClientError,
    YouTubeConfigurationError,
    extract_video_id,
)

logger = logging.getLogger(__name__)


def configured_origins():
    value = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in value.split(",") if origin.strip()]

app = FastAPI(title="YouTube Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not extract_video_id(req.url):
        raise HTTPException(status_code=400, detail="Enter a valid YouTube video URL.")

    try:
        result = analyze_comments(req.url)
        return jsonable_encoder(result)
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
        logger.exception("Unexpected analysis failure")
        raise HTTPException(status_code=500, detail="Failed to analyze comments.") from None
