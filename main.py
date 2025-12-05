# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from agent import analyze_comments


app = FastAPI(title="YouTube Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        result = analyze_comments(req.url)
        return jsonable_encoder(result)
    except Exception as e:
        print("Error in /analyze:", e)
        raise HTTPException(status_code=500, detail="Failed to analyze comments")
