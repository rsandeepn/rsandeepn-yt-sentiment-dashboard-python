import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


def extract_video_id(url: str):
    """
    Extracts the video ID from a YouTube URL.
    """
    import re
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def fetch_comments(video_id: str, max_comments: int = 50000):
    """
    Fetches top-level comments + replies for a video, with pagination.
    """
    comments = []
    page_token = None

    while True:
        url = (
            "https://www.googleapis.com/youtube/v3/commentThreads"
            f"?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
        )
        if page_token:
            url += f"&pageToken={page_token}"

        resp = requests.get(url)
        data = resp.json()

        if "error" in data:
            raise Exception(data["error"])

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "") or ""
            comments.append({"text": text})

            # replies
            if item["snippet"].get("totalReplyCount", 0) > 0:
                parent_id = item["id"]
                _fetch_replies(parent_id, comments)

            if len(comments) >= max_comments:
                return comments

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments


def _fetch_replies(parent_id: str, comments: list):
    """
    Fetch replies for a given top-level comment.
    """
    url = (
        "https://www.googleapis.com/youtube/v3/comments"
        f"?part=snippet&parentId={parent_id}&key={API_KEY}&maxResults=100"
    )
    resp = requests.get(url)
    data = resp.json()

    for item in data.get("items", []):
        snippet = item["snippet"]
        text = snippet.get("textDisplay", "") or ""
        comments.append({"text": text})
