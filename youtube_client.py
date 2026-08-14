# youtube_client.py
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


class YouTubeClientError(Exception):
    """Raised when YouTube data cannot be retrieved."""


class YouTubeConfigurationError(YouTubeClientError):
    """Raised when the server has no YouTube API key."""


# -------------------------------------------------------------------
# ✅ Extract Video ID from ALL YouTube URL formats (Watch, Shorts, etc.)
# -------------------------------------------------------------------
def extract_video_id(url: str):
    """
    Extract video ID from:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/embed/VIDEO_ID
    - Some other variants
    """

    if not url:
        return None

    patterns = [
        r"v=([a-zA-Z0-9_-]{6,})",
        r"youtu\.be/([a-zA-Z0-9_-]{6,})",
        r"shorts/([a-zA-Z0-9_-]{6,})",
        r"embed/([a-zA-Z0-9_-]{6,})",
        r"watch/([a-zA-Z0-9_-]{6,})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


# ---------------------------------------------------------
# Internal: API call helper with error handling
# ---------------------------------------------------------
def _call_youtube_api(url: str):
    if not API_KEY:
        raise YouTubeConfigurationError("YOUTUBE_API_KEY is not configured.")

    try:
        resp = requests.get(url, timeout=(5, 30))
    except requests.Timeout as exc:
        raise YouTubeClientError("YouTube API request timed out.") from exc
    except requests.RequestException as exc:
        raise YouTubeClientError("YouTube API network request failed.") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise YouTubeClientError(
            f"YouTube API returned a non-JSON response ({resp.status_code})."
        ) from exc

    if "error" in data:
        message = data["error"].get("message", "Unknown error")
        raise YouTubeClientError(f"YouTube API error: {message}")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise YouTubeClientError(
            f"YouTube API returned HTTP {resp.status_code}."
        ) from exc

    return data


def fetch_video_title(video_id: str) -> str:
    """Return the public YouTube title for a video."""
    if not video_id:
        raise YouTubeClientError("Invalid YouTube video ID.")

    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet&id={video_id}&key={API_KEY}"
    )
    data = _call_youtube_api(url)
    items = data.get("items", [])
    if not items:
        raise YouTubeClientError("Video is private, deleted, or unavailable.")

    title = items[0].get("snippet", {}).get("title", "").strip()
    if not title:
        raise YouTubeClientError("YouTube did not return a title for this video.")
    return title


# ---------------------------------------------------------
# Internal: Fetch replies for a top-level comment
# ---------------------------------------------------------
def _fetch_replies(parent_id: str, max_replies: int):
    replies = []

    page_token = None
    while len(replies) < max_replies:
        page_size = min(100, max_replies - len(replies))
        url = (
            "https://www.googleapis.com/youtube/v3/comments"
            f"?part=snippet&parentId={parent_id}&key={API_KEY}&maxResults={page_size}"
        )
        if page_token:
            url += f"&pageToken={page_token}"

        data = _call_youtube_api(url)

        for item in data.get("items", []):
            snippet = item["snippet"]
            text = snippet.get("textDisplay", "").strip()
            if text:
                replies.append({"text": text})

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return replies


# ---------------------------------------------------------
# MAIN: Fetch all comments (top-level + replies)
# ---------------------------------------------------------
def fetch_comments(video_id: str, max_comments: int = 50000):
    """
    Fetches ALL comments for a YouTube video or Short,
    safely handling pagination and comment availability.
    """

    if not video_id:
        raise YouTubeClientError("Invalid YouTube URL — could not extract video ID.")

    comments = []
    page_token = None
    total_fetched = 0

    print(f"🔍 Fetching comments for video: {video_id}")

    while True:
        url = (
            "https://www.googleapis.com/youtube/v3/commentThreads"
            f"?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
        )

        if page_token:
            url += f"&pageToken={page_token}"

        data = _call_youtube_api(url)

        items = data.get("items", [])

        # If no comments exist (common for Shorts / disabled comments)
        if not items and total_fetched == 0:
            print("⚠️ No comments found — comments may be disabled or not available yet.")
            return []

        for item in items:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "").strip()

            if text:
                comments.append({"text": text})
                total_fetched += 1

            # Fetch replies if present
            reply_count = item["snippet"].get("totalReplyCount", 0)
            if reply_count > 0:
                remaining = max_comments - total_fetched
                replies = _fetch_replies(item["id"], remaining)
                comments.extend(replies)
                total_fetched += len(replies)

            if total_fetched >= max_comments:
                print(f"✔️ Stopped early at {max_comments} comments limit.")
                return comments

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    print(f"✔️ Total comments fetched: {len(comments)}")
    return comments


# -------------------------------------------------------------------
# Debug: Run as script
# -------------------------------------------------------------------
if __name__ == "__main__":
    test_url = input("Enter YouTube URL: ")
    vid = extract_video_id(test_url)
    print("Video ID:", vid)
    if not vid:
        print("❌ Could not extract video ID.")
    else:
        comments = fetch_comments(vid, max_comments=300)
        print("Fetched:", len(comments))
