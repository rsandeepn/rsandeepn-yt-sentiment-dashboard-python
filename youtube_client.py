# youtube_client.py
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


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
    resp = requests.get(url)
    try:
        data = resp.json()
    except Exception:
        raise Exception(f"YouTube API Error: Non-JSON response ({resp.status_code})")

    if "error" in data:
        message = data["error"].get("message", "Unknown error")
        raise Exception(f"YouTube API Error: {message}")

    return data


# ---------------------------------------------------------
# Internal: Fetch replies for a top-level comment
# ---------------------------------------------------------
def _fetch_replies(parent_id: str):
    url = (
        "https://www.googleapis.com/youtube/v3/comments"
        f"?part=snippet&parentId={parent_id}&key={API_KEY}&maxResults=100"
    )
    data = _call_youtube_api(url)

    replies = []
    for item in data.get("items", []):
        snippet = item["snippet"]
        text = snippet.get("textDisplay", "").strip()
        if text:
            replies.append({"text": text})

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
        raise Exception("Invalid YouTube URL — could not extract video ID.")

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
                replies = _fetch_replies(item["id"])
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
