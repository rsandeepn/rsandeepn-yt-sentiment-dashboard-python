# agent.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import re
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from sentiment_model import classify_sentiment_batch
from youtube_client import fetch_comments, extract_video_id

EXAMPLE_LIMIT = 15  # how many example comments to show per cluster


# ---------------------------------------------------------
# Load multilingual embedding model (for themes)
# ---------------------------------------------------------
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


# ---------------------------------------------------------
# Cluster comments into meaning groups
# ---------------------------------------------------------
def cluster_and_summarize(comments, num_clusters: int = 6):
    if len(comments) == 0:
        return {}

    comments = [clean_text(c) for c in comments]

    if len(comments) < num_clusters:
        num_clusters = max(1, len(comments))

    try:
        embeddings = embedder.encode(comments)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
    except Exception as e:
        print("⚠️ Clustering failed:", e)
        return {}

    clusters = {}
    for idx, label in enumerate(labels):
        # Cast label to string so it's JSON-safe
        key = str(int(label))
        clusters.setdefault(key, []).append(comments[idx])

    result = {}
    for key, group in clusters.items():
        result[key] = {
            "summary": summarize_cluster_meaning(group),
            "examples": group[:EXAMPLE_LIMIT],
        }
    return result


def summarize_cluster_meaning(comments):
    if not comments:
        return "No insights."

    rep = max(comments, key=lambda x: len(x))
    return f"Viewers mentioned: {rep}"


# ---------------------------------------------------------
# Simple keyword extraction for overview
# ---------------------------------------------------------
STOPWORDS = {
    "the",
    "and",
    "for",
    "this",
    "that",
    "with",
    "have",
    "from",
    "your",
    "you",
    "movie",
    "video",
    "song",
    "very",
    "just",
    "like",
    "here",
}


def extract_keywords(texts, top_k: int = 10):
    words = []
    for t in texts:
        t = t.lower()
        t = re.sub(r"[^a-zA-Z0-9\u0C00-\u0C7F\u0B80-\u0BFF\u0900-\u097F]+", " ", t)
        tokens = t.split()
        for tok in tokens:
            if len(tok) <= 2:
                continue
            if tok in STOPWORDS:
                continue
            words.append(tok)
    if not words:
        return []
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k)]


# ---------------------------------------------------------
# Build high-level overview (4–5 sentences)
# ---------------------------------------------------------
def build_short_overview(all_items, positives, negatives, neutrals):
    total = len(all_items)
    pos = len(positives)
    neg = len(negatives)
    neu = len(neutrals)

    if total == 0:
        return "There are no comments available for this video yet."

    if pos > neg * 1.5:
        mood = "mostly positive"
    elif neg > pos * 1.5:
        mood = "mostly negative"
    else:
        mood = "mixed, with both positive and negative opinions"

    keywords = extract_keywords([c["text"] for c in all_items], top_k=8)
    topics_text = ", ".join(keywords[:5]) if keywords else "various topics"

    overview_lines = [
        f"Viewers left {total} comments on this video.",
        f"The overall mood of the comment section is {mood}.",
        f"People frequently talk about {topics_text}.",
        "Many positive comments appreciate aspects like performance, presentation, or emotional impact.",
        "Negative comments usually focus on expectations, pacing, certain scenes, or dislike for specific elements.",
    ]

    return " ".join(overview_lines)


# ---------------------------------------------------------
# Existing long summary (kept)
# ---------------------------------------------------------
def build_overall_summary(texts, sentiments, pos_clusters, neg_clusters, neu_count):
    total = len(texts)
    pos = sum(1 for s in sentiments if s["sentiment"] == "positive")
    neg = sum(1 for s in sentiments if s["sentiment"] == "negative")
    neu = sum(1 for s in sentiments if s["sentiment"] == "neutral")

    if total == 0:
        pos_ratio = neg_ratio = neu_ratio = 0.0
    else:
        pos_ratio = round(pos / total * 100, 1)
        neg_ratio = round(neg / total * 100, 1)
        neu_ratio = round(neu / total * 100, 1)

    if pos > neg * 1.5:
        mood = "The overall viewer reaction is strongly positive 🎉"
    elif neg > pos * 1.5:
        mood = "The overall viewer reaction is mostly negative 😕"
    else:
        mood = "The comments show a mix of positive and negative opinions 🤔"

    summary = f"""
### 🧠 **HIGH-LEVEL SUMMARY OF ALL COMMENTS**

- Total comments analyzed: **{total}**
- Positive: **{pos}** ({pos_ratio}%)
- Negative: **{neg}** ({neg_ratio}%)
- Neutral: **{neu}** ({neu_ratio}%)

**Overall mood:** {mood}

---

### ⭐ **Positive Themes (summarized):**
"""

    if pos_clusters:
        for v in pos_clusters.values():
            summary += f"- {v['summary']}\n"
    else:
        summary += "- No major positive sentiment detected.\n"

    summary += """

### ⚠️ **Negative Themes (summarized):**
"""

    if neg_clusters:
        for v in neg_clusters.values():
            summary += f"- {v['summary']}\n"
    else:
        summary += "- No major negative sentiment detected.\n"

    summary += f"""

### 😐 **Neutral Observations:**
- Viewers shared {neu_count} neutral or informational comments.
- These usually include general statements, jokes, or unrelated chat.

---

"""
    return summary


# ---------------------------------------------------------
# Main: analyze full YouTube comments
# ---------------------------------------------------------
def analyze_comments(video_url: str):
    print("📥 Fetching comments...")
    video_id = extract_video_id(video_url)
    raw_comments = fetch_comments(video_id, max_comments=5000)

    texts = [clean_text(c["text"]) for c in raw_comments]
    print(f"🧠 Classifying {len(texts)} comments...")

    sentiment_results = classify_sentiment_batch(texts)

    all_items = [
        {
            "text": r["text"],
            "sentiment": r["sentiment"],
            "score": r["score"],
        }
        for r in sentiment_results
    ]

    positives = [c for c in all_items if c["sentiment"] == "positive"]
    negatives = [c for c in all_items if c["sentiment"] == "negative"]
    neutrals = [c for c in all_items if c["sentiment"] == "neutral"]

    print("🔍 Clustering positive comments...")
    pos_clusters = cluster_and_summarize([c["text"] for c in positives], num_clusters=6)

    print("🔍 Clustering negative comments...")
    neg_clusters = cluster_and_summarize([c["text"] for c in negatives], num_clusters=6)

    short_overview = build_short_overview(all_items, positives, negatives, neutrals)
    long_summary = build_overall_summary(texts, sentiment_results, pos_clusters, neg_clusters, len(neutrals))

    return {
        "overview": short_overview,          # 4–5 simple sentences
        "summary": long_summary,             # existing markdown-style block
        "positive_clusters": pos_clusters,
        "negative_clusters": neg_clusters,
        "stats": {
            "total": len(all_items),
            "positive": len(positives),
            "negative": len(negatives),
            "neutral": len(neutrals),
        },
        "all_comments": all_items,           # for search + top-N in UI
    }
