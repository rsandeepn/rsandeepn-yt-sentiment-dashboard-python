# agent.py
import os

# Reduce warnings / thread issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import re
from collections import Counter

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # faster noun-chunking

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from sentiment_model import classify_sentiment_batch
from youtube_client import fetch_comments, extract_video_id


EXAMPLE_LIMIT = 15  # examples per cluster


# ---------------------------------------------------------
# Load multilingual embedding model (for themes)
# ---------------------------------------------------------
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def clean_text(x: str) -> str:
    """Normalize whitespace and strip markup."""
    x = re.sub(r"<.*?>", "", x)
    return re.sub(r"\s+", " ", x).strip()


# ---------------------------------------------------------
# CLUSTER COMMENTS
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
        key = str(int(label))
        clusters.setdefault(key, []).append(comments[idx])

    result = {}
    for key, group in clusters.items():
        result[key] = {
            "summary": summarize_cluster_meaning(group),
            "examples": group[:EXAMPLE_LIMIT]
        }
    return result


def summarize_cluster_meaning(comments):
    """Pick the longest comment as the cluster representative."""
    if not comments:
        return "No insights."

    rep = max(comments, key=lambda x: len(x))
    return f"Viewers mentioned: {rep}"


# ---------------------------------------------------------
# IMPROVED KEYWORD EXTRACTION (USING SPACY NOUN FILTERING)
# ---------------------------------------------------------
REMOVE_POS = {"PRON", "DET", "AUX", "CCONJ", "SCONJ", "ADV", "VERB"}
REMOVE_WORDS = {
    "the","and","for","this","that","with","have","from","your","you",
    "very","just","like","here","please","thanks","thank","thank you",
    "will","should","could","would","how","what","why","when","where",
    "do","does","did","is","am","are","was","were","be","been","being",
    "next","time","want","i","me","my","we","our","they","their","he",
    "she","it","a","an","to","but","not","his","her","its"
}


def extract_keywords(texts, top_k: int = 10):
    """Extracts clean nouns from comments using spaCy."""
    all_nouns = []

    for text in texts:
        text = text.strip().lower()
        doc = nlp(text)

        for tok in doc:
            if len(tok.text) <= 2:
                continue
            if tok.text in REMOVE_WORDS:
                continue
            if tok.pos_ in REMOVE_POS:
                continue
            all_nouns.append(tok.text)

    if not all_nouns:
        return []

    freq = Counter(all_nouns)
    return [w for w, _ in freq.most_common(top_k)]


# ---------------------------------------------------------
# Detect viewer suggestions
# ---------------------------------------------------------
SUGGESTION_PHRASES = [
    "should", "please", "i think you", "you need to", "you must",
    "can you", "could you", "would be better", "need more", "do more",
    "make more", "next time", "try to", "improve", "improvement",
    "suggest", "my suggestion", "request", "please upload", "pls upload",
    "do a video on"
]

# ---------------------------------------------------------
# Stopwords for theme keyword extraction
# ---------------------------------------------------------
SUGGESTION_STOPWORDS = [
    "the","and","for","this","that","with","have","from","your","you","very",
    "just","like","here","please","thanks","thank","thank you","will","how",
    "should","can","could","would","need","make","more","do","did","does",
    "is","am","are","was","were","be","been","being","next","time","want",
    "i","me","my","we","our","they","their","he","she","it","a","an","to",
    "but","what","who","why","when","where"
]



def is_suggestion(text: str) -> bool:
    t = text.lower()
    if len(t) < 15:
        return False
    return any(phrase in t for phrase in SUGGESTION_PHRASES)


def detect_suggestions(all_items):
    return [item for item in all_items if is_suggestion(item["text"])]


def build_suggestions_overview(suggestions):
    if not suggestions:
        return "No clear suggestions or improvement requests were found."

    texts = [s["text"] for s in suggestions]
    keywords = extract_keywords(texts, top_k=8)
    keywords_fmt = ", ".join(keywords[:5]) if keywords else "various content improvements"

    return (
        f"There are {len(suggestions)} viewer suggestions. "
        f"Common themes include: {keywords_fmt}. "
        "These suggestions help improve future videos by highlighting what viewers want."
    )


# ---------------------------------------------------------
# Build short overview
# ---------------------------------------------------------
def build_short_overview(all_items, positives, negatives, neutrals, suggestions):
    total = len(all_items)
    pos = len(positives)
    neg = len(negatives)
    neu = len(neutrals)
    sug = len(suggestions)

    if total == 0:
        return "There are no comments available for this video yet."

    # Mood logic
    if pos > neg * 1.5:
        mood = "mostly positive"
    elif neg > pos * 1.5:
        mood = "mostly negative"
    else:
        mood = "mixed"

    # Clean topic extraction
    keywords = extract_keywords([c["text"] for c in all_items])
    topic_str = ", ".join(keywords[:5]) if keywords else "various topics"

    overview = (
        f"Viewers left {total} comments. "
        f"The overall mood is {mood}. "
        f"People frequently talk about {topic_str}. "
        "Positive comments appreciate performance, clarity, or presentation. "
        "Negative comments focus on expectations or pacing. "
    )

    if sug > 0:
        overview += f"Additionally, {sug} viewers provided improvement suggestions."

    return overview


# ---------------------------------------------------------
# FULL SUMMARY (Markdown)
# ---------------------------------------------------------
def build_overall_summary(texts, sentiments, pos_clusters, neg_clusters, neu_count, suggestions):
    total = len(texts)
    pos = sum(1 for s in sentiments if s["sentiment"] == "positive")
    neg = sum(1 for s in sentiments if s["sentiment"] == "negative")
    neu = sum(1 for s in sentiments if s["sentiment"] == "neutral")
    sug = len(suggestions)

    pos_pct = round((pos / total) * 100, 1) if total else 0
    neg_pct = round((neg / total) * 100, 1) if total else 0
    neu_pct = round((neu / total) * 100, 1) if total else 0

    summary = f"""
### 🧠 High-Level Summary

Total Comments: **{total}**
Positive: **{pos}** ({pos_pct}%)
Negative: **{neg}** ({neg_pct}%)
Neutral: **{neu}** ({neu_pct}%)
Suggestions: **{sug}**

---

### ⭐ Positive Themes
"""
    for v in pos_clusters.values():
        summary += f"- {v['summary']}\n"

    summary += "\n### ⚠️ Negative Themes\n"
    for v in neg_clusters.values():
        summary += f"- {v['summary']}\n"

    summary += f"""

### 😐 Neutral Observations
- {neu_count} neutral or informational comments.

"""

    summary += "\n### 💡 Viewer Suggestions & Requests\n"
    for s in suggestions[:10]:
        summary += f"- {s['text']}\n"

    return summary


# ---------------------------------------------------------
# MAIN ENTRY
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

    # 🔍 Suggestion-style comments
    suggestions = detect_suggestions(all_items)

    print("🔍 Clustering positive comments...")
    pos_clusters = cluster_and_summarize([c["text"] for c in positives], num_clusters=6)

    print("🔍 Clustering negative comments...")
    neg_clusters = cluster_and_summarize([c["text"] for c in negatives], num_clusters=6)

    # ------------------------------
    # NEW THEME OVERVIEW
    # ------------------------------
    theme_overview = {
        "positive": summarize_theme_overview(pos_clusters, "positive"),
        "negative": summarize_theme_overview(neg_clusters, "negative"),
        "neutral": summarize_theme_overview(
            {"neutral": [n["text"] for n in neutrals]},
            "neutral"
        )
    }

    # ------------------------------
    # Summaries
    # ------------------------------
    short_overview = build_short_overview(
        all_items, positives, negatives, neutrals, suggestions
    )

    long_summary = build_overall_summary(
        texts,
        sentiment_results,
        pos_clusters,
        neg_clusters,
        len(neutrals),
        suggestions,
    )

    suggestions_overview = build_suggestions_overview(suggestions)

    # ------------------------------
    # FINAL RETURN OBJECT
    # ------------------------------
    return {
        "overview": short_overview,
        "summary": long_summary,
        "positive_clusters": pos_clusters,
        "negative_clusters": neg_clusters,

        "theme_overview": theme_overview,

        "stats": {
            "total": len(all_items),
            "positive": len(positives),
            "negative": len(negatives),
            "neutral": len(neutrals),
            "suggestions": len(suggestions),
        },

        "all_comments": all_items,

        "suggestions": {
            "count": len(suggestions),
            "overview": suggestions_overview,
            "examples": [s["text"] for s in suggestions[:20]],
        },
    }

def summarize_theme_overview(clusters: dict, label: str):
    """
    clusters = {
       "0": {"summary": "...", "examples": [...]},
       "1": {"summary": "...", "examples": [...]},
       ...
    }
    """
    if not clusters:
        return f"No major {label} themes found."

    summaries = []
    for cid, data in clusters.items():   # data is the dict
        if isinstance(data, dict) and "summary" in data:
            summaries.append(data["summary"])

    if not summaries:
        return f"No major {label} themes identified."

    # Combine them nicely
    return (
        f"Top {label} themes include: "
        + "; ".join(summaries[:4])  # limit to 4 themes
        + "."
    )



