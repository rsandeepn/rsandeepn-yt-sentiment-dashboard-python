import re
from collections import Counter


TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

STOPWORDS = {
    "the", "and", "for", "this", "that", "with", "have", "from", "your",
    "you", "very", "just", "like", "here", "please", "thanks", "thank",
    "will", "how", "should", "can", "could", "would", "need", "make",
    "more", "are", "was", "were", "been", "being", "what", "who", "why",
    "when", "where", "not", "but", "its", "our", "their", "video",
}

def top_keywords(texts, limit=10):
    words = Counter()
    for text in texts:
        for token in TOKEN_RE.findall(text.lower()):
            if token not in STOPWORDS:
                words[token] += 1
    return [
        {"keyword": keyword, "count": count}
        for keyword, count in words.most_common(limit)
    ]


def build_dashboard_insights(items, suggestion_count):
    total = len(items)
    counts = Counter(item.get("sentiment", "neutral") for item in items)
    positive = counts["positive"]
    negative = counts["negative"]
    neutral = counts["neutral"]
    dominant = max(
        ("positive", "negative", "neutral"),
        key=lambda label: counts[label],
    ) if total else "neutral"
    scores = [float(item.get("score", 0.0)) for item in items]

    return {
        "dominant_sentiment": dominant,
        "positive_percentage": round(positive / total * 100, 1) if total else 0.0,
        "negative_percentage": round(negative / total * 100, 1) if total else 0.0,
        "neutral_percentage": round(neutral / total * 100, 1) if total else 0.0,
        "suggestion_percentage": round(suggestion_count / total * 100, 1) if total else 0.0,
        "average_sentiment_score": round(sum(scores) / total, 3) if total else 0.0,
        "top_keywords": top_keywords([item.get("text", "") for item in items]),
    }
