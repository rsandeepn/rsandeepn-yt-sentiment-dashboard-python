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

SCRIPT_RANGES = (
    ("Telugu", 0x0C00, 0x0C7F),
    ("Devanagari", 0x0900, 0x097F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Bengali", 0x0980, 0x09FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("CJK", 0x4E00, 0x9FFF),
)


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


def dominant_script(text):
    counts = Counter()
    for char in text:
        codepoint = ord(char)
        if ("a" <= char.lower() <= "z"):
            counts["Latin"] += 1
            continue
        for name, start, end in SCRIPT_RANGES:
            if start <= codepoint <= end:
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else "Other"


def language_breakdown(texts):
    counts = Counter(dominant_script(text) for text in texts)
    total = len(texts)
    return [
        {
            "language": language,
            "count": count,
            "percentage": round(count / total * 100, 1) if total else 0.0,
        }
        for language, count in counts.most_common()
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
        "language_breakdown": language_breakdown(
            [item.get("text", "") for item in items]
        ),
    }
