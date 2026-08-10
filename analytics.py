import re
from collections import Counter


TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

STOPWORDS = {
    "a", "about", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "don", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "like", "make",
    "me", "more", "most", "my", "myself", "need", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "please", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "video", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "thank", "thanks",
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
