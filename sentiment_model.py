# sentiment_model.py
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER once
nltk.download("vader_lexicon", quiet=True)

print("⚡ Loading lightweight multilingual sentiment model...")

vader = SentimentIntensityAnalyzer()

# Common Indian-language positive/negative words
INDIAN_SENTIMENT = {
    "positive": [
        "super",
        "bagundi",
        "chala bagundi",
        "awesome",
        "semma",
        "superb",
        "mass",
        "class",
        "nalla",
        "vallare",
        "bahut accha",
        "accha",
        "mast",
        "jhakaas",
        "super anna",
        "blockbuster",
        "super hit",
        "hit",
        "pogaru",
    ],
    "negative": [
        "worst",
        "boring",
        "waste",
        "thopu",
        "worst scene",
        "not good",
        "pichachi",
        "bad",
        "dabba",
        "falthu",
        "bekaar",
        "idhu sari illa",
    ],
}

# Emoji sentiment
POS_EMOJI = ["😍", "❤️", "🔥", "💥", "😊", "😁", "👍"]
NEG_EMOJI = ["😡", "💔", "👎", "😢", "😭", "🤬"]


def normalize(text: str) -> str:
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_sentiment(text: str):
    """
    Returns (label, score) where:
      - label is 'positive' / 'negative' / 'neutral'
      - score is a float (VADER compound or heuristic)
    """
    if not text:
        return "neutral", 0.0

    t = normalize(text)

    # 1) Emoji sentiment (strong signal)
    if any(e in t for e in POS_EMOJI):
        return "positive", 0.9
    if any(e in t for e in NEG_EMOJI):
        return "negative", -0.9

    # 2) Indian-language keyword matching
    for w in INDIAN_SENTIMENT["positive"]:
        if w in t:
            return "positive", 0.7
    for w in INDIAN_SENTIMENT["negative"]:
        if w in t:
            return "negative", -0.7

    # 3) VADER (handles English + transliterated text)
    score = vader.polarity_scores(t)["compound"]

    if score > 0.25:
        return "positive", score
    elif score < -0.25:
        return "negative", score
    else:
        return "neutral", score


def classify_sentiment_batch(text_list, batch_size: int = 512):
    """
    Returns a list of dicts:
      [
        {"text": "...", "sentiment": "positive", "score": 0.73},
        ...
      ]
    """
    results = []
    for text in text_list:
        label, score = classify_sentiment(text)
        results.append(
            {
                "text": text,
                "sentiment": label,
                "score": float(score),
            }
        )
    return results
