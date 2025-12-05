from sentiment_model import classify_sentiment

samples = [
    "ఈ వీడియో చాలా బాగుంది bro!",        # Telugu - positive
    "Not good, very boring.",            # English - negative
    "Movie ok ok.",                      # English - neutral-ish
    "Song super anna 🔥🔥",              # Tanglish - positive
    "Worst scene ever",                  # English - negative
    "Bahut accha laga",                  # Hindi - positive
    "Idhu sari illa",                    # Tamil - negative
]

for text in samples:
    label = classify_sentiment(text)
    print(f"{text} → {label}")
