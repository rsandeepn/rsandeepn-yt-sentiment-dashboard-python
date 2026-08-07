from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_labels(comments: list[str], num_clusters: int) -> list[int]:
    """Group multilingual text with lightweight character n-gram features."""
    features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
    ).fit_transform(comments)
    labels = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init="auto",
    ).fit_predict(features)
    return [int(label) for label in labels]
