import unittest

from analytics import build_dashboard_insights, dominant_script, top_keywords


class DashboardAnalyticsTests(unittest.TestCase):
    def test_keywords_include_counts_and_exclude_common_words(self):
        keywords = top_keywords([
            "The music is excellent and the acting is excellent",
            "Excellent music",
        ], limit=3)
        self.assertEqual(keywords[0], {"keyword": "excellent", "count": 3})
        self.assertEqual(keywords[1], {"keyword": "music", "count": 2})

    def test_dominant_script_supports_indian_scripts(self):
        self.assertEqual(dominant_script("చాలా బాగుంది"), "Telugu")
        self.assertEqual(dominant_script("बहुत अच्छा"), "Devanagari")
        self.assertEqual(dominant_script("super video"), "Latin")

    def test_dashboard_percentages_and_language_breakdown(self):
        insights = build_dashboard_insights([
            {"text": "Great music", "sentiment": "positive", "score": 0.8},
            {"text": "చాలా బాగుంది", "sentiment": "positive", "score": 0.7},
            {"text": "Not useful", "sentiment": "negative", "score": -0.6},
            {"text": "Information", "sentiment": "neutral", "score": 0.0},
        ], suggestion_count=1)
        self.assertEqual(insights["dominant_sentiment"], "positive")
        self.assertEqual(insights["positive_percentage"], 50.0)
        self.assertEqual(insights["suggestion_percentage"], 25.0)
        self.assertEqual(sum(row["count"] for row in insights["language_breakdown"]), 4)


if __name__ == "__main__":
    unittest.main()
