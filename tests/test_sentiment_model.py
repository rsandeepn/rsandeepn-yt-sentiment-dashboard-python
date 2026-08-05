import unittest

from sentiment_model import classify_sentiment


class SentimentModelTests(unittest.TestCase):
    def test_thopu_is_positive_telugu_slang(self):
        label, score = classify_sentiment("This song is thopu")

        self.assertEqual(label, "positive")
        self.assertGreater(score, 0)

    def test_keyword_matching_uses_word_boundaries(self):
        label, _ = classify_sentiment("The white background is plain")

        self.assertNotEqual(label, "positive")


if __name__ == "__main__":
    unittest.main()
