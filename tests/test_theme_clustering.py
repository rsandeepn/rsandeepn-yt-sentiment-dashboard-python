import unittest

from theme_clustering import cluster_labels


class ThemeClusteringTests(unittest.TestCase):
    def test_clusters_multilingual_and_duplicate_comments(self):
        comments = [
            "చాలా బాగుంది",
            "చాలా బాగుంది",
            "super video anna",
            "please improve the audio",
        ]

        unique_count = len(set(comments))
        labels = cluster_labels(comments, num_clusters=unique_count)

        self.assertEqual(len(labels), len(comments))
        self.assertGreaterEqual(len(set(labels)), 1)
        self.assertLessEqual(len(set(labels)), unique_count)


if __name__ == "__main__":
    unittest.main()
