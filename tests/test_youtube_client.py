import unittest
from unittest.mock import patch

from youtube_client import _fetch_replies, extract_video_id


class YouTubeClientTests(unittest.TestCase):
    def test_extracts_supported_video_urls(self):
        video_id = "96XB-q2-0qo"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://youtu.be/{video_id}",
            f"https://youtube.com/shorts/{video_id}",
            f"https://youtube.com/embed/{video_id}",
        ]

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), video_id)

    @patch("youtube_client._call_youtube_api")
    def test_reply_fetching_paginates_and_respects_limit(self, api_call):
        api_call.side_effect = [
            {
                "items": [
                    {"snippet": {"textDisplay": "first"}},
                    {"snippet": {"textDisplay": "second"}},
                ],
                "nextPageToken": "page-2",
            },
            {"items": [{"snippet": {"textDisplay": "third"}}]},
        ]

        replies = _fetch_replies("parent-id", max_replies=3)

        self.assertEqual([item["text"] for item in replies], ["first", "second", "third"])
        self.assertEqual(api_call.call_count, 2)


if __name__ == "__main__":
    unittest.main()
