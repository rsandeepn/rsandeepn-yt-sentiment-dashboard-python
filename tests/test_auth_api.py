import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///file:auth_tests?mode=memory&cache=shared&uri=true"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-that-is-not-used-outside-tests"

fake_agent = types.ModuleType("agent")
failure_videos = set()


def fake_analyze(url, progress_callback=None):
    if progress_callback:
        progress_callback(15, "Fetching comments")
        progress_callback(45, "Analyzing sentiment")
    if any(video_id in url for video_id in failure_videos):
        raise ValueError("No comments are available for this video.")
    return {
        "video": {
            "id": url.rsplit("/", 1)[-1],
            "title": "Filmymoji Middle Class Madhu Kotha AC MCM",
            "url": url,
        },
        "stats": {"total": 1, "positive": 1, "negative": 0, "neutral": 0, "suggestions": 0},
        "overview": "Positive feedback is dominant.",
        "all_comments": [],
    }


fake_agent.analyze_comments = fake_analyze
sys.modules.setdefault("agent", fake_agent)

from fastapi.testclient import TestClient
import jwt

from database import Base, engine
from main import app


class AuthApiTests(unittest.TestCase):
    def setUp(self):
        failure_videos.clear()
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def register(self, email="person@example.com", first_name="Sandeep", last_name="Rongali"):
        return self.client.post(
            "/auth/register",
            json={
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "password": "secure-password",
                "confirm_password": "secure-password",
            },
        )

    def test_register_login_and_current_user(self):
        registered = self.register("Person@Example.com")
        self.assertEqual(registered.status_code, 201)
        self.assertEqual(registered.json()["user"]["email"], "person@example.com")
        self.assertEqual(registered.json()["user"]["first_name"], "Sandeep")
        self.assertEqual(registered.json()["user"]["last_name"], "Rongali")
        self.assertNotIn("password", registered.json()["user"])

        token = registered.json()["access_token"]
        current = self.client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(current.status_code, 200)

        logged_in = self.client.post(
            "/auth/login",
            json={"email": "person@example.com", "password": "secure-password"},
        )
        self.assertEqual(logged_in.status_code, 200)

    def test_registration_rejects_mismatched_passwords(self):
        response = self.client.post(
            "/auth/register",
            json={
                "first_name": "Sandeep",
                "last_name": "Rongali",
                "email": "person@example.com",
                "password": "secure-password",
                "confirm_password": "different-password",
            },
        )
        self.assertEqual(response.status_code, 422)

    @patch("main.verify_google_credential")
    def test_google_login_creates_and_reuses_verified_user(self, verify):
        verify.return_value = {
            "email": "Google.User@Example.com",
            "email_verified": True,
            "given_name": "Google",
            "family_name": "User",
        }
        first = self.client.post("/auth/google", json={"credential": "google-id-token"})
        second = self.client.post("/auth/google", json={"credential": "google-id-token"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["user"]["email"], "google.user@example.com")
        self.assertEqual(first.json()["user"]["first_name"], "Google")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["user"]["id"], first.json()["user"]["id"])

    def test_duplicate_registration_and_bad_login(self):
        self.assertEqual(self.register().status_code, 201)
        self.assertEqual(self.register().status_code, 409)
        bad_login = self.client.post(
            "/auth/login",
            json={"email": "person@example.com", "password": "wrong-password"},
        )
        self.assertEqual(bad_login.status_code, 401)

    @patch("main.deliver_password_reset_email_safely")
    def test_password_reset_changes_password_invalidates_sessions_and_is_single_use(self, send):
        registered = self.register("reset@example.com")
        original_token = registered.json()["access_token"]

        requested = self.client.post(
            "/auth/forgot-password", json={"email": "Reset@Example.com"}
        )
        self.assertEqual(requested.status_code, 200)
        self.assertEqual(
            requested.json()["message"],
            "If an account exists for that email, a password reset link has been sent.",
        )
        send.assert_called_once()
        reset_link = send.call_args.args[1]
        raw_token = parse_qs(urlparse(reset_link).query)["token"][0]

        reset = self.client.post(
            "/auth/reset-password",
            json={
                "token": raw_token,
                "password": "new-secure-password",
                "confirm_password": "new-secure-password",
            },
        )
        self.assertEqual(reset.status_code, 200)

        old_session = self.client.get(
            "/auth/me", headers={"Authorization": f"Bearer {original_token}"}
        )
        self.assertEqual(old_session.status_code, 401)
        self.assertEqual(
            self.client.post(
                "/auth/login",
                json={"email": "reset@example.com", "password": "secure-password"},
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/auth/login",
                json={"email": "reset@example.com", "password": "new-secure-password"},
            ).status_code,
            200,
        )

        reused = self.client.post(
            "/auth/reset-password",
            json={
                "token": raw_token,
                "password": "another-password",
                "confirm_password": "another-password",
            },
        )
        self.assertEqual(reused.status_code, 400)

    @patch("main.deliver_password_reset_email_safely")
    def test_forgot_password_does_not_reveal_unknown_email(self, send):
        response = self.client.post(
            "/auth/forgot-password", json={"email": "missing@example.com"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["message"],
            "If an account exists for that email, a password reset link has been sent.",
        )
        send.assert_not_called()

    def test_expired_token_is_rejected(self):
        token = jwt.encode(
            {
                "sub": "not-important",
                "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
            },
            os.environ["JWT_SECRET_KEY"],
            algorithm="HS256",
        )
        response = self.client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 401)

    def test_analyze_requires_authentication_and_creates_history(self):
        url = "https://www.youtube.com/shorts/q9rt-hDD4AY"
        self.assertEqual(self.client.post("/analyze", json={"url": url}).status_code, 401)

        token = self.register().json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        analyzed = self.client.post("/analyses", json={"url": url}, headers=headers)
        self.assertEqual(analyzed.status_code, 202)
        completed = self.client.get(
            f"/analyses/{analyzed.json()['id']}", headers=headers
        )
        self.assertEqual(completed.json()["status"], "completed")

        history = self.client.get("/analyses", headers=headers)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["total"], 1)
        self.assertEqual(history.json()["items"][0]["video_id"], "q9rt-hDD4AY")
        self.assertEqual(history.json()["items"][0]["platform"], "youtube")
        self.assertEqual(history.json()["items"][0]["content_type"], "video")
        self.assertEqual(history.json()["items"][0]["content_id"], "q9rt-hDD4AY")
        self.assertEqual(history.json()["items"][0]["content_url"], url)
        self.assertEqual(
            history.json()["items"][0]["video_title"],
            "Filmymoji Middle Class Madhu Kotha AC MCM",
        )
        self.assertEqual(
            history.json()["items"][0]["content_title"],
            "Filmymoji Middle Class Madhu Kotha AC MCM",
        )

        detail = self.client.get(
            f"/analyses/{history.json()['items'][0]['id']}", headers=headers
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["result"]["stats"]["total"], 1)

    def test_instagram_platform_is_reserved_until_account_connection_is_available(self):
        token = self.register().json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = self.client.post(
            "/analyses",
            json={
                "url": "https://www.instagram.com/reel/example/",
                "platform": "instagram",
            },
            headers=headers,
        )

        self.assertEqual(response.status_code, 501)
        self.assertIn("Professional account", response.json()["detail"])
        self.assertEqual(self.client.get("/analyses", headers=headers).json()["total"], 0)

    def test_history_supports_platform_filtering(self):
        token = self.register().json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        self.client.post(
            "/analyses",
            json={"url": "https://youtu.be/q9rt-hDD4AY"},
            headers=headers,
        )

        youtube = self.client.get(
            "/analyses", params={"platform": "youtube"}, headers=headers
        )
        instagram = self.client.get(
            "/analyses", params={"platform": "instagram"}, headers=headers
        )

        self.assertEqual(youtube.status_code, 200)
        self.assertEqual(youtube.json()["total"], 1)
        self.assertEqual(instagram.status_code, 200)
        self.assertEqual(instagram.json()["total"], 0)

    def test_users_cannot_read_each_others_history(self):
        first_token = self.register("first@example.com").json()["access_token"]
        first_headers = {"Authorization": f"Bearer {first_token}"}
        self.client.post(
            "/analyses",
            json={"url": "https://youtu.be/q9rt-hDD4AY"},
            headers=first_headers,
        )
        analysis_id = self.client.get("/analyses", headers=first_headers).json()["items"][0]["id"]

        second_token = self.register("second@example.com").json()["access_token"]
        response = self.client.get(
            f"/analyses/{analysis_id}",
            headers={"Authorization": f"Bearer {second_token}"},
        )
        self.assertEqual(response.status_code, 404)

    def test_duplicate_reuse_and_explicit_reanalysis(self):
        token = self.register().json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"url": "https://youtu.be/q9rt-hDD4AY"}
        first = self.client.post("/analyses", json=payload, headers=headers).json()
        duplicate = self.client.post("/analyses", json=payload, headers=headers).json()
        self.assertEqual(duplicate["id"], first["id"])

        reanalyzed = self.client.post(
            f"/analyses/{first['id']}/reanalyze", headers=headers
        )
        self.assertEqual(reanalyzed.status_code, 200)
        self.assertNotEqual(reanalyzed.json()["id"], first["id"])
        self.assertEqual(self.client.get("/analyses", headers=headers).json()["total"], 2)

    def test_failed_job_can_be_retried_and_deleted(self):
        token = self.register().json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        failure_videos.add("failed1")
        queued = self.client.post(
            "/analyses",
            json={"url": "https://youtu.be/failed1"},
            headers=headers,
        ).json()
        failed = self.client.get(
            f"/analyses/{queued['id']}", headers=headers
        ).json()
        self.assertEqual(failed["status"], "failed")
        self.assertIn("No comments", failed["error_message"])

        failure_videos.clear()
        retried = self.client.post(
            f"/analyses/{failed['id']}/retry", headers=headers
        )
        self.assertEqual(retried.status_code, 200)
        completed = self.client.get(
            f"/analyses/{failed['id']}", headers=headers
        )
        self.assertEqual(completed.json()["status"], "completed")

        deleted = self.client.delete(f"/analyses/{failed['id']}", headers=headers)
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get("/analyses", headers=headers).json()["total"], 0)

    def test_history_search_status_and_pagination(self):
        token = self.register().json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        for video_id in ("video01", "video02", "other03"):
            self.client.post(
                "/analyses",
                json={"url": f"https://youtu.be/{video_id}"},
                headers=headers,
            )

        response = self.client.get(
            "/analyses", params={"search": "video", "status": "completed", "page_size": 1}, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual(response.json()["total_pages"], 2)
        self.assertEqual(len(response.json()["items"]), 1)


if __name__ == "__main__":
    unittest.main()
