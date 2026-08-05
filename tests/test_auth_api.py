import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///file:auth_tests?mode=memory&cache=shared&uri=true"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-that-is-not-used-outside-tests"

fake_agent = types.ModuleType("agent")
fake_agent.analyze_comments = lambda _url: {
    "stats": {"total": 1, "positive": 1, "negative": 0, "neutral": 0, "suggestions": 0},
    "overview": "Positive feedback is dominant.",
    "comments": [],
}
sys.modules.setdefault("agent", fake_agent)

from fastapi.testclient import TestClient
import jwt

from database import Base, engine
from main import app


class AuthApiTests(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def register(self, email="person@example.com"):
        return self.client.post(
            "/auth/register",
            json={"email": email, "password": "secure-password"},
        )

    def test_register_login_and_current_user(self):
        registered = self.register("Person@Example.com")
        self.assertEqual(registered.status_code, 201)
        self.assertEqual(registered.json()["user"]["email"], "person@example.com")
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

    def test_duplicate_registration_and_bad_login(self):
        self.assertEqual(self.register().status_code, 201)
        self.assertEqual(self.register().status_code, 409)
        bad_login = self.client.post(
            "/auth/login",
            json={"email": "person@example.com", "password": "wrong-password"},
        )
        self.assertEqual(bad_login.status_code, 401)

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
        analyzed = self.client.post("/analyze", json={"url": url}, headers=headers)
        self.assertEqual(analyzed.status_code, 200)

        history = self.client.get("/analyses", headers=headers)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()), 1)
        self.assertEqual(history.json()[0]["video_id"], "q9rt-hDD4AY")

        detail = self.client.get(f"/analyses/{history.json()[0]['id']}", headers=headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["result"]["stats"]["total"], 1)

    def test_users_cannot_read_each_others_history(self):
        first_token = self.register("first@example.com").json()["access_token"]
        first_headers = {"Authorization": f"Bearer {first_token}"}
        self.client.post(
            "/analyze",
            json={"url": "https://youtu.be/q9rt-hDD4AY"},
            headers=first_headers,
        )
        analysis_id = self.client.get("/analyses", headers=first_headers).json()[0]["id"]

        second_token = self.register("second@example.com").json()["access_token"]
        response = self.client.get(
            f"/analyses/{analysis_id}",
            headers={"Authorization": f"Bearer {second_token}"},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
