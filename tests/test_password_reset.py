import os
import unittest
from unittest.mock import Mock, patch

from password_reset import reset_url, send_password_reset_email, token_digest


class PasswordResetServiceTests(unittest.TestCase):
    def test_token_digest_does_not_store_raw_token(self):
        digest = token_digest("raw-reset-token")
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, "raw-reset-token")

    @patch("password_reset.requests.post")
    def test_resend_request_uses_configured_sender_and_secure_link(self, post):
        response = Mock()
        post.return_value = response
        environment = {
            "FRONTEND_BASE_URL": "https://analyzeytcomments.com/",
            "PASSWORD_RESET_EXPIRE_MINUTES": "30",
            "PASSWORD_RESET_LOG_LINK": "false",
            "RESEND_API_KEY": "test-api-key",
            "PASSWORD_RESET_FROM_EMAIL": (
                "CommentScope <no-reply@mail.analyzeytcomments.com>"
            ),
        }
        with patch.dict(os.environ, environment, clear=False):
            url = reset_url("token-value")
            send_password_reset_email("person@example.com", url)

        self.assertEqual(
            url,
            "https://analyzeytcomments.com/reset-password?token=token-value",
        )
        request = post.call_args
        self.assertEqual(request.args[0], "https://api.resend.com/emails")
        self.assertEqual(
            request.kwargs["headers"]["Authorization"], "Bearer test-api-key"
        )
        self.assertEqual(request.kwargs["json"]["to"], ["person@example.com"])
        self.assertIn("reset-password?token=token-value", request.kwargs["json"]["html"])
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
