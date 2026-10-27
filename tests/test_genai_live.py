"""Tests for Generative AI Explainer (live SDK initialization, mock responses, and offline fallback)."""
from unittest.mock import MagicMock, patch
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qms.genai.explainer import GenAIExplainer, load_api_key, get_explainer


class TestGenAILiveExplainer(unittest.TestCase):

    def setUp(self):
        self.orig_key = os.environ.get("GEMINI_API_KEY")

    def tearDown(self):
        if self.orig_key is not None:
            os.environ["GEMINI_API_KEY"] = self.orig_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_load_api_key_env(self):
        os.environ["GEMINI_API_KEY"] = "test_env_key_xyz"
        key = load_api_key()
        self.assertEqual(key, "test_env_key_xyz")

    def test_offline_mode_when_no_key(self):
        os.environ.pop("GEMINI_API_KEY", None)
        with patch("qms.genai.explainer._parse_env_file", return_value=None):
            exp = GenAIExplainer()
            self.assertEqual(exp.mode, "offline-template")
            self.assertIn("No GEMINI_API_KEY", exp.last_error)
            ok, msg = exp.test_connection()
            self.assertFalse(ok)
            self.assertIn("Offline mode", msg)

    def test_google_genai_sdk_mock(self):
        os.environ["GEMINI_API_KEY"] = "mock_key_123"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This appointment has elevated no-show risk due to prior history."
        mock_client.models.generate_content.return_value = mock_response

        mock_genai_mod = MagicMock()
        mock_genai_mod.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google.genai": mock_genai_mod}):
            exp = GenAIExplainer()
            exp.set_api_key("mock_key_123")
            self.assertEqual(exp.mode, "gemini")
            self.assertEqual(exp._sdk, "google.genai")

            txt = exp.explain_no_show({"probability": 0.45, "drivers": ["prior no-show"]})
            self.assertEqual(txt, "This appointment has elevated no-show risk due to prior history.")
            mock_client.models.generate_content.assert_called()

            # Verify generation config was passed
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            self.assertIn("config", call_kwargs)
            self.assertEqual(call_kwargs["config"]["max_output_tokens"], 400)
            self.assertEqual(call_kwargs["config"]["temperature"], 0.4)

    def test_connection_test_mock_success(self):
        os.environ["GEMINI_API_KEY"] = "mock_key_123"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_client.models.generate_content.return_value = mock_response

        mock_genai_mod = MagicMock()
        mock_genai_mod.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google.genai": mock_genai_mod}):
            exp = GenAIExplainer()
            exp.set_api_key("mock_key_123")
            ok, msg = exp.test_connection()
            self.assertTrue(ok)
            self.assertIn("Connected to Gemini", msg)

    def test_model_fallback_on_api_error(self):
        os.environ["GEMINI_API_KEY"] = "mock_key_123"
        mock_client = MagicMock()

        def side_effect(model, contents, config):
            if model == "gemini-2.0-flash":
                raise RuntimeError("Quota exceeded for 2.0")
            resp = MagicMock()
            resp.text = "Fallback response from 1.5 flash"
            return resp

        mock_client.models.generate_content.side_effect = side_effect

        mock_genai_mod = MagicMock()
        mock_genai_mod.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google.genai": mock_genai_mod}):
            exp = GenAIExplainer(model_name="gemini-2.0-flash")
            exp.set_api_key("mock_key_123")
            txt = exp.explain_no_show({"probability": 0.40})
            self.assertEqual(txt, "Fallback response from 1.5 flash")

    def test_runtime_set_model(self):
        exp = GenAIExplainer()
        exp.set_model("gemini-1.5-pro")
        self.assertEqual(exp.model_name, "gemini-1.5-pro")


if __name__ == "__main__":
    unittest.main()
