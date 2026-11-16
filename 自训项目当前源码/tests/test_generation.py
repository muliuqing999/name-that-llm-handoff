import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.generate_data import collect, digest, enabled_models, export, load_prompts

ROOT = Path(__file__).resolve().parents[1]


class GenerationTests(unittest.TestCase):
    def catalog(self):
        catalog = json.loads((ROOT / "generation/models.example.json").read_text())
        catalog["gpt-5.5"].update(enabled=True, verified_exact_version=True, version_evidence="provider-version-record",
                                  api_model="verified-snapshot", expected_response_models=["verified-snapshot"])
        return catalog

    def test_unverified_versions_are_not_callable(self):
        catalog = self.catalog()
        catalog["gpt-5.5"]["verified_exact_version"] = False
        with self.assertRaises(ValueError):
            enabled_models(catalog)

    def test_inline_credentials_are_rejected(self):
        catalog = self.catalog()
        catalog["gpt-5.5"]["api_key"] = "test-inline-secret"
        with self.assertRaises(ValueError):
            enabled_models(catalog)

    def test_duplicate_prompts_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text('\n'.join(json.dumps(x) for x in [
                {"prompt_id": "a", "prompt": "Write about rain", "split": "train"},
                {"prompt_id": "b", "prompt": " write  ABOUT rain ", "split": "validation"}]))
            with self.assertRaises(ValueError):
                load_prompts(path)

    def test_version_mismatch_is_quarantined_and_resume_does_not_rebill(self):
        catalog = self.catalog()
        prompt = {"prompt_id": "a", "prompt": "Explain rain", "split": "train"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}), \
                patch("scripts.generate_data.fetch", return_value=({"model": "wrong-version"}, "Answer", "wrong-version", "stop")) as call:
            collect(catalog, [prompt], Path(tmp), 1)
            collect(catalog, [prompt], Path(tmp), 1)
            self.assertEqual(call.call_count, 1)
            row = json.loads(next((Path(tmp) / "raw").glob("*.json")).read_text())
            self.assertEqual(row["status"], "quarantined_model_version")
            self.assertNotIn("test-only", json.dumps(row))
            export(Path(tmp))
            report = json.loads((Path(tmp) / "quality-report.json").read_text())
            self.assertEqual(report["exported"], 0)

    def test_hard_call_limit(self):
        prompts = [{"prompt_id": str(i), "prompt": str(i), "split": "train"} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}), \
                patch("scripts.generate_data.fetch", return_value=({}, "Answer", "verified-snapshot", "stop")) as call:
            collect(self.catalog(), prompts, Path(tmp), 2)
            self.assertEqual(call.call_count, 2)

    def test_configuration_change_creates_different_job_identity(self):
        self.assertNotEqual(digest({"temperature": 0.5}), digest({"temperature": 1.0}))


if __name__ == "__main__":
    unittest.main()
