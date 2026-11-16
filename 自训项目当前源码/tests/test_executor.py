import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_experiment import ROOT, execute, run_child, validate_request

BASE = json.loads((ROOT / "experiments/request.example.json").read_text())


class ExecutorTests(unittest.TestCase):
    def test_ppu_request_bounds(self):
        request = {**BASE, "task": "qwen3_lora", "batch_size": 18, "device": "cuda"}
        validate_request(request)
        for field, value in (("batch_size", 9), ("device", "cpu"), ("epochs", 6)):
            with self.assertRaises(ValueError):
                validate_request({**request, field: value})

    def test_rejects_unbounded_or_executable_requests(self):
        for field, value in (("epochs", 1000), ("batch_size", True), ("max_minutes", float("nan")),
                             ("learning_rate", 0), ("experiment_id", "../escape"),
                             ("task", "shell"), ("enabled", "false")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request({**BASE, field: value})
        with self.assertRaises(ValueError):
            validate_request({**BASE, "command": "echo bad"})

    def run_request(self, request, directory):
        path = Path(directory) / "request.json"
        path.write_text(json.dumps(request))
        code = execute(path, Path(directory) / "reports")
        return code, json.loads((Path(directory) / "reports/latest.json").read_text())

    def test_disabled_needs_no_child_and_replaces_stale_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch("scripts.run_experiment.run_child") as child:
            code, report = self.run_request({**BASE, "enabled": False}, tmp)
            self.assertEqual((code, report["status"]), (0, "skipped"))
            self.assertEqual(len(report["request_sha256"]), 64)
            child.assert_not_called()
            code, report = self.run_request({**BASE, "epochs": -1}, tmp)
            self.assertEqual((code, report["status"]), (1, "failed"))
            self.assertEqual(report["metrics"], {})

    def test_child_failure_and_timeout_are_reported(self):
        for effect in (7, subprocess.TimeoutExpired("training", 1)):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"GPU_RUNNER_OUTPUT_ROOT": tmp}), \
                    patch("scripts.run_experiment.run_child", side_effect=effect if isinstance(effect, Exception) else None,
                          return_value=effect if isinstance(effect, int) else None):
                code, report = self.run_request(BASE, tmp)
                self.assertEqual((code, report["status"]), (1, "failed"))
                self.assertTrue(report["error"])

    def test_repeated_experiment_id_does_not_start_second_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'GPU_RUNNER_OUTPUT_ROOT':tmp}), \
                patch('scripts.run_experiment.run_child',return_value=7) as child:
            first_code,first=self.run_request(BASE,tmp)
            code,second=self.run_request(BASE,tmp)
            self.assertEqual(first['status'],'failed')
            self.assertEqual((code,second['status'],second['reason']),(0,'skipped','experiment_id_already_claimed'))
            self.assertEqual(child.call_count,1)
            self.assertEqual(second['previous_run_id'],first['run_id'])

    def test_timeout_stops_real_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "finished.txt"
            command = [sys.executable, "-c", "import time,pathlib; time.sleep(5); pathlib.Path(__import__('sys').argv[1]).touch()", str(marker)]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_child(command, Path(tmp) / "train.log", 0.2)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_cpu_training_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"GPU_RUNNER_OUTPUT_ROOT": tmp}):
            code, report = self.run_request({**BASE, "device": "cpu", "epochs": 2}, tmp)
            self.assertEqual(code, 0, report)
            self.assertEqual(report["status"], "completed")
            self.assertTrue(Path(report["checkpoint"]).is_file())
            self.assertEqual(report["epochs_completed"], 2)
            self.assertGreater(report["metrics"]["best_val_accuracy"], 0.25)
            import torch
            checkpoint = torch.load(report["checkpoint"], weights_only=True)
            self.assertIn("model_state_dict", checkpoint)
            self.assertEqual(checkpoint["epoch"], report["metrics"]["best_epoch"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_cuda_unavailable_does_not_fall_back(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest("Requires a CPU-only test environment")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"GPU_RUNNER_OUTPUT_ROOT": tmp}):
            code, report = self.run_request({**BASE, "device": "cuda"}, tmp)
            self.assertEqual((code, report["status"]), (1, "failed"))
            self.assertIsNone(report["checkpoint"])
            self.assertEqual(report["error_type"], "cuda_unavailable")
            self.assertIn("CUDA was requested", report["error"])


if __name__ == "__main__":
    unittest.main()
