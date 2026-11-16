import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

from scripts.performance_preflight import choose_checkpointing, compare_gradients, fingerprint, load_cached, preserve_probe_state


class PerformanceTests(unittest.TestCase):
    def test_rejects_fast_but_incorrect_or_unsafe_candidate(self):
        baseline = {"mean_forward_backward_seconds": 2.0}
        fast = {"safe": True, "gradients_match": True, "mean_forward_backward_seconds": 1.0}
        self.assertFalse(choose_checkpointing(baseline, fast))
        for change in ({"safe": False}, {"gradients_match": False},
                       {"mean_forward_backward_seconds": float("nan")},
                       {"mean_forward_backward_seconds": 1.95}):
            self.assertTrue(choose_checkpointing(baseline, {**fast, **change}))
        self.assertTrue(choose_checkpointing(baseline, None))

    def test_cache_rejects_changed_environment_expiry_and_corruption(self):
        key = fingerprint({"dataset": "a", "torch": "vendor", "max_length": 448})
        changed = fingerprint({"dataset": "b", "torch": "vendor", "max_length": 448})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.json"
            path.write_text(json.dumps({"key": key, "created_at": 100, "checkpointing": False}))
            self.assertFalse(load_cached(path, key, now=101)["checkpointing"])
            self.assertIsNone(load_cached(path, changed, now=101))
            self.assertIsNone(load_cached(path, key, now=99))
            self.assertIsNone(load_cached(path, key, now=100 + 7 * 86400))
            path.write_text("partial write")
            self.assertIsNone(load_cached(path, key, now=101))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_gradient_guard_rejects_scaling_errors_hidden_by_clipping(self):
        import torch
        reference = [torch.ones(10000), torch.ones(2) * 1e-4]
        self.assertTrue(compare_gradients(reference, reference, 2.0, 2.0)["gradients_match"])
        scaled = [x * 1.01 for x in reference]
        self.assertFalse(compare_gradients(scaled, reference, 2.0, 2.0)["gradients_match"])
        tiny_tensor_error = [reference[0], reference[1] * 1.2]
        self.assertFalse(compare_gradients(tiny_tensor_error, reference, 2.0, 2.0)["gradients_match"])
        self.assertFalse(compare_gradients(reference, reference, 2.1, 2.0)["gradients_match"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_gradient_guard_handles_small_near_zero_rounding(self):
        import torch
        reference = [torch.ones(10000)]
        reference[0][0] = 0
        actual = [reference[0].clone()]
        actual[0][0] = 4e-5
        result = compare_gradients(actual, reference, 2.0, 2.0)
        self.assertFalse(result["raw_elementwise_match"])
        self.assertTrue(result["gradients_match"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_probe_restores_rng_buffers_modes_and_gradients_after_failure(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout(0.5))
        model.register_buffer("counter", torch.tensor(0))
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.manual_seed(17)
        random.seed(17)
        expected_torch, expected_python = torch.rand(3), random.random()
        torch.manual_seed(17)
        random.seed(17)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            with preserve_probe_state(model):
                model(torch.randn(2, 4)).sum().backward()
                model.counter.add_(1)
                model.eval()
                random.random()
                raise RuntimeError("simulated probe failure")
        self.assertTrue(all(m.training for m in model.modules()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertEqual(random.random(), expected_python)


if __name__ == "__main__":
    unittest.main()
