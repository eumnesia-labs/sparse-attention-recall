"""Dependency-free checks of recipe translation and historical precision behavior."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from m4_routing_arms import _args, configure_precision
from reproduce_m5 import command


class ReproductionTests(unittest.TestCase):
    def test_all_recorded_recipes_parse_without_transport_or_false_provenance(self):
        manifests = list(Path(__file__).with_name("m5_results").rglob("manifest.json"))
        self.assertEqual(len(manifests), 20)
        for path in manifests:
            with self.subTest(path=path):
                m = json.loads(path.read_text())
                argv = command(m, "/tmp/new results", "/tmp/fresh cache")
                a = _args(argv[2:])
                for field in ("arm", "seed", "task", "precision", "data_rng"):
                    self.assertEqual(getattr(a, field), m[field])
                self.assertIsNone(a.commit)
                self.assertIsNone(a.dirty)
                self.assertNotIn("--sync-cmd", argv)
                if a.arm != "dense":
                    self.assertEqual(a.s1, m["s1"])
                    self.assertEqual(a.router_telemetry, m["router_telemetry"])

    def test_reject_changed_recipe(self):
        path = next(Path(__file__).with_name("m5_results").rglob("manifest.json"))
        original = json.loads(path.read_text())
        for field, value in (("learning_rate", 0.1), ("grad_accum", 1), ("epochs", 2)):
            m = copy.deepcopy(original)
            m["protocol"][field] = value
            with self.assertRaises(ValueError):
                command(m, "/tmp/out", "/tmp/cache")
        m = copy.deepcopy(original)
        m["overrides"]["unknown"] = True
        with self.assertRaises(ValueError):
            command(m, "/tmp/out", "/tmp/cache")

    def test_precision_matches_historical_branch(self):
        calls = []
        torch = SimpleNamespace(set_float32_matmul_precision=calls.append,
                                backends=SimpleNamespace(cudnn=SimpleNamespace(allow_tf32=False)))
        configure_precision(torch, "fp32")
        self.assertEqual(calls, [])
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        configure_precision(torch, "tf32")
        self.assertEqual(calls, ["high"])
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertEqual(_args(["--arm", "baseline"]).precision, "fp32")


if __name__ == "__main__":
    unittest.main()
