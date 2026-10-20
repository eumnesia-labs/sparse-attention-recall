"""CPU regressions for opt-in isolated MQAR data and protocol certification.

    ZOOLOGY_DIR=/path/to/patched/zoology python experiments/test_m4_data_rng.py

No GPU or Modal calls. All generated data, run artifacts, and caches are temporary.
The unchanged test_m4_harness.py separately covers legacy trainer equivalence H1--H10.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ZDIR = Path(os.environ.get("ZOOLOGY_DIR", str(REPO / "zoology")))
sys.path[:0] = [str(ZDIR), str(REPO / "src")]
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.update(GRAD_ACCUM="2", GRAD_CLIP="0", TQDM_DISABLE="1")

import torch
from zoology.config import DataConfig, ModelConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.data.utils import DataSegment, prepare_data
import zoology.model as zm
from m4_data import (ISOLATED_CACHE_NAMESPACE, SeededMQARConfig, cache_directory,
                     segment_fingerprints, tensor_fingerprint)
import m4_routing_arms as harness


# Pinned zoology defaults embeddings to CUDA. Retarget those constructor defaults
# in these CPU tests regardless of whether this machine happens to have a GPU.
for _, cls in inspect.getmembers(zm, inspect.isclass):
    defaults = getattr(cls.__init__, "__defaults__", None)
    if defaults and "cuda" in defaults:
        cls.__init__.__defaults__ = tuple("cpu" if item == "cuda" else item for item in defaults)
torch.set_num_threads(2)


def config(cls=SeededMQARConfig, n=16):
    return cls(num_examples=n, vocab_size=512, input_seq_len=128, num_kv_pairs=8)


def same_segments(test, left, right):
    test.assertTrue(torch.equal(left.inputs, right.inputs), "inputs differ")
    test.assertTrue(torch.equal(left.labels, right.labels), "targets differ")


def hierarchy_kwargs(s1=2, arm="baseline"):
    return {"num_heads": 4, "sliding_window_size": 32, "compress_block_size": 16,
            "selection_block_size": 16, "num_selected_blocks": 4,
            "superblock_group": 11, "num_selected_superblocks": s1,
            "causal": True, **(harness.ARMS.get(arm) or {})}


class DataRNGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m4-cpu-data-rng-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def run_harness(self, script, label, *, convention=None):
        out, cache = self.folder / label / "runs", self.folder / label / "cache"
        report = self.folder / f"{label}-run-path.txt"
        shim = ("import inspect,runpy,sys,torch,zoology.model as zm\n"
                "torch.set_num_threads(2)\n"
                "for _,c in inspect.getmembers(zm,inspect.isclass):\n"
                " d=getattr(c.__init__,'__defaults__',None)\n"
                " if d and 'cuda' in d: c.__init__.__defaults__=tuple('cpu' if v=='cuda' else v for v in d)\n"
                "sys.argv=sys.argv[1:]\n"
                "loaded=runpy.run_path(sys.argv[0],run_name='m4_cpu_comparison')\n"
                f"loaded['main'].__globals__['REPO']={str(REPO)!r}\n"
                "loaded['main']()\n")
        command = [sys.executable, "-c", shim, str(script),
                   "--arm", "baseline", "--device", "cpu",
                   "--epochs", "1", "--n-train", "8", "--n-val", "4", "--n-test", "4",
                   "--out-root", str(out), "--cache-dir", str(cache),
                   "--report-run-dir", str(report)]
        if convention is not None:
            command += ["--data-rng", convention]
        env = dict(os.environ, ZOOLOGY_DIR=str(ZDIR), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", PYTHONPATH=f"{ZDIR}{os.pathsep}{HERE}{os.pathsep}{REPO / 'src'}")
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(completed.returncode, 0, (completed.stdout + completed.stderr)[-8000:])
        return Path(report.read_text()), cache

    def test_isolated_noise_is_independent_of_ambient_model_draws(self):
        torch.manual_seed(4)
        first = config().build(seed=231)
        torch.rand(1300)
        second = config().build(seed=231)
        same_segments(self, first, second)
        third = config().build(seed=312)
        self.assertFalse(torch.equal(first.inputs, third.inputs))
        self.assertFalse(torch.equal(first.labels, third.labels))

    def test_dense_and_hier_models_have_identical_cold_isolated_splits(self):
        models, streams, segments = [], [], []
        for label in ("dense", "hier"):
            torch.manual_seed(123)
            mixer = (ModuleConfig(name="zoology.mixers.attention.MHA",
                                  kwargs={"num_heads": 4, "dropout": 0.1})
                     if label == "dense" else
                     ModuleConfig(name="hier_nsa.HierSparseAttention", kwargs=hierarchy_kwargs()))
            model = zm.LanguageModel(ModelConfig(vocab_size=512, max_position_embeddings=128,
                                                   d_model=32, n_layers=1,
                                                   sequence_mixer=mixer))
            models.append(model)
            state = torch.get_rng_state().clone()
            streams.append(state)
            data = DataConfig(train_configs=[config()], test_configs=[config(n=8)],
                              batch_size=4, seed=123,
                              cache_dir=cache_directory(str(self.folder / label), "isolated"))
            train_dl, validation_dl = prepare_data(data)
            heldout = DataSegment.from_config(config(n=8), cache_dir=data.cache_dir, seed=999)
            self.assertTrue(torch.equal(state, torch.get_rng_state()), label)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
            segments.append([train_dl.dataset.segments[0], validation_dl.dataset.segments[0], heldout])
        self.assertFalse(torch.equal(streams[0], streams[1]),
                         "fixture must exercise distinct post-model RNG states")
        for left, right in zip(*segments):
            same_segments(self, left, right)

    def test_cold_warm_and_partial_cache_preserve_data_and_rng(self):
        cache = cache_directory(str(self.folder), "isolated")
        cfg = DataConfig(train_configs=[config()], test_configs=[config(n=8)],
                         batch_size=4, seed=123, cache_dir=cache)
        def load():
            state = torch.get_rng_state().clone()
            train_dl, validation_dl = prepare_data(cfg)
            heldout = DataSegment.from_config(config(n=8), cache_dir=cache, seed=999)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            return [train_dl.dataset.segments[0], validation_dl.dataset.segments[0], heldout]
        cold = load()
        files = sorted(Path(cache).glob("data_*.pt"))
        self.assertEqual(len(files), 3)
        warm = load()
        files[1].unlink()
        torch.rand(500)
        partial = load()
        self.assertEqual(len(list(Path(cache).glob("data_*.pt"))), 3)
        for reference, warmed, mixed in zip(cold, warm, partial):
            same_segments(self, reference, warmed)
            same_segments(self, reference, mixed)

    def test_local_cpu_seed_never_calls_cuda_seed_or_rng_apis(self):
        torch.manual_seed(5)
        state = torch.get_rng_state().clone()
        with patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA seed touched")), \
             patch.object(torch.cuda, "get_rng_state", side_effect=AssertionError("CUDA RNG read")), \
             patch.object(torch.cuda, "set_rng_state", side_effect=AssertionError("CUDA RNG written")), \
             patch.object(torch.cuda, "device_count", side_effect=AssertionError("CUDA enumerated")):
            segment = config().build(seed=312)
        self.assertEqual(segment.inputs.device.type, "cpu")
        self.assertEqual(segment.labels.device.type, "cpu")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_cpu_rng_is_restored_when_generation_raises(self):
        state = torch.get_rng_state().clone()
        def fail(*args, **kwargs):
            torch.rand(100)
            raise RuntimeError("synthetic generator failure")
        with patch.object(MQARConfig, "build", fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic generator failure"):
                config().build(seed=123)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_legacy_keeps_original_namespace_data_and_rng_consumption(self):
        base = str(self.folder)
        self.assertEqual(cache_directory(base, "legacy"), base)
        self.assertEqual(cache_directory(base, "isolated"), str(self.folder / ISOLATED_CACHE_NAMESPACE))
        self.assertIn("v1", ISOLATED_CACHE_NAMESPACE)
        with self.assertRaises(ValueError):
            cache_directory(base, "unknown")
        torch.manual_seed(123)
        reference = config(MQARConfig).build(seed=231)
        reference_state = torch.get_rng_state().clone()
        torch.manual_seed(123)
        generated = DataSegment.from_config(config(MQARConfig), cache_dir=base, seed=231)
        same_segments(self, reference, generated)
        self.assertTrue(torch.equal(reference_state, torch.get_rng_state()))
        before = torch.get_rng_state().clone()
        cached = DataSegment.from_config(config(MQARConfig), cache_dir=base, seed=231)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        same_segments(self, generated, cached)
        with patch.object(sys, "argv", ["m4_routing_arms.py", "--arm", "baseline"]):
            self.assertEqual(harness._args().data_rng, "legacy")

    def test_isolated_cache_key_and_namespace_do_not_reuse_legacy_data(self):
        base = str(self.folder)
        torch.manual_seed(100)
        legacy = DataSegment.from_config(config(MQARConfig), cache_dir=base, seed=123)
        isolated_dir = cache_directory(base, "isolated")
        isolated = DataSegment.from_config(config(), cache_dir=isolated_dir, seed=123)
        self.assertEqual(len(list(self.folder.glob("data_*.pt"))), 1)
        self.assertEqual(len(list(Path(isolated_dir).glob("data_*.pt"))), 1)
        self.assertNotEqual(config(MQARConfig).model_dump(), config().model_dump())
        self.assertFalse(torch.equal(legacy.inputs, isolated.inputs))
        self.assertTrue(torch.equal(legacy.labels, isolated.labels))

    def test_fingerprints_cover_inputs_targets_dtype_shape_and_bytes(self):
        segment = config().build(seed=123)
        state = torch.get_rng_state().clone()
        hashes = segment_fingerprints(segment)
        self.assertEqual(set(hashes), {"inputs", "targets"})
        for name, tensor in (("inputs", segment.inputs), ("targets", segment.labels)):
            record = hashes[name]
            self.assertEqual(record["shape"], list(tensor.shape))
            self.assertEqual(record["dtype"], str(tensor.dtype))
            self.assertEqual(record["sha256"], hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            self.assertEqual(len(record["sha256"]), 64)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        segment.labels[0, 0] += 1
        changed_targets = segment_fingerprints(segment)
        self.assertEqual(hashes["inputs"], changed_targets["inputs"])
        self.assertNotEqual(hashes["targets"], changed_targets["targets"])
        segment.inputs[0, 0] += 1
        self.assertNotEqual(hashes["inputs"], segment_fingerprints(segment)["inputs"])
        non_contiguous = segment.inputs.t()
        self.assertEqual(tensor_fingerprint(non_contiguous),
                         tensor_fingerprint(non_contiguous.contiguous()))

    def test_manifest_records_all_three_real_split_hashes_and_data_profile(self):
        run_dir, _ = self.run_harness(HERE / "m4_routing_arms.py", "isolated", convention="isolated")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        result = json.loads((run_dir / "result.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["protocol_conformant"])
        self.assertFalse(manifest["protocol_conformant"])
        self.assertEqual(manifest["data_rng"], "isolated")
        self.assertEqual(manifest["cache_namespace"], ISOLATED_CACHE_NAMESPACE)
        self.assertIn("isolated", manifest["protocol_profile"])
        self.assertEqual(set(manifest["split_hashes"]), {"train", "validation", "heldout_test"})
        for split in ("train", "validation", "heldout_test"):
            cfg = SeededMQARConfig(num_examples=manifest["split_sizes"][split], vocab_size=8192,
                                   input_seq_len=2048, num_kv_pairs=128)
            segment = DataSegment.from_config(cfg, cache_dir=manifest["data_cache_dir"],
                                              seed=manifest["data_seeds"][split])
            self.assertEqual(manifest["split_hashes"][split], segment_fingerprints(segment))
        self.assertEqual(result["protocol_profile"], manifest["protocol_profile"])
        self.assertEqual(result["data_rng"], manifest["data_rng"])
        self.assertFalse(manifest["protocol_certification"]["conformant"])

    def test_legacy_full_harness_data_curves_and_weights_match_pre_t2(self):
        original = subprocess.run(["git", "show", "d35a5ad:experiments/m4_routing_arms.py"],
                                  cwd=REPO, capture_output=True, text=True, check=True).stdout
        old_script = self.folder / "pre-t2-m4.py"
        old_script.write_text(original)
        old_run, old_cache = self.run_harness(old_script, "before")
        # Omitting the new flag explicitly verifies that the default still follows legacy.
        new_run, new_cache = self.run_harness(HERE / "m4_routing_arms.py", "after")
        old_files = {path.name: path for path in old_cache.glob("data_*.pt")}
        new_files = {path.name: path for path in new_cache.glob("data_*.pt")}
        self.assertEqual(set(old_files), set(new_files))
        self.assertEqual(len(old_files), 3)
        for name in old_files:
            old_data, new_data = torch.load(old_files[name]), torch.load(new_files[name])
            self.assertTrue(torch.equal(old_data["inputs"], new_data["inputs"]), name)
            self.assertTrue(torch.equal(old_data["labels"], new_data["labels"]), name)
        def measurements(run):
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            return [{key: row[key] for key in ("epoch", "valid_accuracy", "valid_loss", "lr")}
                    for row in rows]
        self.assertEqual(measurements(old_run), measurements(new_run))
        for checkpoint in ("best.pt", "final.pt"):
            old_state, new_state = torch.load(old_run / checkpoint), torch.load(new_run / checkpoint)
            self.assertEqual(set(old_state), set(new_state))
            for key in old_state:
                self.assertTrue(torch.equal(old_state[key], new_state[key]), f"{checkpoint}: {key}")
        old_result = json.loads((old_run / "result.json").read_text())
        new_result = json.loads((new_run / "result.json").read_text())
        for key in ("PRIMARY_test_accuracy_at_best_validation", "test_accuracy_final",
                    "best_validation_accuracy", "best_validation_epoch", "final_validation_accuracy",
                    "final_five_validation_mean", "jump_epoch_validation_above_0.30", "split_overlap"):
            self.assertEqual(old_result[key], new_result[key], key)


class ProtocolCertificationTests(unittest.TestCase):
    def certify(self, **changes):
        arguments = dict(arm="baseline", s1=2, data_rng="legacy",
                         mixer_name="hier_nsa.HierSparseAttention",
                         mixer_kwargs=hierarchy_kwargs(), recipe=dict(harness.PROTOCOL),
                         overrides={}, max_hours=0.0)
        arguments.update(changes)
        return harness.certify_protocol(**arguments)

    def test_design_and_data_profiles_are_explicit_and_distinct(self):
        legacy = self.certify()
        isolated = self.certify(data_rng="isolated")
        self.assertTrue(legacy["conformant"])
        self.assertTrue(isolated["conformant"])
        self.assertNotEqual(legacy["profile"], isolated["profile"])
        self.assertEqual(legacy["preregistration_status"], "frozen")
        self.assertEqual(isolated["preregistration_status"], "draft")
        self.assertEqual(isolated["actual_design"]["data_rng"], "isolated")
        self.assertFalse(self.certify(data_rng="ambient-unknown")["conformant"])

    def test_arm_mixer_s1_and_dense_dropout_mismatches_are_not_certified(self):
        cases = [dict(arm="router"), dict(arm="unknown"),
                 dict(mixer_name="zoology.mixers.attention.MHA"),
                 dict(s1=12), dict(s1=3, mixer_kwargs=hierarchy_kwargs(s1=3)),
                 dict(s1=True),
                 dict(arm="dense", s1=None, mixer_name="zoology.mixers.attention.MHA",
                      mixer_kwargs={"num_heads": 4, "dropout": 0.2})]
        for change in cases:
            with self.subTest(change=change):
                certification = self.certify(**change)
                self.assertFalse(certification["conformant"])
                self.assertTrue(certification["mismatches"])
        dense = self.certify(arm="dense", s1=None, mixer_name="zoology.mixers.attention.MHA",
                             mixer_kwargs={"num_heads": 4, "dropout": 0.1})
        self.assertTrue(dense["conformant"])
        self.assertEqual(dense["actual_design"]["dense_dropout"], 0.1)
        self.assertEqual(dense["expected_design"]["dense_dropout"], 0.1)

    def test_registered_budgets_and_arm_flags_are_certified(self):
        for mode, budgets in (("legacy", [2, 12]), ("isolated", [2, 4, 6, 12])):
            for s1 in budgets:
                with self.subTest(mode=mode, s1=s1):
                    self.assertTrue(self.certify(s1=s1, data_rng=mode,
                                                mixer_kwargs=hierarchy_kwargs(s1=s1))["conformant"])
        for arm in ("router", "maxpool", "router+maxpool"):
            self.assertTrue(self.certify(arm=arm, mixer_kwargs=hierarchy_kwargs(arm=arm))["conformant"])
            self.assertFalse(self.certify(arm=arm, s1=12,
                                         mixer_kwargs=hierarchy_kwargs(s1=12, arm=arm))["conformant"])

    def test_epochs_split_overrides_and_time_caps_are_not_certified(self):
        for key, value in (("epochs", 1), ("n_train", 8), ("n_val", 4), ("n_test", 4)):
            with self.subTest(key=key):
                recipe = dict(harness.PROTOCOL, **{key: value})
                certification = self.certify(recipe=recipe, overrides={key: value})
                self.assertFalse(certification["conformant"])
                self.assertIn("recipe." + key, certification["mismatches"])
                self.assertIn("overrides", certification["mismatches"])
                self.assertEqual(certification["actual_design"]["recipe"][key], value)
        self.assertFalse(self.certify(max_hours=0.001)["conformant"])

    def test_legacy_requires_cold_cache_and_isolated_allows_warm_or_partial(self):
        cold = {"train": False, "validation": False, "heldout_test": False}
        self.assertTrue(self.certify(cache_state=cold)["conformant"])
        for state in ({"train": True, "validation": True, "heldout_test": True},
                      {"train": False, "validation": True, "heldout_test": False}):
            legacy = self.certify(cache_state=state)
            self.assertFalse(legacy["conformant"])
            self.assertIn("legacy_cache_not_all_cold", legacy["mismatches"])
            self.assertTrue(self.certify(data_rng="isolated", cache_state=state)["conformant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
