"""M4 routing-arms screen: ONE run = one (arm, S1, seed) at L2048, K=4 (S1=2 unless --s1); or dense.

Protocol (frozen before any result; see docs/protocol-m4-routing-arms.md):
  recipe  m2c_budget_sweep.py's s2-k4 control, unchanged: L2048, 128 KV pairs, vocab 8192,
          d_model 128, 2 layers, lr 3.2e-4, wd 0.1, cosine over the epoch cap, micro-batch 8 x
          grad-accum 2 (effective 16), no clipping, 60 epochs, early stop at valid acc > 0.98.
          This recipe transitioned in both runs that used it (Kaggle 0.726, AWS 0.759).
  arms    baseline | router | maxpool | router+maxpool  (opt-in flags in src/hier_nsa.py)
          | dense (zoology MHA, dropout 0.1, 4 heads: the M3 attention anchor; --s1 ignored)
  s1      --s1 sets the selector budget (super-blocks per query) for hier arms; a design
          variable recorded in the run id and manifest, not a protocol override.
  seed    drives the data draw AND weight init, like M3's SEED knob.
  data    --data-rng legacy (default) retains the as-launched convention; isolated
          seeds CPU noise per segment in a new cache namespace for the next batch.
  splits  train (20k) | validation (2k: early stopping + checkpoint selection only)
          | held-out test (2k, its own seed; never seen by training or selection).
  report  PRIMARY: held-out test accuracy of the best-validation checkpoint (earliest on ties).
          Also: final-checkpoint test accuracy, the validation curve, the jump epoch.
  records one new folder per run, never reused: manifest.json (code/data/hardware provenance,
          incl. whether each data split came from cache -- see cache_hit below),
          metrics.jsonl (per epoch), result.json, train.log, best.pt, final.pt, and progress.json
          (live phase/step/loss, rewritten every few seconds for experiments/m4_dashboard.py).
  cap     --max-hours: checked after each epoch; training stops, the held-out evaluation still
          runs, and the result is flagged time_capped.

Local smoke on CPU (tiny data, output outside the repo):
  ZOOLOGY_DIR=/path/to/patched/zoology python experiments/m4_routing_arms.py --arm router \\
      --epochs 2 --n-train 64 --n-val 32 --n-test 32 --out-root /tmp/m4-smoke --device cpu
"""
import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ARMS = {"baseline": {}, "router": {"router_gate": True}, "maxpool": {"superblock_pool": "max"},
        "router+maxpool": {"router_gate": True, "superblock_pool": "max"}, "dense": None}
DENSE = {"dropout": 0.1, "num_heads": 4}   # M3's attention anchor (m3_length_frontier.py mixer)
PROTOCOL = dict(seq_len=2048, num_kv_pairs=128, vocab_size=8192, d_model=128, n_layers=2,
                learning_rate=3.2e-4, weight_decay=0.1, early_stopping_threshold=0.98,
                s1=2, k_sel=4, compress_block_size=16, sliding_window_size=32, num_heads=4,
                micro_batch=8, grad_accum=2, grad_clip=0.0,
                epochs=60, n_train=20_000, n_val=2_000, n_test=2_000)
HELDOUT_SALT = 0x5EED7E57   # separates the held-out seed stream from zoology's train/val draws


def _args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--s1", type=int, default=PROTOCOL["s1"], help="super-blocks per query (hier arms)")
    ap.add_argument("--epochs", type=int, default=PROTOCOL["epochs"])
    ap.add_argument("--n-train", type=int, default=PROTOCOL["n_train"])
    ap.add_argument("--n-val", type=int, default=PROTOCOL["n_val"])
    ap.add_argument("--n-test", type=int, default=PROTOCOL["n_test"])
    ap.add_argument("--out-root", default=os.path.join(REPO, "experiments", "m4_results", "runs"))
    ap.add_argument("--max-hours", type=float, default=0.0, help="soft wall-clock cap; 0 = off")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--data-rng", choices=["legacy", "isolated"], default="legacy",
                    help="isolated seeds CPU noise per segment in a versioned cache namespace")
    ap.add_argument("--mask-mode", choices=["sentinel", "bool"], default="sentinel",
                    help="hier only: opt-in boolean eligibility and safe masked softmax")
    ap.add_argument("--router-telemetry", action="store_true",
                    help="record detached training-epoch router statistics (hier only)")
    ap.add_argument("--commit", default=None, help="code commit (launchers pass it; else read from git)")
    ap.add_argument("--dirty", type=int, choices=[0, 1], default=None)
    ap.add_argument("--report-run-dir", default=None, help="write the run folder path to this file")
    ap.add_argument("--task", choices=["mqar"], default="mqar")
    ap.add_argument("--precision", choices=["fp32", "tf32"], default="fp32")
    return ap.parse_args(argv)


def configure_precision(torch, precision):
    """Match the historical M5 runner; fp32 preserves the legacy defaults."""
    if precision == "tf32":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True


def _git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "unknown", None


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def certify_protocol(*, arm, s1, data_rng, mixer_name, mixer_kwargs, recipe,
                     overrides, max_hours=0.0, cache_state=None,
                     mask_mode="sentinel", router_telemetry=False):
    """Certify a named computational profile, including its recorded design.

    The isolated profile is a draft for future runs, not a claim that its data
    can be pooled with M4 or authorization to run an unregistered experiment.
    Other budgets remain executable but are not certified by these profiles.
    """
    profiles = {"legacy": "m4-amendment2-legacy-v1",
                "isolated": "m5-isolated-data-draft-v1"}
    dense = arm == "dense"
    expected_mixer = "zoology.mixers.attention.MHA" if dense else "hier_nsa.HierSparseAttention"
    expected_kwargs = dict(DENSE) if dense else {
        "num_heads": PROTOCOL["num_heads"],
        "sliding_window_size": PROTOCOL["sliding_window_size"],
        "compress_block_size": PROTOCOL["compress_block_size"],
        "selection_block_size": PROTOCOL["compress_block_size"],
        "num_selected_blocks": PROTOCOL["k_sel"], "superblock_group": 11,
        "num_selected_superblocks": s1, "causal": True, **(ARMS.get(arm) or {})}
    if not dense and mask_mode != "sentinel": expected_kwargs["mask_mode"] = mask_mode
    if not dense and router_telemetry: expected_kwargs["router_telemetry"] = True
    allowed_s1 = ([2, 12] if data_rng == "legacy" else [2, 4, 6, 12]) if arm == "baseline" else [2]
    mismatches = []
    if data_rng not in profiles: mismatches.append("data_rng")
    if mask_mode not in ("sentinel", "bool"): mismatches.append("mask_mode")
    if not dense and data_rng == "legacy" and mask_mode != "sentinel":
        mismatches.append("mask_mode_not_frozen_legacy_protocol")
    if arm not in ARMS: mismatches.append("arm")
    if (dense and s1 is not None) or (not dense and (
            not isinstance(s1, int) or isinstance(s1, bool) or s1 not in allowed_s1)):
        mismatches.append("s1")
    if mixer_name != expected_mixer: mismatches.append("mixer")
    if mixer_kwargs != expected_kwargs: mismatches.append("mixer_kwargs")
    mismatches.extend("recipe." + key for key, value in PROTOCOL.items()
                      if key != "s1" and recipe.get(key) != value)
    if overrides: mismatches.append("overrides")
    if max_hours != 0.0: mismatches.append("time_cap")
    if data_rng == "legacy" and cache_state is not None and any(
            cache_state.get(split) is not False for split in ("train", "validation", "heldout_test")):
        mismatches.append("legacy_cache_not_all_cold")
    return {"conformant": not mismatches, "profile": profiles.get(data_rng, "unknown"),
            "preregistration_status": "draft" if data_rng == "isolated" else "frozen",
            "expected_design": {"arm": arm, "allowed_s1": [None] if dense else allowed_s1,
                                "mixer": expected_mixer, "mixer_kwargs": expected_kwargs,
                                "dense_dropout": DENSE["dropout"] if dense else None,
                                "data_rng": data_rng, "mask_mode": None if dense else mask_mode,
                                "router_telemetry": False if dense else router_telemetry,
                                "recipe": {**PROTOCOL, "s1": s1}},
            "actual_design": {"arm": arm, "s1": s1, "mixer": mixer_name,
                              "mixer_kwargs": dict(mixer_kwargs),
                              "dense_dropout": mixer_kwargs.get("dropout") if dense else None,
                              "data_rng": data_rng, "recipe": dict(recipe),
                              "mask_mode": None if dense else mask_mode,
                              "router_telemetry": False if dense else router_telemetry,
                              "data_cache_hit_at_start": cache_state},
            "mismatches": mismatches}


def _write_manifest(run_dir, manifest):
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


def screen_trainer(Trainer, nn, optim, on_eval, stop_check):
    """Subclass of zoology's Trainer. fit() is zoology 1ad20d1's Trainer.fit line for line, plus a
    stop_check() hook after each epoch's early-stopping test and a wall-clock timer around
    train_epoch (timing only; no effect on the computation); test() also calls on_eval()."""
    class ScreenTrainer(Trainer):
        def test(self, epoch_idx):
            metrics = super().test(epoch_idx)
            on_eval(self, epoch_idx, metrics)
            return metrics

        def fit(self):
            self.model.to(self.device)
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs,
                                                                  eta_min=0.0)
            for epoch_idx in range(self.max_epochs):
                t_ep = time.time()
                self.train_epoch(epoch_idx)
                self.last_train_epoch_s = time.time() - t_ep
                metrics = self.test(epoch_idx)
                self.epochs_run = epoch_idx + 1
                if (self.early_stopping_metric is not None) and metrics[
                        self.early_stopping_metric] > self.early_stopping_threshold:
                    print(f"Early stopping triggered at epoch {epoch_idx} with "
                          f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > "
                          f"{self.early_stopping_threshold}")
                    self.early_stopped = True
                    break
                if stop_check(self, epoch_idx):
                    break
                self.scheduler.step()
    return ScreenTrainer


def main():
    a = _args()
    P = PROTOCOL
    # The patched trainer reads these at call time; set before zoology is imported.
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["GRAD_ACCUM"] = str(P["grad_accum"])
    os.environ["GRAD_CLIP"] = "0"
    os.environ.setdefault("TQDM_MININTERVAL", "30")   # keeps train.log small
    zdir = os.environ.get("ZOOLOGY_DIR") or os.path.join(REPO, "zoology")
    train_py = os.path.join(zdir, "zoology", "train.py")
    if not os.path.exists(train_py):
        sys.exit(f"zoology not found at {zdir} -- set ZOOLOGY_DIR (pinned 1ad20d1)")
    if "GRAD_CLIP" not in open(train_py).read():
        sys.exit("zoology is not patched: grad accumulation would silently be OFF. "
                 f"Run: python experiments/patch_zoology_accum.py {zdir}")
    sys.path[:0] = [zdir, os.path.join(REPO, "src"), HERE]

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.data.utils import DataSegment, _SyntheticDataset, prepare_data
    from zoology.logger import WandbLogger
    from zoology.model import LanguageModel
    from zoology.train import Trainer, compute_metrics
    from zoology.utils import set_determinism
    from m4_data import (SeededMQARConfig, ISOLATED_CACHE_NAMESPACE,
                         cache_directory, segment_fingerprints)

    configure_precision(torch, a.precision)

    overrides = {k: v for k, v in (("epochs", a.epochs), ("n_train", a.n_train), ("n_val", a.n_val),
                                   ("n_test", a.n_test)) if v != P[k]}
    if a.precision != "fp32":
        overrides["precision"] = a.precision
    nblk = math.ceil(P["seq_len"] / P["compress_block_size"])
    g = max(2, round(math.sqrt(nblk)))
    nsup = math.ceil(nblk / g)
    dense = ARMS[a.arm] is None
    if a.s1 < 1:
        sys.exit("--s1 must be >= 1")
    s1 = None if dense else a.s1
    run_id = (f"m4-L{P['seq_len']}-dense-seed{a.seed}" if dense
              else f"m4-L{P['seq_len']}-s{s1}-k{P['k_sel']}-{a.arm}-seed{a.seed}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(a.out_root, f"{run_id}__{stamp}")
    os.makedirs(run_dir)                      # exist_ok=False: a run folder is never reused
    if a.report_run_dir:
        with open(a.report_run_dir, "w") as f: f.write(run_dir)
    prog = {"run_id": run_id, "arm": a.arm, "seed": a.seed, "phase": "data", "epoch": 0, "epochs": a.epochs,
            "epochs_done": 0, "step": 0, "steps": None, "loss": None, "loss_hist": []}

    def progress(**kw):
        # live status for the dashboard; atomic replace, and never allowed to break the run
        try:
            prog.update(kw, t=time.time())
            tmp = os.path.join(run_dir, ".progress.tmp")
            with open(tmp, "w") as f: json.dump(prog, f)
            os.replace(tmp, os.path.join(run_dir, "progress.json"))
        except Exception:
            pass
    log = open(os.path.join(run_dir, "train.log"), "w", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
    t0 = time.time()
    progress(started=t0)

    # ---- data: zoology's own train/val seed derivation, plus a disjoint held-out seed ----
    MAX_SEED = 2 ** 32
    np.random.seed(a.seed)                                   # exactly as prepare_data does
    train_seed = int(np.random.randint(0, MAX_SEED // 2, size=1)[0])
    val_seed = int(np.random.randint(MAX_SEED // 2, MAX_SEED, size=1)[0])
    heldout_seed = int(np.random.default_rng([a.seed, HELDOUT_SALT]).integers(MAX_SEED // 2, MAX_SEED))
    assert heldout_seed not in (train_seed, val_seed), "held-out seed collided"
    cache = a.cache_dir or ("/content/zoology_cache" if os.access("/content", os.W_OK)
                            else os.path.expanduser("~/.cache/ssa-recall-zoology"))
    cache = cache_directory(cache, a.data_rng)
    MQARClass = SeededMQARConfig if a.data_rng == "isolated" else MQARConfig
    mq = lambda n: MQARClass(num_examples=n, vocab_size=P["vocab_size"], input_seq_len=P["seq_len"],
                            num_kv_pairs=P["num_kv_pairs"])

    # zoology's MQAR generator fills noise tokens with UNSEEDED torch.randint (multiquery_ar.py:137),
    # so a freshly generated dataset consumes the torch RNG while a cached one does not: cold vs warm
    # cache changes both the data and the training trajectory. Record which segments were cached.
    def cache_hit(cfg, seed):
        key = hashlib.md5(json.dumps({**cfg.model_dump(), "_seed": seed}, sort_keys=True).encode()).hexdigest()
        return os.path.exists(os.path.join(cache, f"data_{key}.pt"))
    cache_state = {"train": cache_hit(mq(a.n_train), train_seed), "validation": cache_hit(mq(a.n_val), val_seed),
                   "heldout_test": cache_hit(mq(a.n_test), heldout_seed)}

    experimental_kwargs = {}
    if not dense and a.mask_mode != "sentinel": experimental_kwargs["mask_mode"] = a.mask_mode
    if not dense and a.router_telemetry: experimental_kwargs["router_telemetry"] = True
    mixer = (ModuleConfig(name="zoology.mixers.attention.MHA", kwargs=dict(DENSE)) if dense else
             ModuleConfig(name="hier_nsa.HierSparseAttention", kwargs={
                 "num_heads": P["num_heads"], "sliding_window_size": P["sliding_window_size"],
                 "compress_block_size": P["compress_block_size"],
                 "selection_block_size": P["compress_block_size"], "num_selected_blocks": P["k_sel"],
                 "superblock_group": g, "num_selected_superblocks": s1, "causal": True,
                 **ARMS[a.arm], **experimental_kwargs}))
    config = TrainConfig(
        data=DataConfig(train_configs=[mq(a.n_train)], test_configs=[mq(a.n_val)],
                        batch_size=P["micro_batch"], seed=a.seed, cache_dir=cache),
        model=ModelConfig(vocab_size=P["vocab_size"], max_position_embeddings=P["seq_len"],
                          d_model=P["d_model"], n_layers=P["n_layers"], sequence_mixer=mixer),
        max_epochs=a.epochs, learning_rate=P["learning_rate"], weight_decay=P["weight_decay"],
        early_stopping_metric="valid/accuracy", early_stopping_threshold=P["early_stopping_threshold"],
        seed=a.seed, run_id=run_id, launch_id=stamp)

    effective_recipe = {**P, "s1": s1, "epochs": a.epochs, "n_train": a.n_train,
                        "n_val": a.n_val, "n_test": a.n_test}
    certification = certify_protocol(
        arm=a.arm, s1=s1, data_rng=a.data_rng, mixer_name=mixer.name,
        mixer_kwargs=mixer.kwargs, recipe=effective_recipe,
        overrides=overrides, max_hours=a.max_hours, cache_state=cache_state,
        mask_mode=a.mask_mode, router_telemetry=a.router_telemetry)

    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    commit, dirty = (a.commit, bool(a.dirty) if a.dirty is not None else None) if a.commit else _git_info()
    manifest = {
        "run_id": run_id, "run_folder": os.path.basename(run_dir), "started_utc": stamp,
        "arm": a.arm, "arm_kwargs": DENSE if dense else ARMS[a.arm], "seed": a.seed,
        "mixer": mixer.name, "s1": s1,
        "dense_dropout": mixer.kwargs.get("dropout") if dense else None,
        "task": a.task, "precision": a.precision,
        "data_rng": a.data_rng,
        "mask_mode": None if dense else a.mask_mode,
        "router_telemetry": False if dense else a.router_telemetry,
        "data_config_class": MQARClass.__module__ + "." + MQARClass.__name__,
        "cache_namespace": ISOLATED_CACHE_NAMESPACE if a.data_rng == "isolated" else "legacy",
        "split_hashes": {},
        "data_seeds": {"train": train_seed, "validation": val_seed, "heldout_test": heldout_seed},
        "split_sizes": {"train": a.n_train, "validation": a.n_val, "heldout_test": a.n_test},
        "data_cache_hit_at_start": cache_state, "data_cache_dir": cache,
        "protocol": {**P, "s1": s1, "superblock_group": None if dense else g,
                     "n_superblocks": None if dense else nsup,
                     "coarse_pairs_per_query": None if dense else nsup + min(s1, nsup) * g,
                     "selection_rule": "best validation accuracy, earliest epoch on ties",
                     "primary_metric": "held-out test accuracy of the best-validation checkpoint",
                     "train_data_order": "fixed each epoch (zoology DataLoader shuffle=False)"},
        "protocol_conformant": certification["conformant"],
        "protocol_profile": certification["profile"], "protocol_certification": certification,
        "overrides": overrides, "max_hours": a.max_hours,
        "code": {"commit": commit, "dirty": dirty,
                 "hier_nsa_sha256": _sha256(os.path.join(REPO, "src", "hier_nsa.py")),
                 "script_sha256": _sha256(os.path.abspath(__file__)),
                 "data_helper_sha256": _sha256(os.path.join(HERE, "m4_data.py")),
                 "telemetry_helper_sha256": _sha256(os.path.join(REPO, "src", "hier_telemetry.py")),
                 "zoology_dir": zdir, "zoology_pin": "1ad20d1", "zoology_patched": True},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
                "hostname": socket.gethostname(), "platform": platform.platform(), "argv": sys.argv},
    }
    _write_manifest(run_dir, manifest)
    print(f"[m4] {run_id} -> {run_dir}\n[m4] protocol_conformant={certification['conformant']} overrides={overrides}")

    # From here on, result.json is ALWAYS written (finally), even if setup itself crashes.
    result = {"run_id": run_id, "arm": a.arm, "s1": s1, "seed": a.seed,
              "protocol_conformant": certification["conformant"],
              "protocol_profile": certification["profile"], "data_rng": a.data_rng,
              "task": a.task, "precision": a.precision,
              "status": "started"}
    st = {"task": None, "logger": None}
    try:
        # Held-out set FIRST: set_determinism (next, inside) resets every RNG, so training is
        # exactly what the protocol would have produced without it.
        if a.data_rng == "legacy":
            torch.manual_seed(heldout_seed)   # retains the as-launched held-out RNG behavior
        heldout = DataSegment.from_config(mq(a.n_test), cache_dir=cache, seed=heldout_seed)
        manifest["split_hashes"]["heldout_test"] = segment_fingerprints(heldout)
        _write_manifest(run_dir, manifest)
        _train_and_evaluate(a, P, config, heldout, device, run_dir, t0, result, st, progress, np, torch, nn, optim,
                            DataLoader, _SyntheticDataset, prepare_data, WandbLogger, LanguageModel,
                            Trainer, compute_metrics, set_determinism, manifest)
    except BaseException as e:
        task = st["task"]
        result.update({"status": "failed", "error": f"{type(e).__name__}: {e}",
                       "epochs_run": getattr(task, "epochs_run", 0) if task else 0,
                       "wall_seconds": round(time.time() - t0, 1)})
        raise
    finally:
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        progress(phase=result["status"])
        print("[m4] result " + json.dumps(result))
        if st["logger"] is not None:
            st["logger"].finish()


def _train_and_evaluate(a, P, config, heldout, device, run_dir, t0, result, st, progress, np, torch, nn, optim,
                        DataLoader, _SyntheticDataset, prepare_data, WandbLogger, LanguageModel,
                        Trainer, compute_metrics, set_determinism, manifest=None):
    # ---- training: zoology.train.train() step for step, with the screen trainer ----
    set_determinism(config.seed)
    logger = st["logger"] = WandbLogger(config)
    logger.log_config(config)
    config.print()
    model = LanguageModel(config.model)
    train_dl, val_dl = prepare_data(config.data)
    if manifest is not None:
        from m4_data import segment_fingerprints
        for name, loader in (("train", train_dl), ("validation", val_dl)):
            manifest["split_hashes"][name] = segment_fingerprints(loader.dataset.segments[0])
        _write_manifest(run_dir, manifest)
    logger.log_model(model, config=config)

    # Live progress rides on the trainer's own per-step logger.log({"train/loss": loss.item(), ...}):
    # it only reads a float the trainer already computed, so the computation is unchanged.
    steps = len(train_dl)
    every = max(1, steps // 25)                 # ~25 loss points per epoch in the live curve
    lg = {"ep": None, "n": 0, "wrote": 0.0}
    orig_log = logger.log

    def live_log(d):
        if "train/loss" in d:
            if d.get("epoch") != lg["ep"]:
                lg.update(ep=d.get("epoch"), n=0)
            lg["n"] += 1
            n, ep, loss = lg["n"], lg["ep"] or 0, float(d["train/loss"])
            if n % every == 0 or n == steps:
                prog_hist.append([round(ep + n / steps, 3), round(loss, 4)])
            now = time.time()
            if now - lg["wrote"] > 2 or n == steps:
                lg["wrote"] = now
                progress(phase="training" if n < steps else "validating", epoch=ep, step=n, steps=steps,
                         loss=loss, loss_hist=prog_hist)
        return orig_log(d)
    prog_hist = []
    logger.log = live_log
    test_dl = DataLoader(_SyntheticDataset([heldout], batch_size=P["micro_batch"]),
                         batch_size=None, num_workers=0, shuffle=False)

    def rowhashes(t):
        return {hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in t.cpu().numpy()}
    tr, va, te = (rowhashes(dl.dataset.segments[0].inputs) for dl in (train_dl, val_dl, test_dl))
    result["split_overlap"] = {"heldout_in_train": len(te & tr), "heldout_in_validation": len(te & va),
                               "validation_in_train": len(va & tr)}
    print(f"[m4] split overlap (identical sequences): {result['split_overlap']}")

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best = {"acc": -1.0, "epoch": None, "state": None}
    curve = []

    def on_eval(tr_, epoch_idx, m):
        acc = float(m["valid/accuracy"])
        curve.append(acc)
        rec = {"epoch": epoch_idx, "valid_accuracy": acc, "valid_loss": float(m["valid/loss"]),
               "lr": tr_.optimizer.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1),
               "train_epoch_s": round(getattr(tr_, "last_train_epoch_s", float("nan")), 2)}
        if a.router_telemetry:
            rec["router_telemetry"] = {
                name: module.consume_router_telemetry()
                for name, module in tr_.model.named_modules()
                if hasattr(module, "consume_router_telemetry")}
            if not rec["router_telemetry"]:
                rec["router_telemetry_status"] = "not_applicable_to_dense"
        with open(metrics_path, "a") as f: f.write(json.dumps(rec) + "\n")
        progress(phase="training", epochs_done=epoch_idx + 1, valid_accuracy=acc)
        if acc > best["acc"]:     # strict: the earliest epoch wins ties
            best.update(acc=acc, epoch=epoch_idx,
                        state={k: v.detach().to("cpu", copy=True) for k, v in tr_.model.state_dict().items()})

    cap_s = a.max_hours * 3600

    def stop_check(tr_, epoch_idx):
        if cap_s and time.time() - t0 > cap_s:
            print(f"[m4] time cap: {a.max_hours} h reached after epoch {epoch_idx}; stopping training")
            tr_.time_capped = True
            return True
        return False

    TrainerCls = screen_trainer(Trainer, nn, optim, on_eval, stop_check)
    task = st["task"] = TrainerCls(
        model=model, train_dataloader=train_dl, test_dataloader=val_dl, input_type=config.input_type,
        max_epochs=config.max_epochs, learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold, slice_keys=config.slice_keys,
        loss_type=config.loss_type, device=device, logger=logger)
    task.epochs_run, task.early_stopped, task.time_capped = 0, False, False

    def evaluate(loader):
        task.model.eval()
        losses, res = [], []
        with torch.no_grad():
            for inputs, targets, slices in loader:
                inputs, targets = inputs.to(task.device), targets.to(task.device)
                loss, preds = task.compute_loss(inputs, targets)
                losses.append(loss.item())
                res.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
        return float(np.mean([r["accuracy"] for r in res])), float(np.mean(losses))

    task.fit()
    progress(phase="held-out test")
    final_state = {k: v.detach().to("cpu", copy=True) for k, v in task.model.state_dict().items()}
    torch.save(final_state, os.path.join(run_dir, "final.pt"))
    test_final = evaluate(test_dl)
    task.model.load_state_dict(best["state"])
    torch.save(best["state"], os.path.join(run_dir, "best.pt"))
    test_best = evaluate(test_dl)
    jump = next((i for i, v in enumerate(curve) if v > 0.30), None)
    result.update({
        "status": "complete", "epochs_run": task.epochs_run, "early_stopped": task.early_stopped,
        "time_capped": task.time_capped,
        "PRIMARY_test_accuracy_at_best_validation": test_best[0],
        "best_validation_accuracy": best["acc"], "best_validation_epoch": best["epoch"],
        "test_loss_at_best_validation": test_best[1],
        "final_validation_accuracy": curve[-1], "test_accuracy_final": test_final[0],
        "final_five_validation_mean": float(np.mean(curve[-5:])),
        "jump_epoch_validation_above_0.30": jump, "wall_seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
