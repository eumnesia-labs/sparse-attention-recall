"""Modal launcher for the M4 routing-arms screen (docs/protocol-m4-routing-arms.md).

Three modes, from the repo root. Nothing is spent unless you pass --smoke or --go.

  0. plan only, no Modal contact, $0:
       .venv/bin/python experiments/modal_m4_arms.py
  1. smoke on one L40S (~$0.25): builds the image, runs every arm for 2 small epochs at the
     REAL batch shape (micro-batch 8 x L2048), and projects each full run's time and cost:
       .venv/bin/python -m modal run experiments/modal_m4_arms.py --smoke
  2. the screen: 4 arms x seed 123, one L40S container each, in parallel, as INDEPENDENT jobs:
       .venv/bin/python experiments/modal_m4_arms.py --launch
     Deploys the app, then spawns one job per arm and exits. The jobs belong to Modal, not to this
     machine: closing the lid, losing Wi-Fi or shutting down cannot stop them. Refuses a dirty tree
     (provenance) unless --allow-dirty. Collect results any time (safe to repeat):
       .venv/bin/python experiments/modal_m4_arms.py --collect
     (`modal run ... --go` is retired: its blocking starmap was tied to the local client, and a lid
     close cancelled all four runs of screen-20260930T044345Z at epoch ~13.)

Run length: NO soft time cap -- every run trains its full protocol (60 epochs, or zoology's own
early stop at validation > 0.98), because a truncated run is not a valid result. The only guard
is a HARD Modal timeout (HARD_TIMEOUT_H, ~2.4x a healthy run) that kills a hung container so it
cannot bill forever; a killed run is incomplete and is rerun, never reported.
(The old m2c launcher used a 24 h timeout, ~$47 per stuck run.)

Data caches stay inside each container (/content), never on the shared volume, so every arm
generates its data cold. Under zoology's unseeded noise-token RNG, this is what makes the arms
receive identical data and identical initial weights (a paired comparison).

Live view (terminal, $0): in a second terminal run
  .venv/bin/python experiments/m4_dashboard.py
Each container relays its run's progress.json + metrics to the Modal Dict `ssa-recall-m4-live`
every few seconds; closing the dashboard never affects the runs.

Results: each run folder is written to the Modal volume `ssa-recall-m4-runs`, and result +
manifest come back to this machine as experiments/m4_results/screen-<utc>.json. Full folders
(logs, metrics, checkpoints) can be downloaded with
  .venv/bin/python -m modal volume get ssa-recall-m4-runs runs experiments/m4_results/
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import modal
import modal.exception

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ARMS = ["baseline", "router", "maxpool", "router+maxpool"]

ZOO_COMMIT = "1ad20d1"
TORCH, TORCHVISION = "2.14.0", "0.29.0"          # the versions the CPU tests ran against
GPU_RATE, CPU_CORES, MEM_GIB = 1.95, 2.0, 12      # Modal list prices 2026-09 (modal billing rates)
HOURLY = GPU_RATE + CPU_CORES * 0.0473 + MEM_GIB * 0.008    # ~$2.14 / h per GPU container
EXPECTED_H = 2.5                                  # 60 epochs x ~2.4 min (M2c logs, same recipe) + setup
SOFT_CAP_H = 0.0                                  # off: runs always complete the protocol
HARD_TIMEOUT_H = 6.0                              # kills only a hung container (healthy run ~2.5 h)
EPOCHS = 60
VOLUME = "ssa-recall-m4-runs"
LIVE = "ssa-recall-m4-live"
FULL = {"n_train": 20_000, "n_val": 2_000, "n_test": 2_000}
# 2 epochs x 250 steps: epoch 2 is timed warm (epoch 1 carries CUDA start-up), so the projection
# is not inflated by warm-up the way a single 50-step epoch was.
SMOKE = {"epochs": 2, "n_train": 2_000, "n_val": 200, "n_test": 200}
BUDGET = 30.0                                     # Modal credit for this month (Oct 2026)

# A run = (arm, s1, seed). Plans name the run sets; --runs "arm[:s1]@seed,..." gives any other set.
PLANS = {
    "screen": [("baseline", 2, 123), ("router", 2, 123), ("maxpool", 2, 123), ("router+maxpool", 2, 123)],
    # 2026-10-01 "variance first": dense reference, S1=2 seed spread, the S1=12 top of the budget curve
    "variance": [("dense", None, 123), ("baseline", 2, 231), ("baseline", 2, 312), ("baseline", 12, 123)],
}


def _spec(arm, s1, seed):
    s1 = None if arm == "dense" else int(s1 or 2)
    label = f"dense-seed{seed}" if arm == "dense" else f"{arm}-s{s1}-seed{seed}"
    return {"arm": arm, "s1": s1, "seed": int(seed), "label": label}


def parse_runs(plan="variance", runs=""):
    if runs:
        out = []
        for tok in (t.strip() for t in runs.split(",") if t.strip()):
            head, _, seed = tok.partition("@")
            arm, _, s1 = head.partition(":")
            out.append((arm, s1 or None, seed or 123))
    elif plan in PLANS:
        out = PLANS[plan]
    else:
        raise SystemExit(f"unknown plan {plan!r}; choose from {list(PLANS)} or pass --runs")
    specs = [_spec(*r) for r in out]
    bad = [x["arm"] for x in specs if x["arm"] not in ARMS + ["dense"]]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {ARMS + ['dense']}")
    if len({x["label"] for x in specs}) != len(specs):
        raise SystemExit("duplicate runs in the plan")
    return specs


def configure_specs(specs, data_rng="legacy", mask_mode="sentinel", router_telemetry=False):
    """Pure configuration only. Defaults retain the original run specifications."""
    if data_rng not in ("legacy", "isolated") or mask_mode not in ("sentinel", "bool"):
        raise ValueError("invalid data or mask convention")
    configured = []
    for original in specs:
        spec = dict(original)
        suffixes = []
        if data_rng != "legacy":
            spec["data_rng"] = data_rng
            suffixes.append(data_rng)
        if spec["arm"] != "dense":
            if mask_mode != "sentinel":
                spec["mask_mode"] = mask_mode
                suffixes.append(mask_mode)
            if router_telemetry:
                spec["router_telemetry"] = True
                suffixes.append("telemetry")
        if suffixes: spec["label"] += "-" + "-".join(suffixes)
        configured.append(spec)
    return configured


def harness_options(spec):
    """Pure argv builder, tested without importing or evaluating Modal."""
    options = []
    if spec.get("data_rng", "legacy") != "legacy":
        options += ["--data-rng", spec["data_rng"]]
    if spec["arm"] != "dense":
        if spec.get("mask_mode", "sentinel") != "sentinel":
            options += ["--mask-mode", spec["mask_mode"]]
        if spec.get("router_telemetry", False): options.append("--router-telemetry")
    return options

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(f"torch=={TORCH}", f"torchvision=={TORCHVISION}", "numpy", "pandas", "einops",
                 "pydantic", "tqdm", "rich", "wandb")
    .run_commands(f"git clone https://github.com/HazyResearch/zoology.git /zoology && "
                  f"cd /zoology && git checkout -q {ZOO_COMMIT}")
    .add_local_file(os.path.join(HERE, "patch_zoology_accum.py"), "/opt/patch_zoology_accum.py", copy=True)
    .run_commands("python /opt/patch_zoology_accum.py /zoology && grep -q GRAD_CLIP /zoology/zoology/train.py")
    .env({"WANDB_MODE": "disabled", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
          "ZOOLOGY_DIR": "/zoology", "TQDM_MININTERVAL": "30"})
    .add_local_dir(os.path.join(REPO, "src"), "/repo/src")
    .add_local_file(os.path.join(HERE, "m4_routing_arms.py"), "/repo/experiments/m4_routing_arms.py")
    .add_local_file(os.path.join(HERE, "m4_data.py"), "/repo/experiments/m4_data.py")
)
app = modal.App("ssa-recall-m4-arms", image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
live = modal.Dict.from_name(LIVE, create_if_missing=True)


def _read_state(run_dir):
    st = {}
    for name in ("progress", "result"):
        p = os.path.join(run_dir, f"{name}.json")
        try:
            st[name] = json.load(open(p))
        except Exception:
            pass
    try:
        st["metrics"] = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    except Exception:
        pass
    return st


def _relay(key, rd, base, stop):
    """Push this run's live state to the Modal Dict every 3 s (dashboard only; never fails the run)."""
    while True:
        done = stop.wait(3)
        try:
            st = dict(base, t=time.time())
            if os.path.exists(rd):
                st.update(_read_state(open(rd).read().strip()))
            live[key] = st
        except Exception:
            pass
        if done:
            return


def _run(spec, extra, out_root, commit, dirty, launch, container_start):
    """One run in this container. Cache stays container-local (/content); outputs go to the volume."""
    arm, seed, label = spec["arm"], spec["seed"], spec["label"]
    os.makedirs("/content", exist_ok=True)
    os.makedirs(out_root, exist_ok=True)
    rd = f"/tmp/rundir-{label.replace('+', '_')}.txt"
    s1 = [] if spec["s1"] is None else ["--s1", str(spec["s1"])]
    cmd = [sys.executable, "/repo/experiments/m4_routing_arms.py", "--arm", arm, "--seed", str(seed), *s1,
           "--out-root", out_root, "--commit", commit, "--dirty", str(int(bool(dirty))),
           "--report-run-dir", rd, "--device", "cuda", "--cache-dir", "/content/zoology_cache",
           *harness_options(spec), *extra]
    base = {"arm": arm, "s1": spec["s1"], "seed": seed, "label": label,
            "container": os.environ.get("MODAL_TASK_ID", "?"),
            "container_start": container_start}
    stop = threading.Event()
    relay = threading.Thread(target=_relay, args=(f"{launch}|{label}", rd, base, stop), daemon=True)
    relay.start()
    rc = subprocess.run(cmd).returncode
    stop.set()
    relay.join(timeout=30)
    out = {"arm": arm, "s1": spec["s1"], "seed": seed, "label": label, "returncode": rc}
    if os.path.exists(rd):
        run_dir = open(rd).read().strip()
        out["run_dir"] = run_dir
        for name in ("result", "manifest"):
            p = os.path.join(run_dir, f"{name}.json")
            if os.path.exists(p):
                out[name] = json.load(open(p))
        mp = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(mp):
            out["metrics"] = [json.loads(l) for l in open(mp)]
    vol.commit()
    return out


@app.function(gpu="L40S", cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=int(HARD_TIMEOUT_H * 3600),
              volumes={"/vol": vol})
def run_arm(spec: dict, commit: str, dirty: bool, launch: str = ""):
    t = time.time()
    return _run(spec, ["--epochs", str(EPOCHS), "--max-hours", str(SOFT_CAP_H)], "/vol/runs", commit, dirty,  # 0 = off
                launch, t)


@app.function(gpu="L40S", cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=30 * 60, volumes={"/vol": vol})
def smoke_arms(specs: list, commit: str, dirty: bool, launch: str = ""):
    t = time.time()
    small = ["--epochs", str(SMOKE["epochs"]), "--n-train", str(SMOKE["n_train"]), "--n-val", str(SMOKE["n_val"]),
             "--n-test", str(SMOKE["n_test"])]
    return [_run(sp, small, "/vol/smoke", commit, dirty, launch, t) for sp in specs]


def _git():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())
    return commit or "unknown", dirty


def _plan(specs):
    n = len(specs)
    print(f"M4 plan: {n} runs on L40S, {EPOCHS} epochs each, one container per run, in parallel")
    print(f"  runs: {', '.join(x['label'] for x in specs)}")
    print(f"  rate: ${HOURLY:.2f}/h per container (L40S ${GPU_RATE} + {CPU_CORES:g} CPU + {MEM_GIB} GiB)")
    print(f"  expected: ~{EXPECTED_H} h/run -> ~${HOURLY * EXPECTED_H:.2f}/run, ~${n * HOURLY * EXPECTED_H:.2f} total")
    print(f"  no soft cap: every run completes {EPOCHS} epochs (or early-stops at val > 0.98)")
    print(f"  hard timeout {HARD_TIMEOUT_H:g} h: kills only a hung container; a killed run is rerun, never reported")
    print(f"  WORST CASE: ${HOURLY * HARD_TIMEOUT_H:.2f}/run, ${n * HOURLY * HARD_TIMEOUT_H:.2f} total "
          f"(budget ${BUDGET:.0f})")
    print(f"  smoke first: ~${HOURLY * 7 / 60:.2f} (one L40S, ~7 min)")


def _project(o):
    """Full-run hours from a smoke run: warm epoch-2 train and validation times scaled to the full
    split sizes, plus setup (data generation etc.) and the final held-out evaluations scaled by size."""
    m, r = o.get("metrics") or [], o.get("result") or {}
    if len(m) < 2:
        return None
    train2 = m[1]["train_epoch_s"]
    val2 = m[1]["elapsed_s"] - m[0]["elapsed_s"] - train2
    setup = m[0]["elapsed_s"] - m[0]["train_epoch_s"] - val2          # data generation, model build
    tail = max(0.0, (r.get("wall_seconds") or m[1]["elapsed_s"]) - m[1]["elapsed_s"])   # held-out evals
    n_sm = SMOKE["n_train"] + SMOKE["n_val"] + SMOKE["n_test"]
    n_full = sum(FULL.values())
    train_s = train2 * FULL["n_train"] / SMOKE["n_train"]
    val_s = val2 * FULL["n_val"] / SMOKE["n_val"]
    total = EPOCHS * (train_s + val_s) + setup * n_full / n_sm + tail * FULL["n_test"] / SMOKE["n_test"] + 120
    return {"train_s": train_s, "val_s": val_s, "setup_s": setup, "total_h": total / 3600}


def _table(outs):
    print(f"\n{'run':<22} {'status':<9} {'ep':>3} {'jump':>4} {'best val (ep)':>14} {'TEST@best':>9} "
          f"{'test final':>10} {'hours':>6} {'~$':>6}")
    for o in outs:
        if isinstance(o, BaseException):
            print(f"{'?':<22} EXCEPTION {type(o).__name__}: {str(o)[:80]}")
            continue
        r = o.get("result", {})
        h = (r.get("wall_seconds") or 0) / 3600
        bv = f"{r.get('best_validation_accuracy', float('nan')):.3f} ({r.get('best_validation_epoch')})"
        print(f"{o.get('label', o['arm']):<22} {r.get('status', 'rc=' + str(o['returncode'])):<9} {r.get('epochs_run', ''):>3} "
              f"{str(r.get('jump_epoch_validation_above_0.30')):>4} {bv:>14} "
              f"{r.get('PRIMARY_test_accuracy_at_best_validation', float('nan')):>9.3f} "
              f"{r.get('test_accuracy_final', float('nan')):>10.3f} {h:>6.2f} {h * HOURLY:>6.2f}")


@app.local_entrypoint()
def main(plan: str = "variance", runs: str = "", smoke: bool = False, go: bool = False,
         allow_dirty: bool = False, data_rng: str = "legacy", mask_mode: str = "sentinel",
         router_telemetry: bool = False):
    specs = configure_specs(parse_runs(plan, runs), data_rng, mask_mode, router_telemetry)
    commit, dirty = _git()
    _plan(specs)
    print(f"  code: {commit[:10]}{' (DIRTY working tree)' if dirty else ''}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(REPO, "experiments", "m4_results")
    if not (smoke or go):
        print("\nPlan only -- nothing launched, $0 spent. Add --smoke (~$0.25) or --go (the screen).")
        return
    if go and not smoke:
        raise SystemExit("--go is retired (runs were tied to this machine's connection). Use:\n"
                         "  .venv/bin/python experiments/modal_m4_arms.py --launch")
    launch = f"{'smoke' if smoke else 'screen'}-{stamp}"
    live["launch"] = {"id": launch, "mode": "smoke", "arms": [x["label"] for x in specs],
                      "epochs": SMOKE["epochs"], "started": time.time(), "hourly": HOURLY,
                      "budget": BUDGET, "hard_timeout_h": 0.5,
                      "commit": commit[:10], "dirty": dirty}
    print(f"  live dashboard: .venv/bin/python experiments/m4_dashboard.py   (launch {launch})")
    if smoke:
        print(f"\n[smoke] one L40S, {SMOKE['epochs']} small epochs per run at the real batch shape ...")
        outs = smoke_arms.remote(specs, commit, dirty, launch)
        _table(outs)
        print(f"\n[smoke] projection for one FULL run ({EPOCHS} epochs, {FULL['n_train']:,} train / "
              f"{FULL['n_val']:,} val per epoch), from the warm epoch 2:")
        worst = 0.0
        for o in outs:
            h = _project(o)
            if h is None:
                print(f"[smoke] {o.get('label', o['arm']):<22} no projection (missing metrics)")
                continue
            worst = max(worst, h["total_h"])
            o["projection"] = h
            print(f"[smoke] {o.get('label', o['arm']):<22} {h['train_s']:6.1f} s train + {h['val_s']:5.1f} s val per epoch "
                  f"-> {h['total_h']:.2f} h  (~${h['total_h'] * HOURLY:.2f})")
        if worst:
            print(f"[smoke] plan: {len(outs)} runs in parallel -> wall time ~{worst:.2f} h, "
                  f"cost ~${sum(o['projection']['total_h'] for o in outs if 'projection' in o) * HOURLY:.2f} "
                  f"(+ early stop at val > 0.98 would make it shorter); hard timeout {HARD_TIMEOUT_H:g} h")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{launch}.json")
        with open(path, "w") as f:
            json.dump(outs, f, indent=2)
        print(f"\nsaved {path}")
        return


def _calls_path(launch=None):
    d = os.path.join(REPO, "experiments", "m4_results")
    if launch:
        return os.path.join(d, f"{launch}-calls.json")
    found = sorted((f for f in os.listdir(d) if f.endswith("-calls.json")),
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    if not found:
        raise SystemExit("no launched plan found in experiments/m4_results/")
    return os.path.join(d, found[-1])


def launch_detached(specs, plan_name, allow_dirty):
    """Deploy, then spawn one independent job per run. Nothing here waits on the runs."""
    commit, dirty = _git()
    _plan(specs)
    print(f"  code: {commit[:10]}{' (DIRTY working tree)' if dirty else ''}")
    if dirty and not allow_dirty:
        raise SystemExit("refusing --launch: working tree is dirty. Commit first (results must name exact code), "
                         "or pass --allow-dirty to override.")
    print("\n[launch] deploying the app (image + code at this commit) ...")
    subprocess.run([sys.executable, "-m", "modal", "deploy", os.path.abspath(__file__)], cwd=REPO, check=True)
    fn = modal.Function.from_name(app.name, "run_arm")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch = f"{plan_name}-{stamp}"
    live["launch"] = {"id": launch, "mode": plan_name, "arms": [x["label"] for x in specs], "epochs": EPOCHS,
                      "started": time.time(), "hourly": HOURLY, "budget": BUDGET,
                      "hard_timeout_h": HARD_TIMEOUT_H, "commit": commit[:10], "dirty": dirty}
    calls = []
    for sp in specs:
        fc = fn.spawn(sp, commit, dirty, launch)
        calls.append({**sp, "call_id": fc.object_id})
        print(f"[launch] {sp['label']:<22} spawned  {fc.object_id}")
    rec = {"launch": launch, "plan": plan_name, "commit": commit, "dirty": dirty, "calls": calls,
           "spawned_utc": stamp}
    os.makedirs(os.path.dirname(_calls_path(launch)), exist_ok=True)
    with open(_calls_path(launch), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\n[launch] {launch}: {len(calls)} independent jobs running on Modal. This machine can sleep now.")
    print("  live view : .venv/bin/python experiments/m4_dashboard.py")
    print("  results   : .venv/bin/python experiments/modal_m4_arms.py --collect")
    print(f"  stop all  : .venv/bin/python -m modal app stop {app.name}   (only if something is wrong)")


def collect(launch=None):
    """Fetch each spawned job's output (never blocks); save the screen JSON once all are finished."""
    path = _calls_path(launch)
    rec = json.load(open(path))
    outs, pending = [], []
    for c in rec["calls"]:
        try:
            outs.append(modal.FunctionCall.from_id(c["call_id"]).get(timeout=0))
        except (TimeoutError, modal.exception.TimeoutError):     # modal's is not a builtin subclass
            pending.append(c.get("label", c["arm"]))
        except Exception as e:
            outs.append({"arm": c["arm"], "label": c.get("label", c["arm"]), "returncode": None,
                         "exception": repr(e)})
    print(f"{rec['launch']}: {len(outs)} finished, {len(pending)} still running {pending or ''}")
    if outs:
        _table([o if "exception" not in o else RuntimeError(o["exception"]) for o in outs])
    if pending:
        print("\nnot all finished -- run --collect again later (the dashboard shows live progress).")
        return
    out = os.path.join(os.path.dirname(path), f"{rec['launch']}.json")
    with open(out, "w") as f:
        json.dump(outs, f, indent=2)
    print(f"\nsaved {out}\nfull run folders: .venv/bin/python -m modal volume get {VOLUME} runs experiments/m4_results/")


if __name__ == "__main__":   # plain python: plan ($0), --launch (spawn detached jobs), --collect
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="deploy + spawn the plan as independent jobs")
    ap.add_argument("--collect", nargs="?", const="", default=None, help="collect results (optional launch id)")
    ap.add_argument("--plan", default="variance", help=f"named run set: {', '.join(PLANS)}")
    ap.add_argument("--runs", default="", help='explicit runs, e.g. "dense@123,baseline:4@123"')
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--data-rng", choices=["legacy", "isolated"], default="legacy")
    ap.add_argument("--mask-mode", choices=["sentinel", "bool"], default="sentinel")
    ap.add_argument("--router-telemetry", action="store_true")
    a = ap.parse_args()
    if a.collect is not None:
        collect(a.collect or None)
    else:
        specs = configure_specs(parse_runs(a.plan, a.runs), a.data_rng, a.mask_mode, a.router_telemetry)
        if a.launch:
            launch_detached(specs, a.plan if not a.runs else "custom", a.allow_dirty)
        else:
            _plan(specs)
            print("\nPlan only -- $0. Smoke: `.venv/bin/python -m modal run experiments/modal_m4_arms.py --smoke`; "
                  "launch: `.venv/bin/python experiments/modal_m4_arms.py --launch`.")
