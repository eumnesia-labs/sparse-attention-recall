"""Selection-vs-reading diagnostic on the 12 M5-profile selector checkpoints, on Modal L40S (inference only).

Owner-approved 2026-10-20 (paper Part 2). Each container replays one run's full held-out split through
experiments/diagnose_routing.py with the run's own numerics (TF32) and isolated data, and must pass that
script's validity gate (reproduced held-out accuracy, loss and split fingerprint) or its numbers are untrusted.
No training. ~$2.14/h per container; expected total a few dollars.

  .venv/bin/modal run experiments/modal_diagnose_m5.py --mode smoke   # one run, 8 sequences (~cents)
  .venv/bin/modal run experiments/modal_diagnose_m5.py --mode full    # all 12 runs in parallel

Checkpoints (best.pt) are read from CKPT_DIR on the Mac, with each run's manifest.json and result.json.
Output: experiments/m5_results/selection_diagnostic/routing-m5-<mode>.json
"""
import json
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CKPT_DIR = os.environ.get("M5_CKPT_DIR", os.path.expanduser("~/SSA/checkpoints/m5"))
TORCH, TORCHVISION, ZOO_COMMIT = "2.14.0", "0.29.0", "1ad20d1"
LABELS = ["hier-s2-s123", "hier-s4-s123", "hier-s6-s123", "hier-s12-s123", "hier-s2-s231", "hier-s2-s312",
          "hier-s2-s810001", "hier-s2-s810002", "hier-s2-s810003", "hier-s6-s810001", "hier-s6-s810002",
          "hier-s6-s810003"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(f"torch=={TORCH}", f"torchvision=={TORCHVISION}", "numpy", "pandas", "einops",
                 "pydantic", "tqdm", "rich", "wandb")
    .run_commands(f"git clone https://github.com/HazyResearch/zoology.git /zoology && "
                  f"cd /zoology && git checkout -q {ZOO_COMMIT}")
    .add_local_file(os.path.join(HERE, "patch_zoology_accum.py"), "/opt/patch_zoology_accum.py", copy=True)
    .run_commands("python /opt/patch_zoology_accum.py /zoology")
    .env({"ZOOLOGY_DIR": "/zoology", "WANDB_MODE": "disabled"})
    .add_local_dir(os.path.join(REPO, "src"), "/repo/src")
    .add_local_file(os.path.join(HERE, "diagnose_routing.py"), "/repo/experiments/diagnose_routing.py")
    .add_local_file(os.path.join(HERE, "m4_data.py"), "/repo/experiments/m4_data.py")
    .add_local_dir(CKPT_DIR, "/ckpt")
)
app = modal.App("ssa-recall-m5-diagnostic", image=image)


@app.function(gpu="L40S", cpu=2.0, memory=12 * 1024, timeout=60 * 60)
def diagnose(label: str, n: int):
    out = f"/tmp/{label}.json"
    p = subprocess.run(["python", "/repo/experiments/diagnose_routing.py", f"/ckpt/{label}", "--n", str(n),
                        "--batch", "16", "--out", out], capture_output=True, text=True)
    report = json.load(open(out)) if os.path.exists(out) else None
    return {"label": label, "returncode": p.returncode, "stderr_tail": p.stderr[-3000:], "report": report}


@app.local_entrypoint()
def main(mode: str = "smoke"):
    labels, n = (["hier-s6-s123"], 8) if mode == "smoke" else (LABELS, 2000)
    results = list(diagnose.starmap([(lb, n) for lb in labels]))
    dest = os.path.join(REPO, "experiments/m5_results/selection_diagnostic")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"routing-m5-{mode}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        rep = (r["report"] or [{}])[0]
        print(r["label"], "rc", r["returncode"], "acc", rep.get("accuracy_reproduced"), "recorded",
              rep.get("recorded_primary"), "gate", rep.get("validity_gate"), "s", rep.get("seconds"))
        if r["returncode"] != 0:
            print("   stderr:", r["stderr_tail"][-800:])
    print("wrote", path)
