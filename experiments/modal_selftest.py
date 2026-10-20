"""hier_nsa self-test in a Modal CPU container (local Windows torch is DLL-broken).

Usage (repo root):  python -m modal run experiments/modal_selftest.py

Runs src/hier_nsa.py's _selftest — shape/causality/grad plus the polarized-score nan
regression — and an op-level demo of the s12-k32 ep13 crash class: sigmoid().log()
mints 0*inf = nan in backward once a score crosses the fp32 underflow cliff (~-88);
F.logsigmoid stays finite. CPU only, costs cents.
"""
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch")
         .add_local_dir("src", "/repo/src"))
app = modal.App("hier-selftest", image=image)


@app.function()
def run():
    import sys
    sys.path.insert(0, "/repo/src")
    import torch
    import torch.nn.functional as F
    import hier_nsa

    hier_nsa._selftest()

    # mechanism demo: the exact op class that killed s12-k32, old vs fixed
    s = torch.tensor([-200.0, -5.0, 3.0], requires_grad=True)   # -200 = past the cliff
    torch.sigmoid(s).log().nan_to_num(-1e9, -1e9, -1e9).sum().backward()
    old_grad = s.grad.clone()
    s.grad = None
    F.logsigmoid(s).clamp_min(-1e9).sum().backward()
    return {"selftest": "OK",
            "old_grad_finite": bool(torch.isfinite(old_grad).all()),
            "new_grad_finite": bool(torch.isfinite(s.grad).all()),
            "old_grad": old_grad.tolist(), "new_grad": s.grad.tolist()}


@app.local_entrypoint()
def main():
    print(run.remote())
