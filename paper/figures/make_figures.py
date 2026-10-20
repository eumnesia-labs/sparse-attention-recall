"""Regenerate the paper's figures from the committed per-run records (no GPU, no network).

Run from the repo root:  python3 paper/figures/make_figures.py
Inputs: experiments/m5_results/{runs,followup/runs,round2ef/runs}/<label>/{result.json,metrics.jsonl}
Outputs: paper/figures/fig1_budget.pdf, fig2_training.pdf (+ .png previews)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
M5 = ROOT / "experiments/m5_results"
OUT = Path(__file__).resolve().parent
RUN_DIRS = [M5 / "runs", M5 / "followup/runs", M5 / "round2ef/runs"]
PAIRS = {2: 34, 4: 56, 6: 78, 12: 144}
C = {"dense": "#222222", 2: "#d95f02", 4: "#7570b3", 6: "#1b9e77", 12: "#66a61e"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif"})


def run(label):
    for d in RUN_DIRS:
        p = d / label
        if p.exists():
            res = json.loads((p / "result.json").read_text()) if (p / "result.json").exists() else None
            curve = [json.loads(l)["valid_accuracy"] for l in (p / "metrics.jsonl").read_text().splitlines() if l.strip()]
            return res, curve
    raise FileNotFoundError(label)


def heldout(label):
    return run(label)[0]["PRIMARY_test_accuracy_at_best_validation"]


def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.75, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) budget curve, seed 123, plus S1=2 pilot seeds
    xs = [PAIRS[s] for s in (2, 4, 6, 12)]
    ys = [heldout(f"hier-s{s}-s123") for s in (2, 4, 6, 12)]
    a.plot(xs, ys, "-o", color="#555555", ms=4, lw=1, zorder=2, label="selector, seed 123")
    for s, x, y in zip((2, 4, 6, 12), xs, ys):
        a.annotate(f"S1={s}" + (" (flat)" if s == 12 else ""), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
    for sd in (231, 312):
        a.plot(PAIRS[2], heldout(f"hier-s2-s{sd}"), "o", mfc="none", color=C[2], ms=4,
               label="S1=2, seeds 231/312" if sd == 231 else None)
    a.axhline(1.0, color=C["dense"], ls="--", lw=0.8)
    a.text(150, 1.005, "dense, isolated-data runs that learned (1.000)", ha="right", va="bottom", fontsize=6.5)
    a.set_xlabel("selector scores per query (count, not runtime)")
    a.set_ylabel("held-out recall")
    a.set_ylim(0.5, 1.05)
    a.legend(frameon=False, fontsize=6.5, loc="lower right")
    a.set_title("(a) recall vs selector budget (L=2048, K=4)", fontsize=8)
    # (b) reserved seeds: dense / S1=2 / S1=6, identical data within a seed
    seeds = (810001, 810002, 810003)
    arms = [("dense", "dense-s{}"), ("S1=2", "hier-s2-s{}"), ("S1=6", "hier-s6-s{}")]
    cols = [C["dense"], C[2], C[6]]
    w = 0.26
    for i, ((name, pat), col) in enumerate(zip(arms, cols)):
        vals = [heldout(pat.format(sd)) for sd in seeds]
        xs_b = [j + (i - 1) * w for j in range(3)]
        bars = b.bar(xs_b, vals, w, color=col, label=name)
        for x, v in zip(xs_b, vals):
            if v < 0.30:  # never crossed the 0.30 transition threshold within 60 epochs
                b.text(x, 0.03, "no\ntransition", ha="center", va="bottom", fontsize=5.5, rotation=90)
            else:
                b.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    b.set_xticks(range(3), [str(s) for s in seeds])
    b.set_xlabel("reserved seed (identical data within a seed)")
    b.set_ylim(0, 1.3)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.set_title("(b) confirmation on untouched seeds", fontsize=8)
    b.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_budget.{ext}", dpi=200)


def fig2():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    dense = ["dense-s123", "dense-s810001", "dense-s810002", "dense-s810003",
             "dense-fp32-isolated-s123", "dense-fp32-isolated-s810001", "dense-fp32-isolated-s810003"]
    s16 = ["hier-s6-s123", "hier-s6-s810001", "hier-s6-s810002", "hier-s6-s810003"]
    for lab in dense:
        ax.plot(run(lab)[1], color=C["dense"], lw=0.9, alpha=0.8, ls="--" if "fp32" in lab else "-")
    for lab in s16:
        ax.plot(run(lab)[1], color=C[6], lw=0.9, alpha=0.9)
    ax.axhline(0.30, color="#999999", lw=0.6, ls=":")
    ax.plot([], [], color=C["dense"], label="dense, TF32 (4 runs)")
    ax.plot([], [], color=C["dense"], ls="--", label="dense, float32 (3 runs)")
    ax.plot([], [], color=C[6], label="selector S1=6 (4 runs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation recall")
    ax.set_xlim(0, 60)
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_training.{ext}", dpi=200)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote", sorted(p.name for p in OUT.glob("fig*")))
