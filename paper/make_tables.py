"""Regenerate paper/sections/appendix_runs.tex (every M5-profile run) from the committed run records.
Run from the repo root: python3 paper/make_tables.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M5 = ROOT / "experiments/m5_results"
ROWS = [  # (folder, label, model, seed, numerics, data, round)
    ("runs", "hier-s2-s123", "$S_1{=}2$", 123, "TF32", "isolated", "M5"),
    ("runs", "dense-s123", "dense", 123, "TF32", "isolated", "M5"),
    ("runs", "hier-s2-s231", "$S_1{=}2$", 231, "TF32", "isolated", "M5"),
    ("runs", "hier-s2-s312", "$S_1{=}2$", 312, "TF32", "isolated", "M5"),
    ("runs", "hier-s12-s123", "$S_1{=}12$", 123, "TF32", "isolated", "M5"),
    ("runs", "hier-s4-s123", "$S_1{=}4$", 123, "TF32", "isolated", "M5"),
    ("runs", "hier-s6-s123", "$S_1{=}6$", 123, "TF32", "isolated", "M5"),
    ("runs", "dense-s810001", "dense", 810001, "TF32", "isolated", "M5"),
    ("runs", "hier-s2-s810001", "$S_1{=}2$", 810001, "TF32", "isolated", "M5"),
    ("runs", "dense-s810002", "dense", 810002, "TF32", "isolated", "M5"),
    ("runs", "hier-s2-s810002", "$S_1{=}2$", 810002, "TF32", "isolated", "M5"),
    ("runs", "dense-s810003", "dense", 810003, "TF32", "isolated", "M5"),
    ("runs", "hier-s2-s810003", "$S_1{=}2$", 810003, "TF32", "isolated", "M5"),
    ("followup/runs", "hier-s6-s810001", "$S_1{=}6$", 810001, "TF32", "isolated", "2c"),
    ("followup/runs", "hier-s6-s810002", "$S_1{=}6$", 810002, "TF32", "isolated", "2c"),
    ("followup/runs", "hier-s6-s810003", "$S_1{=}6$", 810003, "TF32", "isolated", "2c"),
    ("followup/runs", "dense-tf32-legacy-s123", "dense", 123, "TF32", "legacy", "check"),
    ("followup/runs", "dense-fp32-isolated-s123", "dense", 123, "float32", "isolated", "check"),
    ("round2ef/runs", "dense-fp32-isolated-s810001", "dense", 810001, "float32", "isolated", "2e"),
    ("round2ef/runs", "dense-fp32-isolated-s810003", "dense", 810003, "float32", "isolated", "2f"),
]


def row(folder, label, model, seed, num, data, rnd):
    d = M5 / folder / label
    curve = [json.loads(l)["valid_accuracy"] for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    best = max(range(len(curve)), key=lambda i: (curve[i], -i))
    jump = next((i for i, v in enumerate(curve) if v > 0.30), None)
    res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
    held = f"{res['PRIMARY_test_accuracy_at_best_validation']:.3f}" if res else "--$^\\ddagger$"
    return (f"{model} & {seed} & {num} & {data} & {rnd} & {held} & {curve[best]:.3f} ({best}) & "
            f"{jump if jump is not None else 'never'} & {len(curve)} \\\\")


lines = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\begin{tabular}{lrllrrrrr}", r"\toprule",
         r"Model & seed & numerics & data & round & held-out & best val. (epoch) & transition & epochs \\",
         r"\midrule"] + [row(*r) for r in ROWS] + [
         r"\bottomrule", r"\end{tabular}",
         r"\caption{\textbf{Every run on the M5 profile and its diagnostics, including all non-transitions.} Epochs are"
         r" zero-based; early stopping at validation recall above 0.98 ends some runs before 60 epochs."
         r" $^\ddagger$Stopped by us after 55 of 60 epochs; no held-out score.}",
         r"\label{tab:allruns}", r"\end{table}"]
(ROOT / "paper/sections/appendix_runs.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
