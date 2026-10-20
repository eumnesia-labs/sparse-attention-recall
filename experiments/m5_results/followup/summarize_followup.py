"""M5 follow-ups (docs/protocol-round2cd.md): S1=6 confirmation (AWS round 2c) and the dense check (Modal).
Reads runs/<label>/{result,manifest}.json here and M5's runs in ../runs; writes summary_followup.json and
prints the tables quoted in RESULTS_followup.md. Run: python summarize_followup.py"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
M5 = HERE.parent / "runs"
SEEDS = (810001, 810002, 810003)
NEAR_DENSE = 0.02                                 # pre-set: S1=6 near-dense at 810002 iff >= dense - 0.02


def load(root, label):
    d = root / label
    r, m = json.loads((d / "result.json").read_text()), json.loads((d / "manifest.json").read_text())
    return {"label": label, "heldout": r["PRIMARY_test_accuracy_at_best_validation"],
            "best_val": r["best_validation_accuracy"], "best_val_epoch": r["best_validation_epoch"],
            "jump_epoch": r["jump_epoch_validation_above_0.30"], "epochs": r["epochs_run"],
            "early_stopped": r["early_stopped"], "wall_h": round(r["wall_seconds"] / 3600, 2), "status": r["status"],
            "precision": m["precision"], "data_rng": m["data_rng"], "commit": str(m.get("code", {}).get("commit", ""))[:8]
            if isinstance(m.get("code"), dict) else "",
            "split_hashes": {s: (h["inputs"]["sha256"], h["targets"]["sha256"]) for s, h in m["split_hashes"].items()}}


def main():
    fu = {p.name: load(HERE / "runs", p.name) for p in sorted((HERE / "runs").iterdir()) if p.is_dir()}
    s6 = [fu[f"hier-s6-s{s}"] for s in SEEDS]
    vals = [r["heldout"] for r in s6]
    per_seed = []
    for s in SEEDS:
        h6, h2, d = fu[f"hier-s6-s{s}"], load(M5, f"hier-s2-s{s}"), load(M5, f"dense-s{s}")
        per_seed.append({"seed": s, "s1_6": h6["heldout"], "s1_2": h2["heldout"], "dense": d["heldout"],
                         "s1_6_minus_s1_2": round(h6["heldout"] - h2["heldout"], 4),
                         "dense_transitioned": d["jump_epoch"] is not None,
                         "dense_minus_s1_6": round(d["heldout"] - h6["heldout"], 4) if d["jump_epoch"] is not None else None,
                         "identical_data_with_m5_dense_and_s1_2": h6["split_hashes"] == d["split_hashes"] == h2["split_hashes"]})
    p2 = next(p for p in per_seed if p["seed"] == 810002)
    m5_dense123 = load(M5, "dense-s123")
    fp32, tf32 = fu["dense-fp32-isolated-s123"], fu["dense-tf32-legacy-s123"]
    summary = {
        "s1_6": {"values": vals, "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4),
                 "min": round(min(vals), 4), "sample_sd": round(statistics.stdev(vals), 4),
                 "transitions": f"{sum(r['jump_epoch'] is not None for r in s6)}/3"},
        "per_seed": per_seed,
        "near_dense_at_810002": {"rule": f"S1=6 >= dense - {NEAR_DENSE}", "dense": p2["dense"], "s1_6": p2["s1_6"],
                                 "near_dense": p2["s1_6"] >= p2["dense"] - NEAR_DENSE},
        "dense_check_seed123": {
            "m4_fp32_legacy_context": {"heldout": 0.992, "jump_epoch": 28, "note": "M4 follow-up, Modal, older code"},
            "m5_tf32_isolated": {k: m5_dense123[k] for k in ("heldout", "jump_epoch", "epochs")},
            "tf32_legacy": {k: tf32[k] for k in ("heldout", "jump_epoch", "epochs", "early_stopped")},
            "fp32_isolated": {k: fp32[k] for k in ("heldout", "jump_epoch", "epochs", "early_stopped")},
            "fp32_isolated_same_data_as_m5_dense123": fp32["split_hashes"] == m5_dense123["split_hashes"],
            "reading": "both transition -> neither change alone explains M5's dense non-transition (protocol table row 3)"},
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "split_hashes"} for k, v in fu.items()},
    }
    (HERE / "summary_followup.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("| run | held-out | best val (epoch) | jump > 0.30 | epochs | hours |")
    print("|---|--:|--:|--:|--:|--:|")
    for r in fu.values():
        print(f"| {r['label']} | {r['heldout']:.3f} | {r['best_val']:.3f} ({r['best_val_epoch']}) | "
              f"{'ep ' + str(r['jump_epoch']) if r['jump_epoch'] is not None else 'never'} | {r['epochs']}"
              f"{' (early stop)' if r['early_stopped'] else ''} | {r['wall_h']} |")
    print("\nS1=6:", summary["s1_6"])
    for p in per_seed:
        print("seed", p)
    print("near-dense at 810002:", summary["near_dense_at_810002"])
    print("dense check:", json.dumps(summary["dense_check_seed123"]))


if __name__ == "__main__":
    main()
