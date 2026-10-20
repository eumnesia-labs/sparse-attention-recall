"""M5 (AWS round 2a) summary: reads runs/<label>/{result,manifest}.json and writes summary.json plus the
tables quoted in RESULTS.md. Protocol: docs/protocol-round2.md (machine A). Run: python summarize.py"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_SEEDS = (123, 231, 312)
CONFIRM_SEEDS = (810001, 810002, 810003)
PAIRS_PER_QUERY = {2: 34, 4: 56, 6: 78, 12: 144}       # scored selector pairs/query at L2048 (protocol)


def load(label):
    d = HERE / "runs" / label
    return json.loads((d / "result.json").read_text()), json.loads((d / "manifest.json").read_text())


def row(label):
    r, m = load(label)
    return {"label": label, "arm": r["arm"], "s1": r["s1"], "seed": r["seed"], "status": r["status"],
            "heldout": r["PRIMARY_test_accuracy_at_best_validation"], "best_val": r["best_validation_accuracy"],
            "best_val_epoch": r["best_validation_epoch"], "jump_epoch": r["jump_epoch_validation_above_0.30"],
            "epochs": r["epochs_run"], "early_stopped": r["early_stopped"], "wall_h": round(r["wall_seconds"] / 3600, 2),
            "precision": m["precision"], "data_rng": m["data_rng"], "protocol_profile": r["protocol_profile"],
            "split_hashes": {s: (h["inputs"]["sha256"], h["targets"]["sha256"]) for s, h in m["split_hashes"].items()}}


def spread(vals):
    return {"values": vals, "range_R": round(max(vals) - min(vals), 4), "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4), "sample_sd": round(statistics.stdev(vals), 4)}


def main():
    rows = {p.name: row(p.name) for p in sorted((HERE / "runs").iterdir()) if p.is_dir()}
    s1_123 = {s1: rows[f"hier-s{s1}-s123"] for s1 in PAIRS_PER_QUERY}
    pilot_s2 = [rows[f"hier-s2-s{s}"] for s in PILOT_SEEDS]
    all_s2 = pilot_s2 + [rows[f"hier-s2-s{s}"] for s in CONFIRM_SEEDS]
    dense = [rows[f"dense-s{s}"] for s in (123,) + CONFIRM_SEEDS]
    pairs = []
    for s in (123,) + CONFIRM_SEEDS:
        d, h = rows[f"dense-s{s}"], rows[f"hier-s2-s{s}"]
        pairs.append({"seed": s, "dense": d["heldout"], "hier_s2": h["heldout"],
                      "dense_minus_hier": round(d["heldout"] - h["heldout"], 4),
                      "dense_transitioned": d["jump_epoch"] is not None, "hier_transitioned": h["jump_epoch"] is not None,
                      "identical_data": d["split_hashes"] == h["split_hashes"],
                      "stage": "pilot" if s == 123 else "confirmation"})
    summary = {
        "all_complete": all(r["status"] == "complete" for r in rows.values()),
        "profiles": sorted({(r["precision"], r["data_rng"], r["protocol_profile"]) for r in rows.values()}),
        "budget_curve_seed123": [{"s1": k, "pairs_per_query": PAIRS_PER_QUERY[k], "heldout": v["heldout"],
                                  "best_val": v["best_val"], "jump_epoch": v["jump_epoch"]} for k, v in s1_123.items()],
        "s1_2_pilot_spread": {**spread([r["heldout"] for r in pilot_s2]),
                              "non_transitions": sum(r["jump_epoch"] is None for r in pilot_s2)},
        "s1_2_all_six": {**spread([r["heldout"] for r in all_s2]), "non_transitions": sum(r["jump_epoch"] is None for r in all_s2)},
        "dense_transitions": f"{sum(r['jump_epoch'] is not None for r in dense)}/{len(dense)}",
        "pairs": pairs,
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "split_hashes"} for k, v in rows.items()},
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("| run | seed | held-out | best val (epoch) | jump > 0.30 | epochs | hours |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for r in rows.values():
        name = "dense" if r["arm"] == "dense" else f"hier S1={r['s1']}"
        print(f"| {name} | {r['seed']} | {r['heldout']:.3f} | {r['best_val']:.3f} ({r['best_val_epoch']}) | "
              f"{'ep ' + str(r['jump_epoch']) if r['jump_epoch'] is not None else 'never'} | {r['epochs']}"
              f"{' (early stop)' if r['early_stopped'] else ''} | {r['wall_h']} |")
    print("\nbudget curve, seed 123:", [(b["s1"], b["pairs_per_query"], round(b["heldout"], 3)) for b in summary["budget_curve_seed123"]])
    print("S1=2 pilot spread:", summary["s1_2_pilot_spread"])
    print("S1=2 all six:", summary["s1_2_all_six"])
    print("dense transitions:", summary["dense_transitions"])
    for p in pairs:
        print("pair", p)
    print("all complete:", summary["all_complete"], "| profiles:", summary["profiles"])


if __name__ == "__main__":
    main()
