"""Rounds 2e/2f (docs/protocol-round2ef.md): dense float32 + isolated data on confirmation seeds 810001 (AWS) and
810003 (Modal). Reads runs/<label>/ here, M5's runs in ../runs and the follow-ups in ../followup/runs; writes
summary_round2ef.json and prints the tables quoted in RESULTS_round2ef.md. Run: python summarize_round2ef.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M5, FU = HERE.parent / "runs", HERE.parent / "followup" / "runs"
SEEDS = (810001, 810003)
JUMP = 0.30


def hashes(d):
    m = json.loads((d / "manifest.json").read_text())
    return {s: (h["inputs"]["sha256"], h["targets"]["sha256"]) for s, h in m["split_hashes"].items()}


def run(d):
    """A run folder: result.json when the run completed; otherwise the per-epoch metrics up to where it stopped."""
    m = json.loads((d / "manifest.json").read_text())
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    val = [(r["epoch"], r["valid_accuracy"]) for r in rows]
    best = max(val, key=lambda x: x[1])
    jump = next((e for e, v in val if v > JUMP), None)
    out = {"precision": m["precision"], "data_rng": m["data_rng"], "epochs_done": len(rows),
           "best_val": round(best[1], 6), "best_val_epoch": best[0], "jump_epoch": jump,
           "last_val": round(val[-1][1], 6), "last_valid_loss": round(rows[-1]["valid_loss"], 4)}
    rp = d / "result.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        out.update(status=r["status"], heldout=r["PRIMARY_test_accuracy_at_best_validation"],
                   wall_h=round(r["wall_seconds"] / 3600, 2), early_stopped=r["early_stopped"])
    else:
        out.update(status="stopped by owner (not complete)", heldout=None)
    return out


def heldout(d):
    return json.loads((d / "result.json").read_text())["PRIMARY_test_accuracy_at_best_validation"]


def transitioned(d):
    return json.loads((d / "result.json").read_text())["jump_epoch_validation_above_0.30"] is not None


def main():
    new = {s: run(HERE / "runs" / f"dense-fp32-isolated-s{s}") for s in SEEDS}
    pairs = []
    for s in (810001, 810002, 810003):
        d5, h2, h6 = M5 / f"dense-s{s}", M5 / f"hier-s2-s{s}", FU / f"hier-s6-s{s}"
        row = {"seed": s, "dense_tf32_m5": heldout(d5), "dense_tf32_m5_transitioned": transitioned(d5),
               "s1_2": heldout(h2), "s1_6": heldout(h6)}
        if s in new:
            nd = HERE / "runs" / f"dense-fp32-isolated-s{s}"
            row.update(dense_fp32=new[s]["heldout"], dense_fp32_best_val=new[s]["best_val"],
                       dense_fp32_transitioned=new[s]["jump_epoch"] is not None,
                       dense_fp32_status=new[s]["status"],
                       identical_data_with_m5_dense_s1_2_s1_6=hashes(nd) == hashes(d5) == hashes(h2) == hashes(h6))
        row["valid_dense_pair"] = row["dense_tf32_m5_transitioned"] or row.get("dense_fp32_transitioned", False)
        pairs.append(row)
    dense_isolated = {"tf32 (M5: 123, 810001, 810002, 810003)": [transitioned(M5 / f"dense-s{s}") for s in (123, 810001, 810002, 810003)],
                      "fp32 (123 dense check; 810001, 810003 here)": [transitioned(FU / "dense-fp32-isolated-s123")]
                      + [new[s]["jump_epoch"] is not None for s in SEEDS]}
    hier_isolated = {"S1=6 (123, 810001-3)": [transitioned(M5 / "hier-s6-s123")] + [transitioned(FU / f"hier-s6-s{s}") for s in (810001, 810002, 810003)],
                     "S1=2 (123, 231, 312, 810001-3)": [transitioned(M5 / f"hier-s2-s{s}") for s in (123, 231, 312, 810001, 810002, 810003)]}
    tally = {k: f"{sum(v)}/{len(v)}" for k, v in {**dense_isolated, **hier_isolated}.items()}
    summary = {"runs": {f"dense-fp32-isolated-s{s}": new[s] for s in SEEDS}, "per_seed": pairs,
               "transitions_isolated_data": tally,
               "note": "810001 was stopped by the owner after 55 of 60 epochs (no held-out evaluation); 810003 completed 60"}
    (HERE / "summary_round2ef.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("| run | status | epochs | best val (epoch) | jump > 0.30 | held-out | last val loss |")
    print("|---|---|--:|--:|--:|--:|--:|")
    for s in SEEDS:
        r = new[s]
        held = "—" if r["heldout"] is None else f"{r['heldout']:.4f}"
        jump = "never" if r["jump_epoch"] is None else r["jump_epoch"]
        print(f"| dense fp32 @{s} | {r['status']} | {r['epochs_done']} | {r['best_val']:.4f} ({r['best_val_epoch']}) | "
              f"{jump} | {held} | {r['last_valid_loss']} |")
    print()
    for p in pairs:
        print("seed", p)
    print("\ntransitions on isolated data:", tally)


if __name__ == "__main__":
    main()
