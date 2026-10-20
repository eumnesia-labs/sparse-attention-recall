"""Routing support-hit diagnostic for trained M4/M5 checkpoints (CPU or GPU; upgrade plan Step A, Phase 0 Part A).

2026-10-20: also supports isolated-data (M5) runs (m4_data.SeededMQARConfig) and replays on CUDA when
available, with TF32 set from the run manifest so the replay matches the recorded numerics.

For each held-out MQAR query, find the block holding its key (keys sit at even positions of the
first 2*num_kv tokens and are unique per example, zoology multiquery_ar.py), then record per layer
and head whether the hierarchical selector picked that block's parent super-block and the block
itself, whether the key was inside the sliding window anyway, and whether the answer was correct.

Validity gate (enforced; full split only): the reproduced held-out accuracy must match the run's
recorded PRIMARY_test_accuracy_at_best_validation to within GATE_TOL queries (near-tied routing scores
may break differently on CPU vs GPU). A failed gate makes the run's diagnostics untrusted and the
script exit non-zero. Both metrics are per-query means (every example has exactly num_kv queries).

Reading the numbers (independent review, 2026-10-01): a "miss" of the key's block is not proof the
answer was unreachable — layer-0 mixing spreads key/value information into later blocks — so the
nearest selected block offset (>= the key block) is reported; query position confounds both hit
rate and accuracy, so everything is also reported by query-position stratum; at S1=2, parent hits
are forced while <= S1 super-blocks are complete (q < 528 at L2048, g=11), and at S1=12 always.

  ZOOLOGY_DIR=<patched zoology> python experiments/diagnose_routing.py RUN_DIR [RUN_DIR ...] \\
      [--n 2000] [--batch 8] [--out routing.json]
RUN_DIR holds manifest.json, result.json and best.pt (downloaded from volume ssa-recall-m4-runs).
"""
import argparse
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DENSE = {"dropout": 0.1, "num_heads": 4}
GATE_TOL = 2                       # queries out of n*num_kv
LOSS_TOL = 2e-3                    # |reproduced - recorded| held-out loss (CPU vs GPU float differences)
STRATA = [(0, 256), (256, 352), (352, 528), (528, 1024), (1024, 1536), (1536, 2048)]  # query strata (L2048);
# MQAR queries are always >= 256, so its first stratum is empty; spread-MQAR queries can be anywhere


def _args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=2000, help="held-out sequences to use (first n)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    a = _args()
    zdir = os.environ.get("ZOOLOGY_DIR") or os.path.join(REPO, "zoology")
    sys.path[:0] = [zdir, os.path.join(REPO, "src"), HERE]
    import numpy as np
    import torch.nn.functional as F
    import torch
    import zoology.model as zm
    from zoology.config import ModelConfig, ModuleConfig
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.data.utils import DataSegment
    from zoology.model import LanguageModel
    from hier_nsa import HierSparseAttention
    if not torch.cuda.is_available():    # zoology 1ad20d1 hard-codes device='cuda' defaults
        for _, c in inspect.getmembers(zm, inspect.isclass):
            d = getattr(c.__init__, "__defaults__", None)
            if d and "cuda" in d:
                c.__init__.__defaults__ = tuple("cpu" if x == "cuda" else x for x in d)
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report = []
    for run in a.runs:
        t0 = time.time()
        man = json.load(open(os.path.join(run, "manifest.json")))
        res = json.load(open(os.path.join(run, "result.json")))
        P = man["protocol"]
        dense = man["arm"] == "dense" or man.get("mixer") == "zoology.mixers.attention.MHA"
        s1 = None if dense else (man.get("s1") or P["s1"])
        L, nkv, cb = P["seq_len"], P["num_kv_pairs"], P["compress_block_size"]
        g = P.get("superblock_group")
        mixer = (ModuleConfig(name="zoology.mixers.attention.MHA", kwargs=dict(man.get("arm_kwargs") or DENSE))
                 if dense else
                 ModuleConfig(name="hier_nsa.HierSparseAttention", kwargs={
                     "num_heads": P["num_heads"], "sliding_window_size": P["sliding_window_size"],
                     "compress_block_size": cb, "selection_block_size": cb, "num_selected_blocks": P["k_sel"],
                     "superblock_group": g, "num_selected_superblocks": s1, "causal": True,
                     **(man.get("arm_kwargs") or {})}))
        model = LanguageModel(ModelConfig(vocab_size=P["vocab_size"], max_position_embeddings=L,
                                          d_model=P["d_model"], n_layers=P["n_layers"], sequence_mixer=mixer))
        model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu"), strict=True)
        tf32 = man.get("precision") == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        model.to(dev).eval()
        mods = [m for m in model.modules() if isinstance(m, HierSparseAttention)]
        for m in mods:
            assert getattr(m, "mask_mode", "sentinel") == "sentinel", "runs used the sentinel mask path"
            m.keep_routing = True
        hseed = man["data_seeds"]["heldout_test"]
        spread = man.get("task") == "spread"
        if spread:                                  # spread-MQAR: numpy-only RNG (src/spread_mqar.py)
            from spread_mqar import SpreadMQARConfig, key_write_positions
            cfg = SpreadMQARConfig(num_examples=man["split_sizes"]["heldout_test"], vocab_size=P["vocab_size"],
                                   input_seq_len=L, num_kv_pairs=nkv)
        elif man.get("data_rng", "legacy") == "isolated":   # M5: the segment seeds its own noise RNG
            from m4_data import SeededMQARConfig
            cfg = SeededMQARConfig(num_examples=man["split_sizes"]["heldout_test"], vocab_size=P["vocab_size"],
                                   input_seq_len=L, num_kv_pairs=nkv)
        else:                                       # legacy held-out regeneration, as m4_routing_arms.py did
            if man.get("data_rng", "legacy") != "legacy":
                sys.exit("unsupported data_rng")
            torch.manual_seed(hseed)
            cfg = MQARConfig(num_examples=man["split_sizes"]["heldout_test"], vocab_size=P["vocab_size"],
                             input_seq_len=L, num_kv_pairs=nkv)
        seg = DataSegment.from_config(cfg, cache_dir=None, seed=hseed)
        rec_fp = (man.get("split_hashes") or {}).get("heldout_test")
        if rec_fp:                                  # exact identity of the regenerated held-out split
            from m4_data import segment_fingerprints
            fp_match = segment_fingerprints(seg) == rec_fp
        else:
            fp_match = None                         # legacy manifests predate split fingerprints
        X, Y = seg.inputs[: a.n], seg.labels[: a.n]
        nl, nh, w = len(mods), P["num_heads"], P["sliding_window_size"]
        rec = {"correct": [], "qt": [], "win": [], "reach": [], "recency": [], "randexp": []}
        batch_losses = []
        lay = [{"sup_any": [], "hit_any": [], "dmin": [], "head_hit": []} for _ in range(nl)]
        for i in range(0, len(X), a.batch):
            x, y = X[i:i + a.batch], Y[i:i + a.batch]
            logits = model(x.to(dev)).float().cpu()
            if spread:                                               # unique earlier occurrence of the key
                qb, qt, kpos = key_write_positions(x, y)
            else:
                qb, qt = (y != -100).nonzero(as_tuple=True)          # query positions
                keys = x[:, : 2 * nkv][:, 0::2]                      # (B, nkv) keys at even positions
                match = keys[qb] == x[qb, qt].unsqueeze(-1)          # query token == key token
                assert bool((match.sum(-1) == 1).all()), "every query must match exactly one key"
                kpos = match.float().argmax(-1) * 2                  # key position; value at kpos+1
            correct = logits[qb, qt].argmax(-1) == y[qb, qt]
            batch_losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                      ignore_index=-100)))   # as the harness's evaluate()
            # the support is the VALUE token (an induction head attends to the token after the key's
            # earlier occurrence). MQAR keys are even-aligned, so key and value share a block and this
            # equals kpos // cb exactly; in spread-MQAR a key at a block's last slot has its value next.
            kblk = (kpos + 1) // cb
            rec["correct"].append(correct.numpy()); rec["qt"].append(qt.numpy())
            in_win = (qt - kpos) <= w                                 # value inside the sliding window
            rec["win"].append(in_win.numpy())
            if not dense:
                sup_k = kblk // g                                     # parent super-block of the support
                complete = (sup_k + 1) * g * cb <= qt                 # selectable as a live parent at qt
                ncomp = qt // (g * cb)                                # complete super-blocks at qt
                # hier_nsa: with fewer than S1 complete supers, top-S1 also returns masked placeholders, and
                # a placeholder's fully-past child blocks stay live candidates. We assume the current
                # (incomplete) super wins the placeholder tie (observed: parent hit 1.0 on these queries).
                placeholder = (sup_k == ncomp) & (ncomp < s1) & ((kblk + 1) * cb <= qt)
                rec["reach"].append((complete | placeholder | in_win).numpy())
                rec["recency"].append(((complete & (sup_k >= ncomp - s1)) | placeholder).numpy())
                rec["randexp"].append(torch.where(placeholder, torch.ones_like(qt, dtype=torch.float),
                                                  torch.where(complete & (ncomp > 0),
                                                              (s1 / ncomp.clamp_min(1)).clamp(max=1.0),
                                                              torch.zeros_like(qt, dtype=torch.float))).numpy())
            for li, m in enumerate(mods):
                fin = m.last_fin_blk.cpu()[qb, :, qt]                # (Q, h, ke)
                sup = m.last_sup_idx.cpu()[qb, :, qt]                # (Q, h, s1e)
                fh = (fin == kblk[:, None, None]).any(-1)            # (Q, h)
                off = fin - kblk[:, None, None]
                off = torch.where(off >= 0, off, torch.full_like(off, 10 ** 6))
                lay[li]["sup_any"].append((sup == (kblk // g)[:, None, None]).any(-1).any(-1).numpy())
                lay[li]["hit_any"].append(fh.any(-1).numpy())
                lay[li]["dmin"].append(off.amin(dim=(1, 2)).numpy())
                lay[li]["head_hit"].append(fh.numpy())
        C = np.concatenate(rec["correct"]); T = np.concatenate(rec["qt"]); Wn = np.concatenate(rec["win"])
        Q = int(C.size)
        rp = res.get("PRIMARY_test_accuracy_at_best_validation")
        rl = res.get("test_loss_at_best_validation")
        full = len(X) == man["split_sizes"]["heldout_test"]
        loss = float(np.mean(batch_losses))
        acc_ok = None if (rp is None or not full) else abs(int(C.sum()) - round(rp * Q)) <= GATE_TOL
        loss_ok = None if (rl is None or not full) else abs(loss - rl) <= LOSS_TOL
        # gate: accuracy AND loss AND split fingerprint must all agree where recorded
        gate = None if acc_ok is None else bool(acc_ok and loss_ok is not False and fp_match is not False)
        Rch = np.concatenate(rec["reach"]) if rec["reach"] else None
        Rec = np.concatenate(rec["recency"]) if rec["recency"] else None
        Rnd = np.concatenate(rec["randexp"]) if rec["randexp"] else None

        def rate(mask, of=None):
            of = C if of is None else of
            return float(of[mask].mean()) if mask.any() else None

        out = {"run": os.path.basename(os.path.normpath(run)), "arm": man["arm"], "s1": s1, "seed": man["seed"],
               "n_sequences": len(X), "n_queries": Q, "accuracy_reproduced": float(C.mean()),
               "recorded_primary": rp, "validity_gate": gate,
               "gate_parts": {"accuracy": acc_ok, "loss": loss_ok, "split_fingerprint": fp_match,
                              "loss_reproduced": loss, "loss_recorded": rl},
               "value_in_window_frac": float(Wn.mean()),
               "reachable_frac": None if Rch is None else float(Rch.mean()),
               "seconds": round(time.time() - t0, 1), "device": dev, "tf32": tf32,
               "data_rng": man.get("data_rng", "legacy"), "layers": []}
        for li in range(nl):
            H = np.concatenate(lay[li]["hit_any"]); S_ = np.concatenate(lay[li]["sup_any"])
            D = np.concatenate(lay[li]["dmin"]); HH = np.concatenate(lay[li]["head_hit"])
            offs = {}
            for name, mk in (("0_key_block", D == 0), ("1", D == 1), ("2", D == 2), ("3_to_8", (D >= 3) & (D <= 8)),
                             ("over_8_or_none", D > 8)):
                offs[name] = {"frac": float(mk.mean()), "acc": rate(mk)}
            strata = []
            for lo, hi in STRATA:
                mk = (T >= lo) & (T < hi)
                strata.append({"q_from": lo, "q_to": hi, "frac_queries": float(mk.mean()), "acc": rate(mk),
                               "parent_hit_any_head": rate(mk, S_), "block_hit_any_head": rate(mk, H),
                               "acc_given_hit": rate(mk & H), "acc_given_miss": rate(mk & ~H),
                               "reachable_frac": rate(mk, Rch),
                               "parent_hit_recency_policy": rate(mk, Rec),
                               "parent_hit_random_policy": rate(mk, Rnd)})
            reach = {}
            for name, mk in (("reachable", Rch), ("unreachable", ~Rch)):
                reach[name] = {"frac": float(mk.mean()), "acc": rate(mk), "parent_hit_any_head": rate(mk, S_),
                               "block_hit_any_head": rate(mk, H)}
            out["layers"].append({
                "layer": li, "parent_hit_any_head": float(S_.mean()), "block_hit_any_head": float(H.mean()),
                "block_hit_per_head": [float(v) for v in HH.mean(0)],
                "acc_given_block_hit": rate(H), "acc_given_block_miss": rate(~H),
                "nearest_selected_offset": offs, "by_query_position": strata, "by_reachability": reach})
        report.append(out)
        print(json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    bad = [r["run"] for r in report if r["validity_gate"] is False]
    if bad:
        sys.exit(f"VALIDITY GATE FAILED (diagnostics untrusted): {bad}")


if __name__ == "__main__":
    main()
