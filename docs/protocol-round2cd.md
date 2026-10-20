# M5 follow-ups on AWS: S1=6 confirmation (round 2c) and dense check (round 2d)

Frozen 2026-10-05, before any round-2c/2d data. Owner-approved: two machines, cap $175.

**Code:** `arch/bounded-index` commit `f7269dde1feb7001a060d3e10f7352c16678969e`
(`experiments/round2c_plan.py`, `experiments/round2d_plan.py`, `experiments/round_runner.py`).
Each machine runs exactly that commit. The tarballs are in `s3://<checkpoint-bucket>/round2c/code/` and
`.../round2d/code/`.

## Why

M5 ([results](../experiments/m5_results/RESULTS.md); pre-registration [protocol-round2.md](protocol-round2.md))
left two gaps for the Stage 0 write-up:
1. **S1=6 was the best budget point (0.962), but on one seed only.** Dense never transitioned at that seed, so
   S1=6 was never compared with a working dense.
2. **Dense transitioned in only 1 of 4 M5 runs** (TF32 + isolated data). In M4 (float32 + legacy data), dense@123
   reached 0.992. M5 changed the numerics and the data generator together.

## Common

M5's recipe exactly, unchanged:
- zoology MQAR at L2048, 128 pairs, vocab 8192;
- d_model 128, 2 layers, 4 heads;
- AdamW lr 3.2e-4, weight decay 0.1, cosine over 60 epochs;
- micro-batch 8 × accumulation 2;
- early stop at validation > 0.98;
- splits 20k / 2k / 2k;
- **primary metric:** held-out accuracy at the best-validation checkpoint.

Same machine type (g6e.xlarge, L40S). Runs go one at a time. A "transition" means validation > 0.30 within 60
epochs.

## Round 2c (machine 1): S1=6 confirmation

**Runs:** hier S1=6 at seeds 810001, 810002, 810003, with M5's exact hier arguments:
- baseline arm, K=4, sentinel masks, mean pooling, router gate off, router telemetry;
- TF32, isolated data RNG.

**Status of this test:** S1=6 was **selected after** M5's pilot (highest seed-123 validation). This is a follow-up
on a selected point, not part of M5's frozen design, and it is reported as such.

**Pairing:** the isolated RNG gives each seed the same data as M5's dense and S1=2 runs on that seed. This is
verified by split hashes before any comparison.

**Reported, all values, with nothing selected or tuned on them:**
1. Each run's held-out accuracy, jump epoch and transition. Then the mean, median and minimum over the three, with
   non-transitions counted, never dropped.
2. Per seed, S1=6 against M5's S1=2 on identical data (paired).
3. Per seed, S1=6 against M5's dense. The difference is a valid dense comparison **only where that dense
   transitioned** (810002: dense 1.000).
   - **Pre-set reading:** S1=6 is "near-dense at 810002" iff its held-out accuracy is ≥ 0.98 (dense − 0.02, the
     M5 margin).
   - At 810001 and 810003, dense did not transition, so no dense-gap claim is made there.

## Round 2d (machine 2): dense check

**Runs, at seed 123, each changing ONE thing back toward M4:**
1. Dense, **TF32**, **legacy** data RNG: M4's data with M5's numerics. Cold data cache, as in M4.
2. Dense, **float32**, **isolated** data RNG: M5's data with M4's numerics.

**Pre-set reading:** a single seed is suggestive, not conclusive, because dense transitions are stochastic (1 of 4
in M5).

| run 1 (TF32, legacy) | run 2 (fp32, isolated) | reading |
|---|---|---|
| transitions | does not | the isolated data generator (or its seed-123 draw) is implicated, not TF32 |
| does not | transitions | TF32 is implicated |
| transitions | transitions | neither change alone explains it; the combination, or seed-level chance |
| does not | does not | neither single change restores dense@123; seed variance, or a code/hardware change since M4 (Modal, older code). This needs an fp32 + legacy positive control (not in this round) |

**Consequence for the write-up:**
- If TF32 is implicated, the paper's dense comparison must use float32, and M5's TF32 hier numbers are reported
  with that caveat.
- If the data generator is implicated, dense's reliability under isolated data becomes a reported finding.

## Machines, budget, stopping

- **Two g6e.xlarge machines, one per slot:**
  - slot a: round 2c, `control/`;
  - slot b: round 2d, `control-b/`;
  - launched in us-east-1 or us-west-2, wherever quota and capacity allow.
- **Cap: $175 for rounds 1, 1b, 2, 2c and 2d together**, owner-approved 2026-10-05. The meter was at $123.21
  before launch, and every machine's time is counted once.
- **Estimates:**
  - 2c: about 11 h, about $22 (S1=6 smoke: 3.5 h per 60-epoch run);
  - 2d: about 6-8 h, about $14 (float32 is slower; early stop shortens a run that learns).
- **Stopping:** each machine shuts down **0.25 h after its own queue finishes** (owner: stop when done). Neither
  experiment is interrupted to save time. Results and checkpoints stay in S3.

## Reporting

- Results go to `master` under `experiments/m5_results/` (or a follow-up folder), as old-architecture results.
- Every run is reported, including non-transitions and failures.
- These runs are not pooled with M4's float32 / legacy results, except where the dense check is the explicit
  comparison.
