# Selection-vs-reading diagnostic on the M5 checkpoints (2026-10-20)

Owner-approved Part 2 of the paper revision (independent review item 7(ii)1). Inference only, no training.
`experiments/diagnose_routing.py` (extended for isolated data, CUDA and run-manifest TF32) replayed each run's full
held-out split (2,000 sequences, 256,000 queries) on Modal L40S via `experiments/modal_diagnose_m5.py`.
Checkpoints: each run's `best.pt` from the round 2a / 2c S3 store. Raw output: `routing-m5-full.json`.

**Validity gate: 12 / 12 pass.** Every replay reproduced the run's recorded held-out accuracy exactly, its held-out
loss within tolerance, and its held-out split SHA-256 fingerprint.

## Retrieval layer (layer 1), any of 4 heads

| run | recall | parent hit | key-block hit | recall given hit | recall given miss | parent-level miss | fine-level miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| S1=2 seed 123 | 0.705 | 0.859 | 0.463 | 0.986 | 0.463 | 0.141 | 0.397 |
| S1=2 seed 231 | 0.684 | 0.861 | 0.627 | 0.999 | 0.155 | 0.139 | 0.234 |
| S1=2 seed 312 | 0.836 | 0.955 | 0.565 | 0.990 | 0.637 | 0.045 | 0.390 |
| S1=2 seed 810001 | 0.578 | 0.869 | 0.570 | 0.978 | 0.048 | 0.131 | 0.299 |
| S1=2 seed 810002 | 0.709 | 0.864 | 0.526 | 0.997 | 0.389 | 0.136 | 0.338 |
| S1=2 seed 810003 (never learned) | 0.011 | 0.809 | 0.383 | 0.022 | 0.005 | 0.191 | 0.426 |
| S1=4 seed 123 | 0.884 | 1.000 | 0.727 | 0.998 | 0.578 | 0.000 | 0.273 |
| S1=6 seed 123 | 0.962 | 1.000 | 0.962 | 1.000 | 0.002 | 0.000 | 0.038 |
| S1=6 seed 810001 | 0.937 | 1.000 | 0.933 | 1.000 | 0.060 | 0.000 | 0.067 |
| S1=6 seed 810002 | 0.945 | 1.000 | 0.845 | 1.000 | 0.644 | 0.000 | 0.155 |
| S1=6 seed 810003 | 0.945 | 0.996 | 0.748 | 1.000 | 0.781 | 0.004 | 0.248 |
| S1=12 seed 123 (flat) | 0.850 | 1.000 | 0.846 | 1.000 | 0.028 | 0.000 | 0.154 |

Parent-level miss = 1 − parent hit; fine-level miss = (1 − key-block hit) − parent-level miss.

## Reading (descriptive, post hoc)

1. **Reading is not the bottleneck on the M5 profile either.** Recall given a key-block hit was 0.978–1.000 for all
   11 models that learned (the one that never learned: 0.022).
2. **S1=2 models miss the parent even though position would suffice.** Every key lies in super-blocks {0, 1}, yet
   the S1=2 models missed the key's super-block on 4.5–19.1% of queries. From S1=4 upward, parent misses vanished
   (0.000–0.004): with slack beyond two super-blocks, the right one is always kept.
3. **Most of the S1=2 → S1=6 gain is better selection at both levels.** Paired on the same seed, the key-block hit
   rose from 0.463 to 0.962 (seed 123), 0.570 to 0.933 (810001) and 0.526 to 0.845 (810002); parent misses fell from
   0.13–0.14 to 0, and fine misses from 0.30–0.40 to 0.04–0.16.
4. **Flat selection (S1=12) chose the key block less often than S1=6 on the pilot seed** (0.846 vs 0.962), all of it
   at the fine level (0.154 vs 0.038), where it chooses among all past blocks instead of 66 candidates. One seed.
5. **Recall after a miss varies widely across seeds** (0.002–0.781), so how much a model recovers from a wrong choice
   is a seed-level property here, not a property of the budget.

## Cost

12 containers × about 1 min of replay plus start-up; Modal L40S, well under $2 including the smoke test.
