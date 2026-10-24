# Dense in float32 on confirmation seeds 810001 and 810003 (rounds 2e and 2f)

Frozen 2026-10-06, before any round-2e/2f data. Owner-approved 2026-10-06, on AWS, under the existing $175 cap.

**Code:** the development branch commit `03dee3ec86c85d53fab0fbf2c0dcb97f29fbec3b`
(`experiments/round2e_plan.py`, `experiments/round2f_plan.py`, `experiments/round_runner.py`). Every machine runs
exactly that commit. The tarballs are in `s3://<checkpoint-bucket>/round2e/code/` and `.../round2f/code/`.
Build check: `audits/verify-round2ef-build-2026-10-06/` on the branch (tests Q22, Q20, Q18 pass).

## Why

After M5 and its follow-ups ([results](../experiments/m5_results/followup/RESULTS_followup.md)), only one
confirmation seed (810002) has a transitioned dense run, so only one dense-vs-hier pair is valid. M5's dense on
810001 and 810003 (TF32 + isolated data) did not transition within 60 epochs. The dense check found that dense in
**float32** learns on the isolated data at seed 123, late (epoch 41). These two runs ask whether float32 dense
transitions on the other two confirmation seeds, so that each seed can have a valid pair.

## Runs

| round | run | data |
|---|---|---|
| 2e | dense, **float32**, isolated data RNG, seed **810001** | identical to M5's dense, hier S1=2 and hier S1=6 runs on 810001 |
| 2f | dense, **float32**, isolated data RNG, seed **810003** | identical to M5's dense, hier S1=2 and hier S1=6 runs on 810003 |

Everything else is M5's recipe, unchanged: zoology MQAR at L2048, 128 pairs, vocab 8192; d_model 128, 2 layers,
4 heads; AdamW lr 3.2e-4, weight decay 0.1, cosine over 60 epochs; micro-batch 8 × accumulation 2; early stop at
validation > 0.98; splits 20k / 2k / 2k. **Primary metric:** held-out accuracy at the best-validation checkpoint.
A "transition" means validation > 0.30 within 60 epochs. Identical data is verified by split hashes before any
comparison.

## Pre-set reading

Per seed, report held-out accuracy, best validation and its epoch, jump epoch, epochs run and transition, whatever
they are.

- **Dense transitions on a seed:** that seed gets a valid pair. Report dense − S1=6 and dense − S1=2 on it.
  - S1=6 is "near-dense" on that seed iff its held-out accuracy is ≥ dense − 0.02, the M5 margin.
  - Stated in advance: S1=6 already scored 0.937 (810001) and 0.945 (810003), and dense cannot score above 1.000, so
    the S1=6 gap on these seeds is at most 0.063. These runs mainly test whether the dense baseline is reliable, and
    they give S1=2 its first valid gaps on these seeds.
- **Dense does not transition on a seed:** it is reported as a non-transition and the seed keeps no valid pair. No
  rerun with a longer epoch ceiling or other changes happens in this round; any such run needs a new
  pre-registration.
- **The write-up's dense column** then mixes numerics: TF32 at 810002 (M5) and float32 at 810001 / 810003. Every
  hier run is TF32. Both facts are stated wherever the pairs are quoted.
- **Recorded predictions (made 2026-10-06, before these runs; advisory, not evidence):** P(dense transitions on
  both seeds) Jev 0.40 / 0.41 (two option orders), Claude 0.50; P(on at least one) Jev 0.57 / 0.55, Claude 0.85.
  The Jev request and response files are kept outside the repo until the owner asks to add them.

## Machines, budget, stopping

- **g6e.xlarge (L40S) on AWS**, same machine type as M5 and round 2c. Round 2e runs on slot a (`control/`).
  Round 2f runs either on a second machine (slot b, `control-b/`, in another region, since quota allows one GPU per
  region), or on slot a after round 2e (a posted run spec). The choice changes only wall time, not the runs.
- **Cap: $175 for all AWS rounds together** (owner-approved 2026-10-05). The meter was $143.88 before launch.
  Estimate: about 2.3-2.6 h per run including boot and smoke, about $5 each; early stop shortens a run that learns.
- **Stopping:** each machine shuts down 0.25 h after its own queue finishes. Neither run is interrupted to save time.
  Results and checkpoints stay in S3.

## Reporting

Results go to `master` (old-architecture results) under `experiments/m5_results/followup/` or a sibling folder.
Every run is reported, including non-transitions and failures.

## Amendment 1 (2026-10-06, committed 09:56:39 IST, before any round-2e/2f data)

The owner moved seed **810003 to Modal** to finish sooner; seed 810001 stays on AWS (round 2e, slot a, hunting
since 09:55 IST).
- **Modal run:** dense, float32, isolated data RNG, seed 810003 (`experiments/modal_m4_arms.py --runs dense@810003
  --data-rng isolated --launch`, launch `custom-20261006T042651Z`, spawned 09:56:51 IST), from commit `03dee3e`. Same GPU type
  (L40S) and the same run code as AWS:
  `experiments/m4_routing_arms.py`, `experiments/m4_data.py` and `src/` are identical at `f7269dd` (round 2c) and
  `03dee3e` (`git diff` is empty). Software differs as in the dense check (Modal: Python 3.11, torch 2.14.0).
- **Fallback, set now:** if AWS has not started a machine for 810001 by the time the Modal run completes, 810001
  also runs on Modal the same way and the AWS hunt is stopped. Round 2f is not launched on AWS unless the Modal
  run fails.
- Reading, recipe and reporting are unchanged.
