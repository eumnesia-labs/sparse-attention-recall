# M4 routing-arms screen — protocol

**Frozen 2026-09-29, before any result.** Changing anything below after results exist must be
recorded here as a dated amendment, with the reason.

## Question

At the cheap operating point (S1=2, K=4; 34 selector pairs per query vs NSA's 128), why does hier
lose recall at L2048? Recorded S1=2 runs settle at 0.726–0.787, and one never learned (0.007),
against dense attention's 0.992. Two candidate causes, one arm each:

| Arm | Flag (`src/hier_nsa.py`) | Tests the idea that… |
|---|---|---|
| `baseline` | — | (control; also measures run-level instability) |
| `router` | `router_gate=True` | the level-1 router gets **zero gradient** (`ss` feeds only `topk(...).indices`), so it never learns which superblocks matter |
| `maxpool` | `superblock_pool="max"` | averaging ~176 tokens (~11 pairs) into one summary **dilutes** the matching key |
| `router+maxpool` | both | both contribute |

A Jev check of the router idea (two framings) returned *plausible but unproven*; it rated dilution
and instability as contributing factors and the router fix alone as likely partial.

## Fixed recipe

`m2c_budget_sweep.py`'s s2-k4 control, unchanged: L2048, 128 KV pairs, vocab 8192, d_model 128,
2 layers, lr 3.2e-4, weight decay 0.1, cosine over the epoch cap, **micro-batch 8 × grad-accum 2**
(effective 16), no clipping, 60 epochs, early stop at validation accuracy > 0.98. This is the
recipe that transitioned in both runs that used it (Kaggle 0.726, AWS 0.759). The true-batch-16
Modal control that never transitioned used a different recipe. zoology pinned at `1ad20d1`,
patched with `patch_zoology_accum.py`; torch 2.14.0, torchvision 0.29.0. Training data order is
fixed each epoch (zoology does not shuffle).

## Data and seeds

- `SEED` drives the data draw **and** weight initialization (as M3's SEED knob does). Screen: seed 123.
- Train 20,000 / validation 2,000 / **held-out test 2,000**. Train and validation seeds are
  zoology's own derivation; the held-out seed comes from a separate stream and is checked
  distinct. Each run records the number of identical sequences shared between splits (expected 0).
- **zoology fills MQAR noise tokens with unseeded `torch.randint`** (`multiquery_ar.py:137`).
  Freshly generated data therefore consumes the torch RNG while cached data does not, so a cold
  vs warm cache changes both the data and the training trajectory. This had not been recorded
  for any earlier run and is a candidate contributor to the unresolved M2c control disagreement.
  Here every arm generates its data cold, in its own container, and the arms have identical
  parameters. So **all four arms train on identical data from identical initial weights**
  (verified: H8 in `test_m4_harness.py`). Every manifest records the cache state.

## Selection and metrics

- Validation is used **only** for early stopping and checkpoint selection.
- **Primary metric:** held-out test accuracy of the best-validation checkpoint (earliest epoch on ties).
- Also reported: final-checkpoint test accuracy, the validation curve, the first epoch with
  validation > 0.30 (jump), and the mean of the last five validation values.

## Decision rules (set before results)

1. **One seed per arm is a screen, not evidence.** No arm is called effective until it is repeated
   at seeds 123, 231 and 312 alongside the baseline.
2. A run whose validation accuracy never exceeds 0.30 is a **non-transition**. On one seed it is
   reported as a failure event, not attributed to its arm (in M2c the same config and seed
   transitioned on one machine and not another).
3. **Screen "recovers":** primary accuracy ≥ baseline + 0.10, i.e. closing about half of the ~0.2
   gap to dense. The threshold is arbitrary, fixed in advance.
4. **Mission bar** (dense − 0.02 ≈ 0.97): comparing against it requires a dense-attention run
   **under this protocol** (same splits, same selection rule). The recorded 0.992 is a
   best-over-run *validation* number from the old protocol. The screen does not include dense.
5. **Only complete runs are results.** A run counts only if it trained its full protocol (60 epochs,
   or zoology's early stop at validation > 0.98) and `result.json` says `status: complete` with
   `time_capped: false`. A run that crashed, was killed by the timeout, or was cut short is
   incomplete: it is rerun with the same arm and seed, and never reported or averaged in.

## Budget and records

Modal L40S, one container per arm, in parallel. ~$2.14/h per container. Expected ~2.5 h and
~$5.35 per run (~$21 for four). **No soft time cap**: every run completes the protocol (rule 5).
The only guard is a hard Modal timeout of 6 h (~2.4× a healthy run), which can only stop a hung
container; a run it kills is incomplete and rerun. Launcher:
`experiments/modal_m4_arms.py` (plan / `--smoke` / `--go`; refuses a dirty tree).

Each run writes a new folder that is never reused: `manifest.json` (commit, dirty flag, file
hashes, seeds, cache state, hardware, versions), `metrics.jsonl`, `result.json` (always written,
including on failure), `train.log`, and `best.pt` / `final.pt` (not committed).

## Verified before launch

- `experiments/test_hier_arms.py`: default flags byte-identical to `ba09be2`; router arm keeps
  routing decisions and gives the router a gradient (none in the baseline); causality and
  numerical safety for every arm.
- `experiments/test_m4_harness.py`: the screen trainer is **bit-identical** to zoology's Trainer;
  records, time cap, crash handling, no-overwrite, unpatched-zoology refusal, cache-state
  recording, and cross-arm data identity.

## Known limitations

Single seed; d_model 128 and 2 layers; L2048 only; fixed training order. Scored-pair counts are
still not wall-clock cost.

## Addendum 2026-10-01 — "variance first" runs (written before any of these results)

Same recipe, splits, selection rule and primary metric as above. Four runs, one L40S each:

| run | purpose |
|---|---|
| dense, seed 123 | reference under this protocol (rule 4). Mixer = zoology MHA, dropout 0.1, 4 heads — M3's attention anchor; note hier has no dropout |
| baseline S1=2, seed 231 | run-to-run spread at the cheap point |
| baseline S1=2, seed 312 | run-to-run spread at the cheap point |
| baseline S1=12, seed 123 | top of the hier budget curve (12 = all super-blocks: no tree pruning) |

Decision rules, fixed before results:

- **V1 (spread).** The S1=2 spread is the range of held-out accuracy over seeds 123 / 231 / 312
  among runs that transition. An M4 arm difference smaller than this range is reported as within
  run-to-run variation, not as an effect of the arm.
- **V2 (non-transition).** A seed that never exceeds 0.30 validation is counted and reported as a
  failure event (convergence rate k/3); it is excluded from the spread, not averaged in as 0.
- **V3 (reference).** "No recall loss" means held-out accuracy ≥ dense − 0.02 under this protocol.
- **V4 (budget).** If S1=12 meets V3, the full-budget selector is lossless here and the S1=2 gap is a
  budget effect; if not, part of the gap is the hier architecture itself, not the budget.
- **V5 (follow-up).** A corrected router/maxpool arm is re-run only if an independent code review
  finds a concrete implementation flaw, and then at the seeds above, against the same baselines.

## Amendment 2 — 2026-10-01 ~10:45 IST, before any result of the variance batch

Written after an independent review (GPT-6.1 Sol, archived in
[`audits/sol-2026-10-01/`](../audits/sol-2026-10-01/REPORT.md); all its CPU reproductions were
re-run and matched) and while the batch `variance-20261001T044204Z` was training. No metric from
that batch had been read; only the 2-epoch smoke throughput had. These rules **replace V1, V2, V4
and V5** above; V3 stands.

**Facts the rules depend on.**
- *Data convention.* The harness seeds RNGs, builds the model, then generates train/validation
  data; zoology's MQAR fills noise tokens from the ambient torch RNG. All hier runs (any arm, any
  S1) consume the same draws, so hier runs at one seed share train/validation inputs. **Dense
  does not**: its train/validation noise tokens differ (labels and the held-out split are
  identical). The dense comparison is therefore *same test split, not paired training data*.
  This batch keeps the as-launched convention; it is the same convention as the 0.803 screen, so
  seed 123's 0.803 is a valid member of the S1=2 spread.
- *Old runs are not replicates.* The earlier S1=2 numbers (0.007, 0.726, 0.759) used different
  recipes and validation/plateau metrics; they are not used to estimate variance.
- *S1=12 is not pure budget.* At S1=12 the selector also visits the final, still-incomplete
  super-block as a masked placeholder and can attend its fully-past children; S1=2 cannot. An
  S1=12 vs S1=2 difference is budget **plus** candidate coverage.
- *Test reuse.* The seed-123 held-out split has now informed design decisions; seed-123 results
  are exploratory. A final claim needs data and test seeds not yet used.

**Rules.**
1. **Eligibility.** Count only complete, uncapped, zero-overlap runs of the intended protocol.
   Infrastructure failures are repeated unchanged. A completed run that never exceeds 0.30
   validation **stays in the primary-score summary** and is also counted as a non-transition.
   Report every score, curve, jump epoch, final score and epochs trained.
2. **Baseline spread.** R = max − min of the S1=2 baseline primary scores at seeds 123/231/312.
   R ≤ 0.05: small; 0.05 < R < 0.15: material; R ≥ 0.15: comparable to the M4 deficits. Report R,
   mean, median, sample SD and the non-transition count. Three draws do not give a stable variance.
3. **Reading M4 against R.** If R ≥ 0.15, the M4 deficits are one-seed paired observations amid
   large variability — not declared noise. If R ≤ 0.05 and all three transition, the −0.15
   deficits exceed the observed spread, which strengthens concern but does not prove harm. A
   baseline range is not the distribution of paired arm differences.
4. **Budget and mission (exploratory, seed 123).** S1=12 "exploratory recovery" if
   A(S1=12) − A(S1=2) ≥ 0.10. A hier point is "exploratory near-dense" if A ≥ A(dense) − 0.02. If
   S1=12 recovers and S1=2 does not, prioritise an S1=4/6 curve and independent confirmation. If
   both hier points miss dense and differ by < 0.05, selector budget alone is not supported as the
   explanation (K=4 reading, representation and training remain open). Otherwise: inconclusive.
5. **Corrected-arm reruns.** A corrected router/maxpool arm gets GPU time only with a
   failing-before / passing-after CPU regression that is relevant to observed training behaviour
   (e.g. recorded router-score telemetry), or as a stated new design hypothesis. The found
   sentinel/padding defects alone do not qualify. Final effectiveness needs matched
   corrected-arm/baseline pairs at three seeds, reporting the mean paired difference and all seeds.
6. **Next batch.** Isolate MQAR noise per segment (seeded, CPU generator only, new cache
   namespace, hashed inputs and targets) so every architecture gets identical data; that changes
   the data, so it needs a fresh S1=2 seed-123 baseline and is not pooled with this batch.
