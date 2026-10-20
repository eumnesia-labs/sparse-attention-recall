# Round 2 on AWS: pre-registration (frozen 2026-10-02, before any round-2 data)

Branch `arch/bounded-index`. The run lists and rules are `experiments/round2a_plan.py` (machine A)
and `experiments/round2b_plan.py` (machine B). They are frozen at the commit that adds this document,
and each machine runs exactly that commit.

This document is the frozen, dated revision of the M5 draft (`docs/protocol-m5-next-batch.md`, kept
unchanged), plus a first screen of the new architecture. The owner approved two machines.

## Why round 2

Rounds 1 and 1b tried to compare the architectures on spread-MQAR, where keys sit anywhere in the
context. The dense reference never learned any spread variant at the M4 recipe:
- spread-v1: 0.0044 (round 1);
- 100k sequences × 12 epochs: 0.0023;
- 32 pairs: 0.0002;
- zoology distances: the third rung (`docs/protocol-arch-round1b.md`). It is reported, but the rounds
  above are already INCONCLUSIVE whatever it gives.

Validation loss rose while training loss fell: memorisation. That exam cannot judge any
architecture. Round 2 returns to the standard benchmark, zoology MQAR, where this exact recipe's
dense reference learns recall (M4: 0.992 held-out).

**Caveat stated up front.** In zoology MQAR every key sits in the first 256 tokens, so a model can
use position as a shortcut (Step A found the old hierarchy doing so). Round 2's results are results
*on that benchmark*. A test without the shortcut is deferred, and nothing here claims position-free
retrieval.

## Common to every run

- **Task and recipe:** zoology MQAR with isolated data RNG (`m4_data.SeededMQARConfig`, cache
  `mqar-isolated-cpu-v1`), and the M4 recipe:
  - L2048, 128 pairs, vocab 8192, d_model 128, 2 layers, 4 heads;
  - AdamW lr 3.2e-4, weight decay 0.1, cosine over 60 epochs;
  - micro-batch 8 × accumulation 2;
  - early stop at validation > 0.98.
- **Splits:** 20k / 2k / 2k.
- **Primary metric:** held-out accuracy at the best-validation checkpoint.
- **TF32 matmuls (`--precision tf32`) for every run.** This is a new numerics profile, so round-2
  numbers are **not pooled** with M4's float32 legacy results; those are quoted only as context.
- Same seed → identical data on both machines (isolated RNG). This is verified by the split
  fingerprints in the manifests before any cross-machine comparison.

## Machine A: the old architecture's M5 study (Stage 0 paper)

The M5 draft's placeholders are filled from the collected M4 batch, on legacy data and float32
(context only):
- dense@123 = 0.992;
- S1=12@123 = 0.920;
- S1=2 @123/231/312 = 0.803 / 0.715 / 0.008: R = 0.796, comparable to the M4 deficits, with 1
  non-transition.

The draft's rule 5 classifies S1=12 as a recovery (+0.117) that is not near-dense, so its priority
list runs, extended to the full budget curve.

**Stage P (pilot), in this order:**
1. hier S1=2 @123
2. dense @123
3. hier S1=2 @231
4. hier S1=2 @312
5. hier S1=12 @123
6. hier S1=4 @123
7. hier S1=6 @123

The hier settings are the baseline arm, sentinel masks, mean pooling, router gate off and
observational router telemetry. No corrected arm runs (draft rule 6).

**Rule P (confirmation point, seed 123 VALIDATION accuracy only; held-out never selects):**
- Choose the smallest S1 in (2, 4, 6, 12) whose best validation accuracy is ≥ dense's − 0.02.
- If none qualifies, choose the S1 with the highest best validation accuracy (ties → smaller S1).
- If dense or any seed-123 hier run failed, no point is chosen and the machine holds for the
  owner.

**Stage F (confirmation):** dense and the chosen hier point on the reserved seeds 810001, 810002,
810003.
- These seeds were reserved on 2026-10-01. Their integers appear nowhere in the repository's run
  records. The owner confirms, by approving this round, that no external run used them.
- All three paired dense − hier differences are released, including non-transitions.
- There is no retuning after any confirmation result.

**Reporting:** as in the M5 draft (rules 1–5), separately for the isolated profile:
- every curve and the jump epoch;
- the S1=2 spread over the three pilot seeds (R, mean, median, sample SD, non-transitions);
- the recall-vs-budget curve S1 = 2/4/6/12 at seed 123, with scored selector pairs/query 34/56/78/144;
- the confirmation pairs.

## Machine B: the new architecture's first screen

The indexed reader (`src/indexed_reader.py`): the Step C model of `docs/arch-gates.md` Amendment 1,
unchanged.

**Stage S, seed 123, in this order:** (λ_route, τ) = (1, 1), (0.1, 1), (1, 0.1).
- Futility rule: at epoch 10, validation candidate support recall < 0.01 stops training. Held-out
  evaluation still runs.

This is a **one-seed screen, not the Step C verdict.** Reported against machine A's dense@123 and
hier@123 (same data):
- overall held-out accuracy;
- in-window and out-of-window accuracy;
- retrieval-layer candidate support recall;
- index telemetry.

Setting selection, for a later three-seed confirmation on 810001–3 with its own budget, uses
validation only. Step C stays INCONCLUSIVE until a pre-registered three-seed comparison is run.

## Machines, budget, holding

- **Two g6e.xlarge machines.** Each has its own slot:
  - A uses `control/` (the machine already held, reloaded from round 1b);
  - B uses `control-b/`, launched in us-west-2 or us-east-1, whichever region has quota free.
  - One machine per slot; the launcher re-checks before every attempt.
- **Cap $160 for rounds 1, 1b and 2 together** (owner-approved). Every machine's time is counted once
  across all stores.
  - Each machine moves its hard backstop to the remaining budget ÷ (rate × machines spending now).
- **Estimates, with no early stopping assumed:**
  - A: pilot ~$37 and confirmation ~$28–35 (TF32 smoke timings);
  - B: ~$59 for three indexed runs of ~10.4 h each (fewer with futility or early stopping);
  - total ~$124–131 against ~$140 left when round 2 starts. It is tight; the budget hold asks the
    owner before the cap.
- The smoke runs every kind alone. The concurrency sets are not repeated: round 1b's smoke on this
  machine type found 2 and 3 copies slower in total for every kind (0.85–0.89), so every job runs
  alone.
- **Holding:** each machine holds until released after its queue (48 h at most per spec). The budget
  hold pauses 2 machine-hours before the cap and waits for the owner. Crash holds and watchdogs are
  as in round 1b.

## Not claimed

- Anything about spread (position-free) retrieval.
- A Step C verdict.
- Speed or O(n) wall-clock.
- Lengths beyond 2048.
- Pooling with legacy float32 results.
