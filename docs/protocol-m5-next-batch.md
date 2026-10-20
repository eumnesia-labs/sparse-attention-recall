# M5 — recall versus selector budget at L2048

**Draft dated 2026-10-01. Not frozen and not launch authorization.** No M5 GPU work is performed by this handoff. This draft predates inspection of the real `variance-20261001T044204Z` outcomes in this checkout. Fill the variance placeholders below, then freeze a new dated revision before any run governed by it. Preserve this draft and earlier protocols unchanged when doing so.

The question is whether recall approaches a matched dense reference as the hierarchical selector budget increases, under identical data and checkpoint selection. Stage 0 seeks a measurement that joins recall and cost at the same configuration; it does not require a positive architecture result to publish (`docs/MISSION.md:33`). The selector itself and its n^1.5 exponent are not claimed novel.

## Decisions pending the current variance batch

| Required input | Value in this draft | Consequence when available |
|---|---|---|
| Eligibility and completed S1=2 seeds 123/231/312 | **UNKNOWN** | Repeat failed infrastructure unchanged; retain completed non-transitions. |
| Legacy S1=2 R, mean, median, sample SD, non-transition count | **UNKNOWN** | Label small/material/comparable spread by the rules below; never call paired M4 deficits noise from R alone. |
| Legacy dense@123 and S1=12@123 primary held-out scores | **UNKNOWN** | Classify recovery and near-dense before deciding whether S1=4/6 deserves funding. |
| Relevant training telemetry for corrected arms | **UNKNOWN / not present in the archived screen** | No automatic corrected-arm rerun; apply the evidence/new-hypothesis gate below. |
| Actual remaining November credit, charged rate, and AWS funding | **UNKNOWN until execution** | Fund complete runs plus contingency before dispatch; quota is not credit. |

Use `experiments/analyze_variance.py` for the legacy batch only. Its frozen eligibility checks intentionally reject the new isolated convention. A separate M5 analysis profile and its synthetic tests must be frozen before M5 launch; do not silently reinterpret legacy eligibility. Numerical thresholds below mirror Amendment 2, but observations from the two data conventions are not pooled (`docs/protocol-m4-routing-arms.md:144`, `:166`).

## Computational convention

Use the pinned patched Zoology recipe and fixed order: L2048, 128 KV pairs, vocabulary 8192, width 128, two layers, four heads, block size 16, window 32, K=4, group g=11, micro-batch 8 with accumulation 2, learning rate 3.2e-4, weight decay 0.1, no clipping, cap 60 epochs, and stop at the **first validation accuracy >0.98**. Train/validation/held-out sizes are 20,000/2,000/2,000. Dense is Zoology MHA with dropout 0.1; hier has no dropout. This matches the recipe, not parameter count or full compute (`experiments/m4_routing_arms.py:47`, `:48`; `docs/protocol-m4-routing-arms.md:23`).

All pilot runs explicitly select `--data-rng isolated`, cache namespace `mqar-isolated-cpu-v1`, and class `m4_data.SeededMQARConfig`. Every segment seeds only the CPU noise generator and restores its previous state. Dense and hier at one root seed must have identical input **and target** SHA-256, shape and dtype for all three splits; cold, warm and partial cache states must agree. Preserve root seed, derived segment seeds, class/version, cache state, recipe, mixer kwargs, code hashes, dirty flag and hardware in each manifest (`experiments/m4_data.py:13`, `:16`, `:37`; `experiments/m4_routing_arms.py:324`). Reject any unexplained hash mismatch. Check zero identical-sequence overlap between all splits.

The pilot changes data convention only: hierarchy is `baseline`, `mask_mode=sentinel`, mean pooling, router gate off. Enable observational router telemetry for every hier pilot (`--router-telemetry`); record its sampling method, training-only forwards, exact counts and approximate quantiles. Quantiles do not certify absence of rare numerical failures; exact row-max threshold counts/nonfinite counts are the relevant evidence (`src/hier_telemetry.py:28`, `:93`). Telemetry overhead has not been measured on GPU. A boolean-mask or corrected router/maxpool experiment is a separately named profile, never a silent substitution.

At L2048 the scored selector pairs/query are `12 + min(S1,12)*11`: S1=2/4/6/12 give 34/56/78/144. These include masked/padded visits and exclude fine/window reads, projection work and runtime overhead. S1=12 visits every parent and can reach fully-past children of the final incomplete parent; the S1 comparison therefore changes both budget and candidate coverage (`src/hier_nsa.py:177`, `:203`; `docs/protocol-m4-routing-arms.md:138`). It is not a pure selector-budget causal intervention.

## Pilot priorities and November budget

The handoff allocates approximately $30 Modal credit on 1 November and reports one approved us-east-1 AWS L40S quota. Neither is verified account state or permission to spend here. Planning uses the archived two-epoch smoke projections, not new benchmarks: dense 2.1388 h, S1=2 seeds 231/312 2.5057/2.4554 h, S1=12 4.1566 h (`experiments/m4_results/smoke-20261001T043739Z.json:48`, `:101`, `:154`, `:207`). Use the requested rounded planning envelope 2.14/2.50/4.16 h at the recorded approximate Modal rate $2.14/h (`docs/protocol-m4-routing-arms.md:72`).

| Priority | Complete run | Planned hours | Planned cost | Cumulative cost |
|---|---|---:|---:|---:|
| 1 | Fresh isolated baseline S1=2 @123 | 2.50 | $5.35 | $5.35 |
| 2 | Isolated dense @123 | 2.14 | $4.58 | $9.93 |
| 3 | Isolated baseline S1=2 @231 | 2.50 | $5.35 | $15.28 |
| 4 | Isolated baseline S1=2 @312 | 2.50 | $5.35 | $20.63 |
| 5 | Isolated baseline S1=12 @123 | 4.16 | $8.90 | **$29.53** |
| 6–7 | Isolated S1=4, then S1=6 @123 | **unmeasured** | **unbudgeted** | Additional funding needed |

Unrounded arithmetic is 13.80 h × $2.14 = $29.532, leaving only $0.468 (1.56%) of $30. This is **not** a safely funded five-run launch. Projection already includes measured setup/validation under its assumptions; new image builds, telemetry overhead, changed throughput, billing details and retries can exceed it. Do not assume early stopping savings. The first four project to $20.6296, leaving $9.3704; their 20% planning reserve is $4.12592 and fits within $30. That reserve is a draft spending rule, not an empirical confidence bound.

The initial stage targets the first four runs. Dispatch serially in priority order, with a checkpoint after the first dense/hier pair and after each later run. Maintain a **20% contingency on the remaining projected runs**; admit a run only if unspent credit covers its complete projection plus that reserve **and** its full hard-timeout liability. The saved launcher rate is $2.1406/h and its timeout is six hours, approximately $12.84 per job before unmeasured ancillary charges (`experiments/modal_m4_arms.py:56`, `:60`). Admission means balance ≥ max(next-run projection + 20% of remaining staged projections, $12.8436 timeout liability), plus separately estimated ancillary charges. After the four projected runs, $9.3704 remains: S1=12 fails both its $10.68288 projection-plus-reserve and $12.8436 liability gates, requiring at least $3.4732 savings/top-up for the latter. This timeout is a hang guard, not a smaller scientific epoch cap. Five simultaneous timeouts would cost approximately $64.22, so the expected-cost fit does not justify parallel dispatch. Five runs would require $35.4384 under the 20% expected-cost guard alone. S1=12 is therefore conditional on measured savings or additional funding; never stop it early to fit credit. Do not reduce epochs/split sizes or count a time-capped run as a result. Preserve failed/capped records, and fund an unchanged retry before starting it. No corrected-arm pair or S1=4/6 sweep fits in this envelope.

AWS alternative: after explicit funding and launch authorization, use the one L40S serially in the same priority order. A quota grant does not establish capacity, price or a budget. Before freezing an AWS plan, record the chosen instance, current regional on-demand/spot price, disk/transfer costs, availability and interruption/retry allowance; **AWS price and total cost remain UNKNOWN here**. Do not substitute the Modal hourly rate or assume equal throughput. Mixed-hardware runs require reporting hardware and an unchanged recipe; never interpret hardware differences as model effects.

## Eligibility, metrics and decision rules

1. Count only the frozen isolated profile: complete, uncapped, zero-overlap, matching fingerprints/recipe/arm/mixer/S1/dropout/mask convention, and full 60 epochs or the legitimate first >0.98 early stop. Missing/duplicate cells remain unknown/ambiguous; infrastructure failures are repeated unchanged. A completed run whose validation never exceeds 0.30 stays in primary summaries and counts as a non-transition. Report every curve, first >0.30 jump epoch, epoch count, final score and failure record.
2. Primary A is **held-out accuracy of the checkpoint with best validation accuracy, earliest epoch on ties**. Select with validation only. Also report final-checkpoint held-out accuracy and last-five-validation mean (a plateau diagnostic, not the primary). No best-test checkpoint selection or exclusion of non-transitions (`experiments/m4_routing_arms.py:339`; `docs/protocol-m4-routing-arms.md:46`).
3. For isolated S1=2 seeds 123/231/312 compute R=max−min, mean, median, sample SD and non-transition count. R≤0.05 is small; 0.05<R<0.15 material; R≥0.15 comparable to historical M4 deficits. Three draws do not estimate stable variance; this spread mixes initialization and data variation. Never add legacy scores to this sample.
4. Historical paired M4 arm deficits remain paired one-seed observations and are read **only against the eligible legacy R**, never against the isolated R from rule 3. Large legacy R does not make them noise. Small legacy R with all transitions strengthens concern that deficits exceed the observed legacy spread, without proving harm. Isolated R describes the new convention separately; it does not calibrate old deficits.
5. At pilot seed 123, S1=12 recovers if A12−A2≥0.10; a hier point is exploratory near-dense if A≥Adense−0.02. If S1=12 recovers and S1=2 misses near-dense, prioritize S1=4/6 plus independent confirmation; recovery need not itself reach near-dense. If both miss near-dense and |A12−A2|<0.05, budget alone is not supported as the explanation. Otherwise inconclusive. K=4 reading, representation, coverage and training remain open (`docs/protocol-m4-routing-arms.md:156`; `audits/sol-2026-10-01/REPORT.md:118`).
6. A corrected arm gets GPU time only with a failing-before/passing-after CPU regression relevant to observed training behavior (telemetry is the evidence route), or as an explicitly preregistered new design hypothesis. Record which path qualifies it. CPU-discovered sentinel/padding defects alone do not qualify. Final effectiveness needs corrected-arm/baseline pairs at three seeds, reporting every paired difference and their mean (`docs/protocol-m4-routing-arms.md:161`).

## Reserved untouched confirmation seeds

Pilot root seeds 123/231/312 are exploratory. Seed 123's test has already informed design. Reserve the following **root/data/initialization seed triplets**, with their deterministically derived segment seeds, for a frozen final dense-versus-selected-hier confirmation. Do not generate, inspect, benchmark, tune on, or evaluate these datasets during development. Root seed also controls model initialization; this is a joint data/init seed design, not isolated training-noise variance.

| Reserved root seed | Train segment | Validation segment | Held-out test segment |
|---:|---:|---:|---:|
| 810001 | 2090300952 | 3593942460 | 3448931778 |
| 810002 | 1226899916 | 3650199530 | 3829225085 |
| 810003 | 596186643 | 4245952058 | 2414974315 |

These numbers use the current NumPy train/validation derivation and held-out salt `0x5EED7E57` (`experiments/m4_routing_arms.py:53`, `:271`). Before writing this reservation, root and segment integers were absent from the non-PDF tracked experiment/docs records at `f6b38a6`; only integer derivation was evaluated, no data was generated. This establishes no recorded use in this checkout, **not knowledge of unrecorded external runs**. The owner must check external caches/logs before freezing; replace any already-used seed by a new dated reservation before generation. All nine segments are distinct.

Freeze model/reader/selector budget, mask/data conventions, optimizer, stop/checkpoint rules and reporting before revealing final tests. For each root seed run dense and the selected hierarchy on the exact same split tensors and record hashes. Select at most one hier operating point using pilot evidence; release all three confirmation scores and dense-relative paired differences, including non-transitions. No retuning after final test inspection; any new design requires another reserved untouched cohort. Three pairs support a limited reproducibility observation, not universal no-loss or a stable population variance claim. Final confirmation is additional work and funding beyond the $30 pilot.

## Before this draft can be frozen

- Populate the legacy variance table from collected artifacts, preserving its data-convention caveat.
- Select priorities under those rules; record any change as a new dated revision.
- Resolve external seed-use checks and freeze a tested isolated analysis profile.
- Verify actual funding/rate, code hashes and CPU suites; obtain separate spending/launch authorization.
- Retain sentinel pilot masks unless a separately named corrected design passes rule 6; report telemetry overhead uncertainty.

No real variance results, November account balance, AWS price, or M5 quality outcome are asserted in this draft.
