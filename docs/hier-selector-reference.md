# Hierarchical selector: implementation and evidence

Updated 2026-09-25 after the no-GPU audit. Historical derivations and claims remain
in Git. Current source: [hier_nsa.py](../src/hier_nsa.py). Research scope:
[novelty review](novelty-review.md), [provenance reconciliation](provenance-reconciliation.md).

## Scope

The study measures synthetic MQAR recall and selector score counts in tiny models.
It does not demonstrate real-LM capability, million-token recall, wall-clock speedup,
or novelty of hierarchical attention as a general idea. Width matching is not
parameter, recurrent-state-memory or compute matching.

## Task and model

MQAR presents key/value associations followed by queries. Accuracy is evaluated at
query positions. M3 increases length and pair count together (N_kv = L/16), keeping
pair density constant. It measures joint length-and-load scaling, not length alone.
Reported values select validation checkpoints; there is no separately established
untouched-test estimate or multi-seed M3 uncertainty interval.

The hier module combines three branches with per-head sigmoid gates:

```
out_t = gate_coarse * coarse_t + gate_fine * fine_t + gate_window * window_t
```

The architecture is inspired by NSA's branch structure, but the implementations
are not otherwise identical. The pinned NSA includes RMSNorm, rotary positions,
learned compression MLPs and memory keys, different gate initialization and a
causal current-block fine path. Hier uses mean pooling and its own selected-score
gate. See [implementation comparison](implementation-comparison.md). Existing
experiments therefore do not isolate routing as the only changed component.

## Pooling and causal selection

At block size c=16, mean-pool keys and values into nblk=ceil(L/c) blocks.
Pool blocks into groups of g=max(2,round(sqrt(nblk))) and nsup=ceil(nblk/g) supers.
Padding is zero-valued. Block summaries are masked unless fully past the query.

For each query:

1. Score all nsup super summaries; mask super summaries not fully past.
2. Take the top s1e=min(S1,nsup) super indices.
3. Gather their s1e*g candidate block summaries and score them.
4. Mask future/invalid candidates and compute coarse attention over candidates.
5. Select ke=min(K_SEL,s1e*g) blocks and gather ke*c raw tokens for fine attention.
6. Add the local causal window, then combine branches.

The implementation zeroes branches with no valid candidate to avoid using an
all-masked softmax result. The existing perturbation tests are initial safeguards,
not exhaustive proof for all boundary lengths and budgets. Correctness tests of
routing and masking are separate from this reporting correction.

## Cost definitions

The corrected hier selector score-tensor count per query/head is:

```
C_hier = nsup + min(S1,nsup)*g
```

It includes scores computed for masked or padded candidates. It excludes pooling,
projection, top-k, gathers, fine/local attention, backward and hardware overhead.
The historical NSA comparator is **block-summary scores**, `C_nsa_blocks=nblk`.
The pinned NSA also prepends one learned memory key, so its full coarse-score
count is `nblk+1`. Keep the block proxy explicitly labeled; neither count is FLOPs.

| L | nblk | g | nsup | NSA block proxy | NSA full coarse count | hier S1=2 | hier S1=12 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 8 | 3 | 3 | 8 | 9 | 9 | 12 |
| 512 | 32 | 6 | 6 | 32 | 33 | 18 | 42 |
| 2048 | 128 | 11 | 12 | 128 | 129 | 34 | 144 |
| 4096 | 256 | 16 | 16 | 256 | 257 | 48 | 208 |
| 8192 | 512 | 23 | 23 | 512 | 513 | 69 | 299 |

The old S1=12 counts 39/78 at L128/L512 did not clamp S1. Raw logs are historical
records; current reporting is corrected to12/42. Counts at longer lengths are
analytic extrapolations, not successful recall measurements.

For fixed S1 and c, with g approximately sqrt(nblk), the total selector score
count is O(L^1.5). This is derived from the loops, not an exponent fitted to three
measurements. Fixed g would leave a quadratic term. The unclamped asymptotic
crossover against the block proxy is approximately nblk>(1+S1)^2; rounding,
clamping and the NSA memory key affect finite-length comparisons.

S1=12 at L2048 gives144 versus NSA128 block scores (129 full coarse scores).
The approximately3.8x reduction at S1=2 does not apply to M3's S1=12 recall.

Fine/window allocated slots for hier are `min(K_SEL,s1e*g)*c + window_size`.
They can include masked candidates and overlapping positions: they are not a count
of unique tokens read, nor evidence about information carried by coarse summaries.
At L2048,k16 the nominal288 slots are14.1% of2048, not16%; this is not a total-context
access percentage because the coarse branch also reads summaries.

## Gradients and numerical behavior

Autograd does not differentiate the discrete superblock indices. The shared q/k
projection receives gradients through visited-candidate softmax, the selected-score
fine gate and local attention. Nonzero qkv gradients alone cannot identify the path.
The gate adds `logsigmoid(selected_block_score)` to fine-token attention logits.
It is a differentiable selected-score gate, not a straight-through estimator of
hard routing. Indirect changes to routing through learned shared projections remain
possible; they have not been isolated experimentally.

`F.logsigmoid` avoids the instability of taking log after a sigmoid that underflows
to zero. The existing self-test checks finite gradients under polarized inputs.
Gathering flattened indices avoids expanding the source across queries, which would
make gather backward allocate a much larger source-gradient tensor. These are
implementation choices, not measured end-to-end speed claims.

## Recorded frontier

| L | KV | dense | hier | NSA | KDA | DeltaNet |
|--:|--:|--:|--:|--:|--:|--:|
|128|8|0.997|0.997|0.820|0.999|0.996|
|512|32|1.000|0.983|0.029|0.983|0.988|
|2048|128|0.992|0.994|0.494|0.558|0.193|

All values are selected validation outcomes from M3, not a replicated capability
ranking. KDA L2048 uses LR1e-4; the protocol LR gives0.000. Its restored run has
peak0.558 and last-five mean0.5568. NSA's L128 batch256 control reaches0.981;
L512 has no equivalent control. M2b-to-M3 NSA L2048 changes effective batch and
K_SEL, so it does not isolate either. The same K_SEL also need not yield the same
fine path across these implementations.

M2c retracts the original claim that block sparsity alone caused the observed loss.
Its budget effects do not establish a universal ceiling, monotonic convergence law,
or a matched-compute dominant lever. Seed and batching differences are documented
in the [reconciled records](provenance-reconciliation.md).

L4096 failed including dense attention. Capacity is one hypothesis; it has not been
isolated from optimization/implementation explanations. L8192 is excluded by the
fixed vocabulary8192 and strict vocabulary>length check.

## Future directions and gates

Deeper trees and routing methods are hypotheses, not implemented results. Their
complexity depends on beam width, retained candidates and construction cost. The
current code provides no generic lower bound excluding linear-time attention, and
scoring all supers does not mean gradients flow through all their discrete choices.

Next: small correctness tests; comparable baseline controls; repeated L2048 seeds
including dense; S1=2 recall and cost under one protocol; then targeted M4 ablations.
Runtime/memory measurement is required for practical efficiency claims. Stage1
requires the paper, public repository and documented L2048 result first.
