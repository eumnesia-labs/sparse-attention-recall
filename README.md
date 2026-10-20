# Does a Cheap Block Selector Keep Recall?

Code, pre-registered protocols, run records and paper source for
**"Does a Cheap Block Selector Keep Recall? A Pre-Registered Budget Study on MQAR"**
(Anuj Dev Singh, Independent Researcher).

We train small models on multi-query associative recall (MQAR) at length 2048 and vary $S_1$, the number of
super-blocks a two-level block selector keeps per query, against dense attention trained on byte-identical splits.

| Result (held-out recall, $L{=}2048$, $K{=}4$) | Value |
|---|---|
| Pilot seed 123: $S_1{=}2$ / 4 / 6 / 12 (flat) | 0.705 / 0.884 / 0.962 / 0.850 |
| $S_1{=}6$ on three untouched seeds | 0.937 / 0.945 / 0.945 |
| Dense − $S_1{=}6$ on the one reserved seed where dense learned | 0.055 (pre-registered bar: 0.02) |
| Recall when the selector chose the key's block (11 learned models) | 0.978–1.000 |

Every run is reported, including runs that never learned.

## Layout

```
src/hier_nsa.py              hierarchical block selector (the model under test)
src/hier_telemetry.py        router telemetry helper
experiments/m4_routing_arms.py   training harness (zoology-based); m4_data.py: isolated, seeded MQAR data
experiments/diagnose_routing.py  selection-vs-reading diagnostic; modal_diagnose_m5.py runs it on Modal
experiments/m5_results/      M5-profile runs (main results): per-run manifest, metrics, result, train log
  selection_diagnostic/      replay of all 12 selector checkpoints (paper Section 4)
  followup/, round2ef/       S1=6 confirmation runs, dense float32 reruns
experiments/m4_results/      M4-profile runs (routing repairs, paper Section 5)
audits/verify-routing-2026-10-01/  M4-profile routing probes (appendix selection table, K probe)
docs/protocol-*.md           protocols, frozen before the runs they govern
paper/                       LaTeX source, figures and the scripts that regenerate them
```

## Reproduce

```bash
bash experiments/setup_cpu_env.sh          # pinned zoology (1ad20d1) + CPU venv, for tests and analysis
export ZOOLOGY_DIR=$PWD/zoology
.venv-cpu/bin/python experiments/test_hier_masks.py
python3 paper/figures/make_figures.py      # figures from the run records
python3 paper/make_tables.py               # appendix all-runs table
python3 experiments/m5_results/summarize.py
```

Training a model needs one GPU (we used an NVIDIA L40S) and the command recorded in that run's `manifest.json`.
`experiments/test_hier_arms.py` compares against a commit of the original development repository; outside it, that
check reports itself as skipped.

## Notes on this release

- This repository is a curated snapshot of a private research repository (snapshot of its commit `694055a`).
  Earlier exploratory experiments (paper Appendix "Earlier exploratory results") are not included.
- Cloud storage locations and machine identifiers in run records are replaced by placeholders such as
  `<checkpoint-bucket>`. Model checkpoints are not included; seeds, split hashes and code hashes in each manifest
  identify every run.

## License

MIT. See [LICENSE](LICENSE).
