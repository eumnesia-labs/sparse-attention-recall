# Portable M5 reproduction

`experiments/m4_routing_arms.py` now supports the recorded
`--task mqar --precision tf32` recipe. TF32 follows the historical runner:
`torch.set_float32_matmul_precision("high")` and cuDNN TF32 enabled, before model
construction. The default `fp32` path preserves historical PyTorch defaults;
it is not a promise to disable every backend's TF32 setting.

Install the dependencies and pinned, patched Zoology using the existing setup
instructions (`experiments/setup_cpu_env.sh` for the CPU environment; CUDA needs
a compatible PyTorch build). Activate that environment before generating a command.

Choose a `manifest.json` under `experiments/m5_results/`, including its `followup/`
and `round2ef/` directories. Generate a fresh-run command:

```sh
python experiments/reproduce_m5.py PATH/TO/manifest.json \
  --out-root /tmp/ssa-reproduction-results \
  --cache-dir /tmp/ssa-fresh-cache-RUN-ID
```

This prints a command only. Review it before running it on approved hardware.
Use a fresh, unique cache directory per run: legacy RNG results depend on cold-cache
generation order. The fixed model-before-training-data order is preserved.
Use CUDA for the original training setting; CPU is for checks, not a numerically
equivalent replacement. No new training or GPU spending is performed by this helper.

The translator validates the supported fixed protocol and retains the arm, seed,
selector budget, precision, data RNG, mask and telemetry. It deliberately does not
execute stored `env.argv`, cloud synchronization, historical machine paths, or
historical commit overrides. Fresh runs record the actual current code provenance.
Historical manifests remain immutable evidence. The stopped run's manifest describes
a fresh full recipe, not a recreation of its interruption or a resumed checkpoint.

This is a portable recipe implementation, not a claim of bitwise equivalence to the
historical cloud runner. Hardware, PyTorch/CUDA versions and nondeterminism may change
outcomes. Compare split fingerprints before interpreting accuracy differences.
The original cloud runner (queue, checkpoint-resume and cloud transport) is not part of this
release, because it also contains unrelated unpublished work; this runner supports fresh runs only.

The legacy certification helper still labels isolated data as a draft profile and
TF32 as an override, matching the stored records. `protocol_conformant=false` is not
retroactively changed; that field is not certification against the complete M5 protocol.

Validation without training dependencies:

```sh
python -m unittest discover -s experiments -p test_reproduce_m5.py
python -m unittest discover -s experiments -p test_analyze_variance.py
```

These checks cover argument handling and recipe translation only. No fresh training run or CUDA
smoke check has been made with this runner, so numerical training equivalence is not established.
