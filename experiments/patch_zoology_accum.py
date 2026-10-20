"""Patch a pinned zoology checkout (1ad20d1) with gradient accumulation.

Why: hier-L2048 does not learn at batch 8 + lr 3.2e-4 (flat 0.005 at ep44 = 110k updates,
session 2026-07-14) — the only working LR was tuned at batch 16, and L512+ is razor
LR-sensitive (M2b repair sweep). But batch 16 OOMs the 14.5GiB T4 for every budget point
above the s2-k4 baseline. Gradient accumulation gives the intended effective batch of 16
(same effective batch, same LR, same 1250 optimizer steps/epoch) at micro-batch memory.

Behavior: GRAD_ACCUM env (default 1 = stock zoology). Loss is scaled by 1/accum; the
optimizer steps every `accum` micro-batches; a trailing partial group still steps.
The per-epoch CosineAnnealingLR schedule is untouched (it never sees micro-steps).

Also (added 2026-08-02, after s12-k32 diverged mid-phase-transition — train loss went
2.05 -> nan in ONE step at ep13 and 47 epochs trained a dead model): GRAD_CLIP env
(default 0 = off, stock dynamics) applies clip_grad_norm_ before every optimizer step;
and an always-on NaN guard raises after 20 consecutive non-finite train losses so a
diverged run dies in seconds, not hours.

Usage: python patch_zoology_accum.py /path/to/zoology   (idempotent; hard-fails if the
pinned anchors are missing — bump the pin deliberately, never patch blind).
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "zoology" / "train.py"
src = path.read_text()

if "GRAD_CLIP" in src:   # marker of the CURRENT patch (an accum-only tree must not be skipped)
    print("already patched — nothing to do")
    sys.exit(0)

REPLACEMENTS = [
    (
        "import argparse\nimport random\n",
        "import argparse\nimport os\nimport random\n",
    ),
    (
        """        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
""",
        """        accum = int(os.environ.get("GRAD_ACCUM", "1"))
        clip = float(os.environ.get("GRAD_CLIP", "0") or 0)
        _nan_streak = 0
        self.optimizer.zero_grad()
        for _step, (inputs, targets, slices) in enumerate(iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
""",
    ),
    (
        """            loss.backward()
            self.optimizer.step()
""",
        """            if not torch.isfinite(loss):
                _nan_streak += 1
                if _nan_streak >= 20:
                    raise RuntimeError(
                        f"[nan-guard] train loss non-finite for {_nan_streak} consecutive "
                        f"steps (epoch {epoch_idx}, step {_step}) -- divergence, aborting")
            else:
                _nan_streak = 0
            (loss / accum).backward()
            if (_step + 1) % accum == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
""",
    ),
    (
        """            self.logger.log({"train/loss": loss.item(), "epoch": epoch_idx})

    def test(self, epoch_idx: int):""",
        """            self.logger.log({"train/loss": loss.item(), "epoch": epoch_idx})
        if (_step + 1) % accum != 0:   # trailing partial group
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def test(self, epoch_idx: int):""",
    ),
]

for old, new in REPLACEMENTS:
    if src.count(old) != 1:
        sys.exit(f"ANCHOR NOT FOUND (count={src.count(old)}) — zoology moved off the pin?\n{old!r}")
    src = src.replace(old, new)

import ast
ast.parse(src)
path.write_text(src)
print(f"patched {path} (GRAD_ACCUM + GRAD_CLIP + nan-guard)")
