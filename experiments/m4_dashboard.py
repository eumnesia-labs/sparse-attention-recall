"""Live terminal dashboard for the M4 routing-arms screen: bars, numbers and graphs. $0 to run.

  .venv/bin/python experiments/m4_dashboard.py                 # follow the latest Modal launch
  .venv/bin/python experiments/m4_dashboard.py --launch ID     # a specific launch (smoke-... / screen-...)
  .venv/bin/python experiments/m4_dashboard.py --local DIR     # local run folders (CPU tests), no Modal
  add --once to print a single frame and exit.

Reads what the launcher's containers relay into the Modal Dict `ssa-recall-m4-live` (see
experiments/modal_m4_arms.py). Read-only: Ctrl+C closes the dashboard, never the runs.
Cost shown is an estimate (container wall time x list price); the Modal billing page is authoritative.
"""
import argparse
import glob
import json
import math
import os
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

LIVE = "ssa-recall-m4-live"
COLORS = {"baseline": "bright_white", "router": "cyan", "maxpool": "magenta", "router+maxpool": "yellow"}
PALETTE = ["bright_white", "cyan", "magenta", "yellow", "green", "bright_blue", "red", "bright_magenta"]


def color_of(label, arms):
    """Named arms keep their colors; other run labels (e.g. dense-seed123) get a fixed palette slot."""
    return COLORS.get(label) or PALETTE[arms.index(label) % len(PALETTE)] if label in arms else COLORS.get(label, "white")
PHASE = {"waiting": ("○ waiting", "dim"), "data": ("◐ making data", "yellow"),
         "training": ("● training", "cyan"), "validating": ("◑ validating", "magenta"),
         "held-out test": ("◑ held-out test", "blue"), "complete": ("✔ complete", "bold green"),
         "failed": ("✘ failed", "bold red"), "stale": ("? no signal", "bold red")}
STALE_S = 90


# ---------------------------------------------------------------- data sources
class ModalSource:
    def __init__(self, launch_id=None):
        import modal
        self.d = modal.Dict.from_name(LIVE, create_if_missing=True)
        self.launch_id = launch_id

    def read(self):
        launch = self.d.get("launch")
        if self.launch_id and (not launch or launch.get("id") != self.launch_id):
            launch = {"id": self.launch_id, "arms": list(COLORS), "hourly": 0, "budget": 30, "mode": "?"}
        if not launch:
            return None, {}
        runs = {a: self.d.get(f"{launch['id']}|{a}") for a in launch["arms"]}
        return launch, {a: r for a, r in runs.items() if r}


class LocalSource:
    def __init__(self, root):
        self.root = root

    def read(self):
        runs = {}
        for d in sorted(glob.glob(os.path.join(self.root, "*", "progress.json"))):
            d = os.path.dirname(d)
            st = {}
            for name in ("progress", "result"):
                try:
                    st[name] = json.load(open(os.path.join(d, f"{name}.json")))
                except Exception:
                    pass
            try:
                st["metrics"] = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
            except Exception:
                pass
            p = st.get("progress", {})
            st.update(arm=p.get("arm"), container=d, container_start=p.get("started"), t=p.get("t"))
            runs[p.get("arm")] = st           # latest folder per arm wins (sorted by timestamp)
        started = min([r["container_start"] for r in runs.values() if r.get("container_start")] or [time.time()])
        launch = {"id": os.path.basename(os.path.abspath(self.root)), "mode": "local", "arms": list(runs),
                  "epochs": max([r.get("progress", {}).get("epochs", 1) for r in runs.values()] or [1]),
                  "started": started, "hourly": 0.0, "budget": 42.0, "commit": "local", "dirty": None}
        return launch, runs


# ---------------------------------------------------------------- drawing helpers
def bar(frac, width, color):
    frac = 0.0 if frac is None or not math.isfinite(frac) else max(0.0, min(1.0, frac))
    full = frac * width
    n = int(full)
    part = " ▏▎▍▌▋▊▉"[int((full - n) * 8)].strip() if n < width else ""
    t = Text("█" * n + part, style=color)
    t.append("░" * (width - n - len(part)), style="grey30")
    return t


def hms(s):
    if s is None or not math.isfinite(s):
        return "—"
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


_BR = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def chart(title, series, w, h, xmax, ymin, ymax, yfmt, xlabel):
    """Braille line chart. series: [(color, [(x, y), ...])]; x from 0..xmax."""
    W, H = w * 2, h * 4
    cells = [[0] * w for _ in range(h)]
    cols = [[None] * w for _ in range(h)]
    span_y = (ymax - ymin) or 1.0

    def px(x, y):
        X = round(x / (xmax or 1) * (W - 1))
        Y = round((ymax - y) / span_y * (H - 1))
        return min(W - 1, max(0, X)), min(H - 1, max(0, Y))

    def dot(X, Y, c):
        cells[Y // 4][X // 2] |= _BR[Y % 4][X % 2]
        cols[Y // 4][X // 2] = c

    for c, pts in series:
        P = [px(x, y) for x, y in pts if x is not None and y is not None and math.isfinite(y)]
        if len(P) == 1:
            dot(*P[0], c)
        for (x0, y0), (x1, y1) in zip(P, P[1:]):          # Bresenham
            dx, dy = abs(x1 - x0), -abs(y1 - y0)
            sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
            err = dx + dy
            while True:
                dot(x0, y0, c)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy; x0 += sx
                if e2 <= dx:
                    err += dx; y0 += sy

    out = Text()
    out.append(f"{title}\n", style="bold")
    for r in range(h):
        lab = yfmt(ymax) if r == 0 else yfmt((ymax + ymin) / 2) if r == h // 2 else yfmt(ymin) if r == h - 1 else ""
        out.append(f"{lab:>6} ┤" if lab else "       │", style="grey50")
        for c in range(w):
            m = cells[r][c]
            out.append(chr(0x2800 + m) if m else " ", style=cols[r][c] or "")
        out.append("\n")
    out.append("       └" + "─" * w + "\n", style="grey50")
    xr = f"{xmax:g}"
    out.append(f"        0{xlabel:^{max(0, w - 1 - len(xr))}}{xr}", style="grey50")
    return out


# ---------------------------------------------------------------- the frame
def render(launch, runs, now, err=None, width=120, height=40):
    if not launch:
        return Panel(Text("No launch found yet in the live store.\nStart one with --smoke or --go; "
                          "this view refreshes by itself.", style="yellow"), title="M4 screen · live")
    arms = launch["arms"]
    hourly, budget = launch.get("hourly", 0.0), launch.get("budget", 42.0)
    epochs_cap = launch.get("epochs", 1)

    # ---- cost: container wall time x list price (containers deduped: the smoke shares one)
    boxes = {}
    for r in runs.values():
        k = r.get("container")
        done = r.get("progress", {}).get("phase") in ("complete", "failed")
        end = r.get("t", now) if done else now
        s0 = r.get("container_start") or now
        b = boxes.setdefault(k, [s0, end, done])
        b[0], b[1], b[2] = min(b[0], s0), max(b[1], end), b[2] and done
    for k, b in boxes.items():   # a container with an unfinished run is still billing
        if not b[2]:
            b[1] = now
    spent = sum((b[1] - b[0]) / 3600 * hourly for b in boxes.values())
    all_done = runs and len(runs) == len(arms) and all(
        r.get("progress", {}).get("phase") in ("complete", "failed") for r in runs.values())

    head = Table.grid(padding=(0, 2))
    head.add_row(Text(f"{launch['id']}", style="bold"),
                 Text(f"mode {launch.get('mode')}", style="cyan"),
                 Text(f"code {launch.get('commit', '?')}{' DIRTY' if launch.get('dirty') else ''}",
                      style="red" if launch.get("dirty") else "grey62"),
                 Text(f"elapsed {hms(now - launch.get('started', now))}"),
                 Text("ALL DONE" if all_done else f"live · {time.strftime('%H:%M:%S')}",
                      style="bold green" if all_done else "green"))
    money = Table.grid(padding=(0, 1))
    if hourly:
        money.add_row(Text("spent ≈", style="bold"), Text(f"${spent:6.2f}", style="bold yellow"),
                      bar(spent / budget, 36, "yellow"), Text(f"of ${budget:.0f} budget  "
                      f"({100 * spent / budget:.1f}%)  containers {sum(1 for b in boxes.values() if not b[2])}"
                      f"  ${hourly:.2f}/h"))
    else:
        money.add_row(Text("local run — no Modal cost", style="grey62"))

    # ---- per-arm table
    t = Table(expand=True, header_style="bold", box=None, pad_edge=False)
    for col, kw in [("run", {"no_wrap": True, "min_width": 20}), ("status", {"no_wrap": True}), ("epochs", {"no_wrap": True, "min_width": 10}), ("", {"no_wrap": True}),
                    ("step", {"no_wrap": True, "min_width": 8}), ("", {"no_wrap": True}),
                    ("loss", {"justify": "right"}), ("val acc", {"justify": "right"}),
                    ("best@ep", {"justify": "right", "no_wrap": True}),
                    ("jump", {"justify": "right", "no_wrap": True}),
                    ("TEST@best", {"justify": "right", "no_wrap": True, "min_width": 9}), ("elapsed", {"justify": "right"}),
                    ("ETA", {"justify": "right"})]:
        t.add_column(col, **kw)
    acc_series, loss_series, all_loss = [], [], []
    for a in arms:
        c = color_of(a, arms)
        r = runs.get(a)
        if not r:
            label, sty = PHASE["waiting"]
            t.add_row(Text(a, style=c), Text(label, style=sty), *[Text("—", style="grey42")] * 11)
            continue
        p, res, met = r.get("progress", {}), r.get("result", {}), r.get("metrics", [])
        phase = p.get("phase", "waiting")
        if phase not in ("complete", "failed") and now - (r.get("t") or now) > STALE_S:
            phase = "stale"
        label, sty = PHASE.get(phase, (phase, "white"))
        ep_cap = p.get("epochs", epochs_cap)
        done_ep = p.get("epochs_done", 0)
        steps, step = p.get("steps"), p.get("step", 0)
        accs = [m["valid_accuracy"] for m in met]
        best = max(range(len(accs)), key=lambda i: (accs[i], -i)) if accs else None
        jump = next((m["epoch"] for m in met if m["valid_accuracy"] > 0.30), None)
        el = (r.get("t") or now) - (p.get("started") or r.get("container_start") or now)
        if phase not in ("complete", "failed"):
            el = now - (p.get("started") or now)
        eta = None
        if met and phase not in ("complete", "failed"):
            per = met[-1]["elapsed_s"] / max(1, len(met))
            eta = per * (ep_cap - done_ep)
        tb = res.get("PRIMARY_test_accuracy_at_best_validation")
        t.add_row(
            Text(a, style=f"bold {c}"), Text(label, style=sty),
            bar(done_ep / ep_cap if ep_cap else 0, 10, c), Text(f"{done_ep}/{ep_cap}"),
            bar(step / steps if steps else 0, 8, "grey70"), Text(f"{step}/{steps or '?'}", style="grey62"),
            Text(f"{p['loss']:.3f}" if p.get("loss") is not None else "—"),
            Text(f"{accs[-1]:.3f}" if accs else "—", style="bold" if accs else "grey42"),
            Text(f"{accs[best]:.3f}@{met[best]['epoch']}" if accs else "—"),
            Text(f"ep {jump}" if jump is not None else "—", style="green" if jump is not None else "grey42"),
            Text(f"{tb:.3f}" if tb is not None else "—", style="bold green" if tb is not None else "grey42"),
            Text(hms(el)), Text(hms(eta), style="grey62"))
        if met:
            acc_series.append((c, [(m["epoch"] + 1, m["valid_accuracy"]) for m in met]))
        hist = p.get("loss_hist") or []
        if hist:
            loss_series.append((c, [(x, y) for x, y in hist]))
            all_loss += [y for _, y in hist]

    # ---- graphs, side by side
    cw = max(20, (width - 24) // 2)
    ch = max(3, min(9, height - 20 - max(0, len(arms) - 4)))   # graphs shrink to fit short terminals
    xmax = max([epochs_cap] + [x for _, s in acc_series for x, _ in s])
    g1 = chart("validation accuracy (jump >0.30, stop >0.98)", acc_series, cw, ch,
               xmax, 0.0, 1.0, lambda v: f"{v:.2f}", "epoch")
    lo, hi = (min(all_loss), max(all_loss)) if all_loss else (0.0, 10.0)
    pad = (hi - lo) * 0.05 or 0.5
    lxmax = max([epochs_cap] + [x for _, s in loss_series for x, _ in s])
    g2 = chart("training loss", loss_series, cw, ch, lxmax, max(0.0, lo - pad), hi + pad,
               lambda v: f"{v:.2f}", "epoch")
    graphs = Table.grid(padding=(0, 3))
    graphs.add_row(g1, g2)
    legend = Text("  ")
    for a in arms:
        legend.append("━━ ", style=color_of(a, arms)); legend.append(f"{a}   ")

    foot = Text("Ctrl+C closes this view only; runs continue.  $ is an estimate (modal.com billing is exact).",
                style="grey50")
    if err:
        foot = Text(f"live store unreachable ({err}); retrying …", style="bold red")
    return Panel(Group(head, money, Text(""), t, Text(""), graphs, legend, Text(""), foot),
                 title="[bold]SSA-recall · M4 routing-arms screen · live[/bold]", border_style="cyan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", default=None)
    ap.add_argument("--local", default=None, help="folder containing run folders (no Modal)")
    ap.add_argument("--every", type=float, default=3.0, help="refresh seconds")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    src = LocalSource(a.local) if a.local else ModalSource(a.launch)
    con = Console()
    launch, runs, err = None, {}, None

    def frame():
        nonlocal launch, runs, err
        try:
            launch, runs = src.read()
            err = None
        except Exception as e:           # keep showing the last good state
            err = f"{type(e).__name__}: {str(e)[:60]}"
        return render(launch, runs, time.time(), err, con.width, con.height)

    if a.once:
        con.print(frame())
        return
    try:
        with Live(frame(), console=con, screen=True, refresh_per_second=2) as lv:
            while True:
                time.sleep(a.every)
                lv.update(frame())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
