"""CPU tests for the S1=2 routing experiment arms in src/hier_nsa.py.

Run from the repo root (needs torch; no GPU, no zoology):
    PYTHONPATH=src python experiments/test_hier_arms.py

What each test proves:
  T1 baseline byte-identity -- default flags reproduce the pre-arms model (ba09be2) exactly:
     identical outputs, parameter gradients and input gradients (torch.equal, not allclose),
     at a small config, the saturated S1=12 config, and the real L2048 S1=2 config.
     Skipped only if git or that commit is unavailable (e.g. a container without .git).
  T2 router arm keeps routing decisions -- same superblock and fine-block indices as the
     baseline with identical weights; only the output weighting changes.
  T3 router arm gives the router a gradient -- baseline level-1 scores get none; the router arm
     gets non-zero gradient on selected live supers and exactly zero on unselected supers.
  T4 max pooling -- helper matches a manual padded max; mean path unchanged.
  T5 causality -- every arm, block/superblock boundary positions, at L256 and the real L2048.
  T6 polarized scores -- every arm keeps finite forward/backward past the fp32 sigmoid cliff.
  T7 L2048 training smoke -- a 2-layer LM, real S1=2/K=4/g=11 config, a few AdamW steps per arm:
     flags reach every mixer, first-step losses differ between arms (each arm really changes the
     computation), finite losses and gradients; reports CPU seconds/step.
"""
import importlib.util, math, os, subprocess, sys, tempfile, time
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import hier_nsa as H
from hier_nsa import HierSparseAttention, NEG, _block_pool

BASE_COMMIT = "ba09be2"
ARMS = {"baseline": {}, "router": {"router_gate": True}, "maxpool": {"superblock_pool": "max"},
        "router+maxpool": {"router_gate": True, "superblock_pool": "max"}}
SMALL = dict(num_heads=4, compress_block_size=16, selection_block_size=16,
             num_selected_blocks=4, superblock_group=4, num_selected_superblocks=2)   # L256: nsup=4
REAL = dict(num_heads=4, sliding_window_size=32, compress_block_size=16, selection_block_size=16,
            num_selected_blocks=4, superblock_group=11, num_selected_superblocks=2)  # L2048: nsup=12
FAILS = []

def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not cond: FAILS.append(name)

def load_base():
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                              check=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:src/hier_nsa.py"], capture_output=True,
                             text=True, check=True, cwd=root).stdout
    except Exception as e:
        return None, str(e).splitlines()[0] if str(e) else type(e).__name__
    path = os.path.join(tempfile.mkdtemp(), "hier_nsa_base.py"); open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("hier_nsa_base", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, None

def t1_baseline_identity():
    print("T1 default flags are byte-identical to", BASE_COMMIT)
    base, err = load_base()
    if base is None:
        print(f"  SKIP  cannot load {BASE_COMMIT}: {err}"); return
    for label, D, L, B, cfg in (("small L256", 64, 256, 2, SMALL),
                                ("saturated S1=12 L256", 64, 256, 2, dict(SMALL, num_selected_superblocks=12)),
                                ("real L2048 S1=2", 128, 2048, 1, REAL)):
        torch.manual_seed(1)
        old = base.HierSparseAttention(D, **cfg)
        new = HierSparseAttention(D, **cfg); new.load_state_dict(old.state_dict())
        x = torch.randn(B, L, D); up = torch.randn(B, L, D)
        xo, xn = x.clone().requires_grad_(), x.clone().requires_grad_()
        yo, yn = old(xo), new(xn)
        (yo * up).sum().backward(); (yn * up).sum().backward()
        same = torch.equal(yo, yn) and torch.equal(xo.grad, xn.grad) and all(
            torch.equal(po.grad, pn.grad) for po, pn in zip(old.parameters(), new.parameters()))
        check(f"{label}: outputs + all gradients exactly equal", same)

def t2_t3_router():
    print("T2/T3 router arm: same routing decisions, and a gradient to the router")
    for label, D, L, B, cfg in (("L256", 64, 256, 2, SMALL), ("L2048", 128, 2048, 1, REAL)):
        torch.manual_seed(2)
        base = HierSparseAttention(D, **cfg); rout = HierSparseAttention(D, router_gate=True, **cfg)
        rout.load_state_dict(base.state_dict())
        base.keep_routing = rout.keep_routing = True
        x = torch.randn(B, L, D)
        yb, yr = base(x), rout(x)
        check(f"{label}: superblock + fine-block indices identical",
              torch.equal(base.last_sup_idx, rout.last_sup_idx) and torch.equal(base.last_fin_blk, rout.last_fin_blk))
        check(f"{label}: router arm output differs (gate is active)", not torch.equal(yb, yr))
        yb.sum().backward(); yr.sum().backward()
        gb, gr, ss = base.last_router_scores.grad, rout.last_router_scores.grad, rout.last_router_scores
        check(f"{label}: baseline router scores get NO gradient", gb is None or gb.abs().sum() == 0,
              "grad is None" if gb is None else f"|grad|={gb.abs().sum():.3g}")
        sel = torch.zeros_like(ss, dtype=torch.bool).scatter_(-1, rout.last_sup_idx, True)
        live = ss.detach() > NEG / 2
        on = sel & live
        frac = (gr[on] != 0).float().mean().item()
        check(f"{label}: router arm gradient on selected live supers", gr is not None and gr[on].abs().sum() > 0,
              f"non-zero on {100 * frac:.1f}% of selected live entries")
        check(f"{label}: exactly zero gradient on unselected supers", gr is not None and gr[~sel].abs().sum() == 0)

def t4_maxpool():
    print("T4 max pooling helper")
    torch.manual_seed(4)
    x = torch.randn(2, 3, 13, 5)   # 13 not divisible by 4 -> padding path
    xp = F.pad(x, (0, 0, 0, 3)).view(2, 3, 4, 4, 5)
    got_max, nb = _block_pool(x, 4, "max"); got_mean, _ = _block_pool(x, 4)
    check("max matches manual padded max", nb == 4 and torch.equal(got_max, xp.amax(3)))
    check("mean path unchanged", torch.equal(got_mean, xp.mean(3)))
    try:
        HierSparseAttention(64, superblock_pool="median"); check("rejects unknown pool", False)
    except AssertionError:
        check("rejects unknown pool", True)

def t5_causality():
    print("T5 causality, all arms (perturb one position; earlier outputs must not move)")
    for label, D, L, B, cfg, ps in (
            ("L256", 64, 256, 2, SMALL, (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 200, 254)),
            ("L2048", 128, 2048, 1, REAL, (100, 175, 176, 177, 1000, 2046))):
        for arm, kw in ARMS.items():
            torch.manual_seed(5); m = HierSparseAttention(D, **kw, **cfg); x = torch.randn(B, L, D)
            with torch.no_grad():
                y = m(x); worst = 0.0
                for p in ps:
                    x2 = x.clone(); x2[:, p] += 5.0
                    worst = max(worst, (m(x2)[:, :p] - y[:, :p]).abs().max().item())
            check(f"{label} {arm}", worst < 1e-5, f"max leak {worst:.2e}")

def t6_polarized():
    print("T6 polarized scores stay finite, all arms")
    for arm, kw in ARMS.items():
        torch.manual_seed(6)
        m = HierSparseAttention(64, **kw, **dict(SMALL, num_selected_blocks=8, num_selected_superblocks=3))
        with torch.no_grad(): m.qkv.weight.mul_(30.0)
        x = (torch.randn(2, 256, 64) * 3).requires_grad_()
        y = m(x); y.sum().backward()
        ok = torch.isfinite(y).all() and torch.isfinite(x.grad).all() and all(
            p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        check(arm, bool(ok))

def t7_smoke():
    print("T7 L2048 training smoke: 2-layer LM, real S1=2/K=4/g=11 config, B=2, 4 AdamW steps")
    V, D, L, B = 8192, 128, 2048, 2
    first = {}
    for arm, kw in ARMS.items():
        torch.manual_seed(7)
        class Blk(nn.Module):
            def __init__(s):
                super().__init__(); s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
                s.mix = HierSparseAttention(D, **kw, **REAL)
                s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
            def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))
        emb, blks, norm, head = nn.Embedding(V, D), nn.ModuleList([Blk(), Blk()]), nn.LayerNorm(D), nn.Linear(D, V)
        flags_ok = all(b.mix.router_gate == kw.get("router_gate", False) and
                       b.mix.superblock_pool == kw.get("superblock_pool", "mean") for b in blks)
        params = [p for mod in (emb, blks, norm, head) for p in mod.parameters()]
        opt = torch.optim.AdamW(params, lr=3.2e-4)
        x, y = torch.randint(0, V, (B, L)), torch.randint(0, V, (B, L))
        losses, t0 = [], None
        for i in range(4):
            if i == 1: t0 = time.time()
            h = emb(x)
            for b in blks: h = b(h)
            loss = F.cross_entropy(head(norm(h)).reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            opt.step(); losses.append(loss.item())
        dt = (time.time() - t0) / 3
        first[arm] = losses[0]
        check(f"{arm}", flags_ok and all(math.isfinite(l) for l in losses) and finite,
              f"loss {losses[0]:.6f}->{losses[-1]:.6f}  {dt:.2f} s/step (B=2, CPU)")
    check("every arm changes the computation (4 distinct first-step losses)",
          len(set(first.values())) == len(first), "  ".join(f"{a}={v:.6f}" for a, v in first.items()))

if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    for t in (t1_baseline_identity, t2_t3_router, t4_maxpool, t5_causality, t6_polarized, t7_smoke):
        t()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
