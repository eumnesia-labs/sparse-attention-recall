"""HierSparseAttention — M2b: content-based sparse attention with a SUB-QUADRATIC selector.

Drop-in zoology sequence-mixer (interface matches zoology.mixers.*: __init__(d_model, **kw),
forward(u: (B,L,D)) -> (B,L,D)). Inspired by NSA's three-branch layout, with a
hierarchical coarse search. Compression, positions, normalization and fine paths
also differ; see docs/implementation-comparison.md. This is not a routing-only ablation.

WHY THIS IS THE M2b CONTRIBUTION (read before touching the math):
  M2a showed NSA rides the attention recall ceiling on MQAR, BUT NSA's selector is quadratic.
  The quadratic cost is NOT the top-k. It is the COARSE branch: every query attends over ALL
  n/compress_block_size compressed blocks  ->  O(n^2 / B). That same dense score feeds top-k.
  So to make *selection* cheap you must make the *coarse stage* cheap.

  HierSparseAttention replaces the dense coarse attention with a 2-level tree:
    level-1: super-block summaries   (n / (cb*g) of them)   <- scored densely (small)
    level-0: compressed blocks       (only the g members of the top-s1 super-blocks)  <- gathered
  Per query the coarse stage scores  nsup + min(s1, nsup)*g  nodes instead of nblk = n/cb.
  With g ~ sqrt(nblk) -> nsup ~ sqrt(nblk) and s1*g ~ sqrt(nblk) -> O(sqrt(n)) per query
  -> O(n^1.5) total. THE HARNESS MUST SET g = round(sqrt(nblk)) PER SEQUENCE LENGTH; a fixed g
  stays O(n^2). The fine blocks for exact attention fall out of the descent.

  => provides selected-token attention with sub-quadratic selector counts.
     Recall and cost must be measured at the same configuration before a joint claim.

GRADIENT: top-s / top-k indices are discrete; autograd does not differentiate the
choice of super-block indices. Gradients reach shared q/k projections through
softmax over visited candidate scores, a differentiable selected-score gate on the
fine branch, and the local attention branch. This is not a straight-through
estimator of hard routing. The shared-qkv gradient self-test is only a smoke check,
not proof that the discrete routing decision itself receives gradients.

EXPERIMENT ARMS (2026-09-29, branch exp/s1-2-routing-arms). Both are opt-in. With the defaults
(router_gate=False, superblock_pool="mean") forward and backward are byte-identical to ba09be2;
experiments/test_hier_arms.py checks that exactly.
  router_gate=True  -- the level-1 superblock scores reach the loss. logsigmoid(selected super
      score) is added (a) to the coarse-branch logits of that super's member blocks and (b) to the
      fine-branch token logits of every fine block it contains. Routing DECISIONS are unchanged:
      top-s1 and top-k indices are still taken on the ungated scores. Softmax is shift-invariant,
      so the signal is relative: it exists only where the attended blocks span >= 2 supers.
      Unselected supers still receive no gradient. Supers not yet fully past get gate 0, so
      early-position behavior matches the baseline.
  superblock_pool="max" -- superblock summaries are the elementwise max of member block keys
      instead of their mean, so one strongly matching block is not averaged away. A cheap proxy,
      not max_j(q.k_j) (that would be the full block scan). The zero-padded last superblock is
      never fully past, so it is always masked and padding cannot affect selection.

OPT-IN NUMERICAL PATH (2026-10-01): mask_mode="bool" uses causal validity tensors,
empty-row-safe masked softmax and -inf max-pool padding. It retains the existing
candidate coverage, including fully-past children of incomplete parent placeholders.
Half/bfloat16 scores use fp32 with autocast disabled; value reads keep their dtype.
mask_mode="sentinel" remains the default and retains the historical computation.
router_telemetry=True collects detached training observations with bounded deterministic
stream samples. consume_router_telemetry() reports exact counts, approximate quantiles
and their sampling method, then resets; validation forwards are excluded.

CAUSALITY: coarse uses only *fully-past* blocks (block_end <= t) so block-mean summaries never
include future tokens. The current/recent region is covered by the sliding window. Clean & exact.

INSTRUMENTATION: after each forward, self.coarse_pairs_per_query (= nsup + min(s1, nsup)*g) and
self.dense_pairs_per_query (= nblk, NSA block proxy excluding its extra memory key) let M3 plot score-tensor pair counts (including masked/padded candidates)
vs length. NOTE: this prototype scores via gather+mask, so wall-clock is NOT the kernel a fused
sparse impl would hit; report the scored-pair COUNT as the cost metric and treat wall-clock as
indicative only ("complexity moved, not removed" cuts both ways — say so in the paper).

Validation: existing _selftest checks shape, two causal perturbations, shared-projection
gradients and finite gradients under polarized inputs. The 2026-09-25 correction
checks count reporting only; it does not rerun Torch or certify routing correctness.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hier_telemetry import RouterTelemetry

NEG = -1e9


def _masked_softmax(logits, live):
    """Normalize explicitly valid entries; an empty row is exactly zero.

    Reset empty rows before softmax so its backward never sees all -inf logits.
    Valid finite scores remain eligible regardless of their magnitude.
    """
    masked = logits.masked_fill(~live, -torch.inf)
    masked = torch.where(live.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return masked.softmax(-1).masked_fill(~live, 0.0)


def _boolean_scores(equation, query, keys, scale):
    """Keep opt-in half/bfloat16 score arithmetic in fp32 under autocast too."""
    dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    with torch.autocast(device_type=query.device.type, enabled=False):
        return torch.einsum(equation, query.to(dtype), keys.to(dtype)) * scale


def _block_pool(x, block, reduce="mean", mask_mode="sentinel"):
    """Pool blocks; padding is zero except opt-in bool/max padding is -inf."""
    B, h, N, d = x.shape
    pad = (-N) % block
    if pad:
        if mask_mode == "bool" and reduce == "max":
            x = F.pad(x, (0, 0, 0, pad), value=-torch.inf)
        else:
            x = F.pad(x, (0, 0, 0, pad))
    nb = x.shape[2] // block
    x = x.view(B, h, nb, block, d)
    return (x.amax(3) if reduce == "max" else x.mean(3)), nb


def _gather_blocks(src, blk_idx):
    """src:(B,h,N,d), blk_idx:(B,h,L,M) block ids -> (B,h,L,M,d). Invalid ids must be pre-clamped.

    Gathers on flattened (L*M) indices against src directly. Do NOT expand src to
    (B,h,L,N,d) first: gather's backward allocates a zeros tensor of the FULL input
    shape, which is O(L*N*d) — a 32GiB alloc at L2048/batch16 (OOM'd the T4)."""
    B, h, N, d = src.shape
    L, M = blk_idx.shape[2], blk_idx.shape[3]
    idx = blk_idx.clamp(0, N - 1).reshape(B, h, L * M, 1).expand(B, h, L * M, d)
    return src.gather(2, idx).reshape(B, h, L, M, d)


class HierSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, sliding_window_size=32,
                 compress_block_size=16, selection_block_size=16,
                 num_selected_blocks=4, superblock_group=4,
                 num_selected_superblocks=2, causal=True,
                 router_gate=False, superblock_pool="mean",
                 mask_mode="sentinel", router_telemetry=False, **kw):
        super().__init__()
        # prototype ties the coarse leaf = compress = selection block (one block granularity)
        assert compress_block_size == selection_block_size, "keep compress==selection block size"
        self.h = num_heads
        self.w = sliding_window_size
        self.cb = compress_block_size
        self.k_sel = num_selected_blocks
        self.g = superblock_group              # harness sets ~ round(sqrt(nblk)); fixed g => O(n^2)
        self.s1 = num_selected_superblocks
        self.causal = causal
        assert superblock_pool in ("mean", "max"), f"superblock_pool must be mean|max, got {superblock_pool!r}"
        self.router_gate = bool(router_gate)      # experiment arm: gradient to level-1 scores
        self.superblock_pool = superblock_pool    # experiment arm: summary of member blocks
        assert mask_mode in ("sentinel", "bool"), f"mask_mode must be sentinel|bool, got {mask_mode!r}"
        self.mask_mode = mask_mode
        self.router_telemetry = bool(router_telemetry)
        self._router_telemetry = RouterTelemetry(neg=NEG) if self.router_telemetry else None
        self.keep_routing = False                 # tests only: keep router scores (grad retained) + indices
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.gate = nn.Linear(d_model, 3 * num_heads)
        self.proj = nn.Linear(d_model, d_model)
        self.coarse_pairs_per_query = 0
        self.dense_pairs_per_query = 0          # = nblk, NSA block proxy; full coarse path adds one memory key

    def consume_router_telemetry(self):
        """Return the training epoch's detached bounded summary and reset it."""
        return self._router_telemetry.consume() if self._router_telemetry is not None else None

    def forward(self, u):
        B, L, D = u.shape
        h, hd = self.h, D // self.h
        cb, g, s1, k_sel = self.cb, self.g, self.s1, self.k_sel
        dev = u.device
        bool_mask = self.mask_mode == "bool"
        telemetry = self._router_telemetry if self.router_telemetry and self.training else None
        scale = 1.0 / math.sqrt(hd)
        q, k, v = self.qkv(u).reshape(B, L, 3, h, hd).permute(2, 0, 3, 1, 4)  # each (B,h,L,hd)
        # The opt-in path uses fp32 score arithmetic for half inputs. -1e9 cannot
        # be represented in fp16, and half dot products can overflow before scaling.
        score_q = q.float() if bool_mask and q.dtype in (torch.float16, torch.bfloat16) else q
        t = torch.arange(L, device=dev)                                       # query positions

        # ---- summaries over fully-past blocks / super-blocks ----
        bk, nblk = _block_pool(k, cb)                 # (B,h,nblk,hd)
        bv, _ = _block_pool(v, cb)
        sk, nsup = _block_pool(bk, g, self.superblock_pool, mask_mode=self.mask_mode)  # (B,h,nsup,hd)
        blk_end = (torch.arange(nblk, device=dev) + 1) * cb          # first t for which block is past
        sup_end = (torch.arange(nsup, device=dev) + 1) * g * cb
        self.coarse_pairs_per_query = nsup + min(s1, nsup) * g
        self.dense_pairs_per_query = nblk

        # ---- level 1: score query vs super summaries, pick top-s1 past super-blocks ----
        if bool_mask:
            ss = _boolean_scores("bhld,bhsd->bhls", score_q, sk, scale)
        else:
            ss = torch.einsum("bhld,bhsd->bhls", q, sk) * scale          # (B,h,L,nsup)
        if bool_mask or telemetry is not None:
            sup_live = (sup_end[None, None, None, :] <= t[None, None, :, None]).expand(B, h, L, nsup)
        ss = ss.masked_fill(~sup_live, -torch.inf) if bool_mask else ss.masked_fill(
            sup_end[None, None, None, :] > t[None, None, :, None], NEG)
        s1e = min(s1, nsup)
        sup_top = ss.topk(s1e, dim=-1)
        sup_idx = sup_top.indices                                    # (B,h,L,s1e)
        if bool_mask or telemetry is not None:
            selected_sup_live = torch.gather(sup_live, -1, sup_idx)

        # ---- level 0: gather member blocks of selected supers, score them ----
        member = (sup_idx.unsqueeze(-1) * g + torch.arange(g, device=dev)).reshape(B, h, L, s1e * g)
        valid = member < nblk                                        # padded supers overrun nblk
        cbk = _gather_blocks(bk, member)                             # (B,h,L,s1e*g,hd)
        cbv = _gather_blocks(bv, member)
        bs = (_boolean_scores("bhld,bhlmd->bhlm", score_q, cbk, scale) if bool_mask else
              torch.einsum("bhld,bhlmd->bhlm", q, cbk) * scale)        # (B,h,L,s1e*g)
        be = blk_end[member.clamp(0, nblk - 1)]                      # block_end per candidate
        if bool_mask or telemetry is not None:
            # Keep the established S1=12 coverage: a masked incomplete parent
            # placeholder can still contain a fully-past, eligible child.
            candidate_live = (be <= t[None, None, :, None]) & valid
        bs = bs.masked_fill(~candidate_live, -torch.inf) if bool_mask else bs.masked_fill(
            (be > t[None, None, :, None]) | ~valid, NEG)

        if self.router_gate:
            # log-space gate per selected super; supers not yet fully past (score NEG) get gate 0
            if bool_mask:
                safe_sup_scores = sup_top.values.masked_fill(~selected_sup_live, 0.0)
                sup_gate = F.logsigmoid(safe_sup_scores).masked_fill(~selected_sup_live, 0.0)
            else:
                sup_gate = torch.where(sup_top.values > NEG / 2, F.logsigmoid(sup_top.values),
                                       torch.zeros_like(sup_top.values))         # (B,h,L,s1e)
            bs_coarse = bs + sup_gate.repeat_interleave(g, dim=-1)           # candidate c -> super c//g
        else:
            bs_coarse = bs

        # coarse output: softmax over visited candidate blocks (this is the gradient path to scores).
        # zero the branch where NO block is fully-past, else the all-masked softmax leaks future tokens.
        if bool_mask:
            coarse = torch.einsum("bhlm,bhlmd->bhld", _masked_softmax(bs_coarse, candidate_live).to(cbv.dtype), cbv)
        else:
            coarse_ok = (bs.max(-1, keepdim=True).values > NEG / 2).float()
            coarse = torch.einsum("bhlm,bhlmd->bhld", bs_coarse.softmax(-1).nan_to_num(0.0), cbv) * coarse_ok

        # ---- fine selection: top-k candidate blocks -> exact attention over their raw tokens ----
        ke = min(k_sel, s1e * g)
        fin = bs.topk(ke, dim=-1)
        fin_blk = torch.gather(member, -1, fin.indices)              # (B,h,L,ke) chosen block ids
        if bool_mask or telemetry is not None:
            selected_block_live = torch.gather(candidate_live, -1, fin.indices)
        # differentiable selected-score gate, in log space via logsigmoid — NEVER sigmoid().log():
        # sigmoid underflows to exactly 0 below x~-88 (fp32) and log(0).backward() mints
        # 0*inf = nan on VALID candidates (killed s12-k32 twice at ep13, mid-transition score
        # polarization). logsigmoid's backward is sigmoid(-x), finite for every finite score.
        if bool_mask:
            log_gate = F.logsigmoid(fin.values.masked_fill(~selected_block_live, 0.0)).unsqueeze(-1)
        else:
            log_gate = F.logsigmoid(fin.values).unsqueeze(-1).clamp_min(NEG)
        if self.router_gate:   # add the parent super's gate to each fine block (selection unchanged)
            log_gate = log_gate + torch.gather(sup_gate, -1, fin.indices // g).unsqueeze(-1)
        if self.keep_routing:
            if ss.requires_grad:
                ss.retain_grad()
            self.last_router_scores, self.last_sup_idx, self.last_fin_blk = ss, sup_idx, fin_blk
        tok = (fin_blk.unsqueeze(-1) * cb + torch.arange(cb, device=dev)).reshape(B, h, L, ke * cb)
        tok_ok = tok < L
        kf = _gather_blocks(k, tok)                                  # gather tokens, treat tok as ids
        vf = _gather_blocks(v, tok)
        fs = (_boolean_scores("bhld,bhlmd->bhlm", score_q, kf, scale) if bool_mask else
              torch.einsum("bhld,bhlmd->bhlm", q, kf) * scale)         # (B,h,L,ke*cb)
        if bool_mask or telemetry is not None:
            token_live = (tok <= t[None, None, :, None]) & tok_ok
            token_live = token_live & selected_block_live.repeat_interleave(cb, dim=-1)
        fs = fs.masked_fill(~token_live, -torch.inf) if bool_mask else fs.masked_fill(
            (tok > t[None, None, :, None]) | ~tok_ok, NEG)
        # broadcast the per-block gate (already log-space) across its cb tokens
        fs = fs + log_gate.expand(B, h, L, ke, cb).reshape(B, h, L, ke * cb)
        if bool_mask:
            fine = torch.einsum("bhlm,bhlmd->bhld", _masked_softmax(fs, token_live).to(vf.dtype), vf)
        else:
            fine_ok = (fs.max(-1, keepdim=True).values > NEG / 2).float()
            fine = torch.einsum("bhlm,bhlmd->bhld", fs.softmax(-1).nan_to_num(0.0), vf) * fine_ok

        if telemetry is not None:
            telemetry.forwards += 1
            telemetry.count("selected_parent_total", sup_top.values.numel())
            telemetry.count("selected_parent_valid", selected_sup_live.sum().item())
            telemetry.count("selected_parent_invalid", (~selected_sup_live).sum().item())
            telemetry.count("selected_leaf_total", fin.values.numel())
            telemetry.count("selected_leaf_valid", selected_block_live.sum().item())
            telemetry.count("selected_leaf_invalid", (~selected_block_live).sum().item())
            telemetry.add("level1_valid_scores", ss, sup_live)
            telemetry.add("selected_parent_valid_scores", sup_top.values, selected_sup_live)
            telemetry.add("selected_leaf_valid_scores", fin.values, selected_block_live)
            telemetry.add("leaf_log_gates", log_gate.squeeze(-1), selected_block_live)
            telemetry.add("leaf_gate_values", log_gate.detach().squeeze(-1).exp(), selected_block_live)
            telemetry.add("fine_valid_logits", fs, token_live)
            for name, scores, live in (("level1_valid_row_max", ss, sup_live),
                                       ("candidate_valid_row_max", bs, candidate_live),
                                       ("fine_valid_row_max", fs, token_live)):
                telemetry.add(name, scores.detach().masked_fill(~live, -torch.inf).amax(-1), live.any(-1))
            if self.router_gate:
                telemetry.add("parent_log_gates", sup_gate, selected_sup_live)
                telemetry.add("parent_gate_values", sup_gate.detach().exp(), selected_sup_live)

        # ---- sliding window (banded, causal) ----
        off = torch.arange(self.w, device=dev)
        wpos = t[None, :] - off[:, None]                             # (w,L) positions t,t-1,...
        wpos_t = wpos.t()                                           # (L,w)
        widx = wpos_t.clamp(0, L - 1)[None, None].expand(B, h, L, self.w)
        kw = _gather_blocks(k, widx)
        vw = _gather_blocks(v, widx)
        if bool_mask:
            wscore = _boolean_scores("bhld,bhlmd->bhlm", score_q, kw, scale)
            window_live = (wpos_t[None, None] >= 0).expand(B, h, L, self.w)
            win = torch.einsum("bhlm,bhlmd->bhld", _masked_softmax(wscore, window_live).to(vw.dtype), vw)
        else:
            wscore = torch.einsum("bhld,bhlmd->bhlm", q, kw) * scale
            wscore = wscore.masked_fill(wpos_t[None, None] < 0, NEG)
            win = torch.einsum("bhlm,bhlmd->bhld", wscore.softmax(-1).nan_to_num(0.0), vw)

        # ---- gated combine (per-head sigmoid weights, NSA-style, not normalized) ----
        gates = torch.sigmoid(self.gate(u)).reshape(B, L, h, 3).permute(0, 2, 1, 3)  # (B,h,L,3)
        out = (gates[..., 0:1] * coarse + gates[..., 1:2] * fine + gates[..., 2:3] * win)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


def _selftest():
    torch.manual_seed(0)
    B, L, D = 2, 128, 64
    m = HierSparseAttention(D, num_heads=4, compress_block_size=16, selection_block_size=16,
                            num_selected_blocks=4, superblock_group=3, num_selected_superblocks=2)
    x = torch.randn(B, L, D, requires_grad=True)
    y = m(x)
    assert y.shape == (B, L, D), y.shape
    assert torch.isfinite(y).all(), "NaN/inf in output"

    # causality: perturbing position p must not change outputs at positions < p.
    # check an early p (catches future-leak via block summaries) and the midpoint.
    for p in (3, L // 2):
        x2 = x.detach().clone(); x2[:, p] += 5.0
        y2 = m(x2)
        assert torch.allclose(y[:, :p], y2[:, :p], atol=1e-5), f"causality violated at p={p}"

    # shared-projection gradient smoke check; does not test discrete routing gradients
    y.sum().backward()
    assert m.qkv.weight.grad is not None and m.qkv.weight.grad.abs().sum() > 0, "no grad to qkv"
    assert m.gate.weight.grad is not None, "no grad to gate"

    # regression (s12-k32 ep13 crash class): polarized scores must not mint nans in backward.
    # Push scores past the fp32 sigmoid-underflow cliff (|score| >> 88) on valid selected
    # blocks, then demand finite grads everywhere. sigmoid().log() dies here; logsigmoid must not.
    m2 = HierSparseAttention(D, num_heads=4, compress_block_size=16, selection_block_size=16,
                             num_selected_blocks=8, superblock_group=3, num_selected_superblocks=3)
    with torch.no_grad():
        m2.qkv.weight.mul_(30.0)
    x3 = (torch.randn(B, L, D) * 3).requires_grad_()
    y3 = m2(x3)
    assert torch.isfinite(y3).all(), "nan/inf in forward under polarized scores"
    y3.sum().backward()
    for n_, p_ in m2.named_parameters():
        assert p_.grad is None or torch.isfinite(p_.grad).all(), f"non-finite grad in {n_}"
    assert torch.isfinite(x3.grad).all(), "non-finite grad to input under polarized scores"

    # experiment arms: same shape / causality / finite-gradient smoke checks
    for arm in ({"router_gate": True}, {"superblock_pool": "max"},
                {"router_gate": True, "superblock_pool": "max"}):
        ma = HierSparseAttention(D, num_heads=4, compress_block_size=16, selection_block_size=16,
                                 num_selected_blocks=4, superblock_group=3,
                                 num_selected_superblocks=2, **arm)
        ya = ma(x.detach())
        assert ya.shape == (B, L, D) and torch.isfinite(ya).all(), f"bad output for {arm}"
        for p in (3, L // 2):
            x2 = x.detach().clone(); x2[:, p] += 5.0
            assert torch.allclose(ya[:, :p], ma(x2)[:, :p], atol=1e-5), f"causality violated p={p} {arm}"
        ya.sum().backward()
        assert all(p_.grad is None or torch.isfinite(p_.grad).all() for p_ in ma.parameters()), \
            f"non-finite grad for {arm}"

    print(f"OK  out={tuple(y.shape)}  coarse_pairs/q={m.coarse_pairs_per_query} "
          f"vs NSA-dense/q={m.dense_pairs_per_query}  (saving "
          f"{m.dense_pairs_per_query / m.coarse_pairs_per_query:.2f}x)")


if __name__ == "__main__":
    _selftest()
