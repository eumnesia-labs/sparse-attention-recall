"""CPU failing-before/passing-after regressions for opt-in boolean masks.

    python experiments/test_hier_masks.py

The extreme-score fixtures reproduce archived review defects. Half-precision
checks concern CPU forward/backward only, not GPU mixed-precision training.
"""
import math
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import hier_nsa as H

torch.set_num_threads(2)
ARMS = [{}, {"router_gate": True}, {"superblock_pool": "max"},
        {"router_gate": True, "superblock_pool": "max"}]


def simple_model(mode="sentinel", *, router=True, branch=0, dtype=torch.float64, telemetry=False):
    """Constant query, scalar input key/value, and one enabled output branch."""
    model = H.HierSparseAttention(3, num_heads=1, sliding_window_size=1,
                                  compress_block_size=2, selection_block_size=2,
                                  num_selected_blocks=4, superblock_group=2,
                                  num_selected_superblocks=2, router_gate=router,
                                  mask_mode=mode, router_telemetry=telemetry).to(dtype)
    with torch.no_grad():
        model.qkv.weight.zero_()
        model.qkv.bias.zero_()
        model.qkv.bias[0] = math.sqrt(3)
        model.qkv.weight[3, 1] = 1
        model.qkv.weight[6, 2] = 1
        model.gate.weight.zero_()
        model.gate.bias.fill_(-1000)
        model.gate.bias[branch] = 1000
        model.proj.weight.copy_(torch.eye(3, dtype=dtype))
        model.proj.bias.zero_()
    model.keep_routing = True
    return model


def finite_backward(test, model, x):
    x = x.clone().requires_grad_()
    output = model(x)
    test.assertTrue(torch.isfinite(output).all())
    output.square().sum().backward()
    test.assertTrue(torch.isfinite(x.grad).all(), "non-finite input gradient")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            test.assertTrue(torch.isfinite(parameter.grad).all(), name)
    return output


class BooleanMaskTests(unittest.TestCase):
    def test_float32_valid_fine_branch_false_zero_is_corrected(self):
        x = torch.zeros(1, 12, 3)
        x[:, :, 1] = -2e8
        x[:, :4, 2] = 1
        old = simple_model("sentinel", branch=1, dtype=torch.float32)
        corrected = simple_model("bool", branch=1, dtype=torch.float32)
        with torch.no_grad():
            old_output, corrected_output = old(x), corrected(x)
        self.assertEqual(old_output[0, 4, 0].item(), 0)
        self.assertEqual(corrected_output[0, 4, 0].item(), 1)
        self.assertTrue(torch.equal(old.last_fin_blk, corrected.last_fin_blk))
        finite_backward(self, corrected, x)

    def test_float64_finite_sentinel_future_leak_is_corrected(self):
        x = torch.zeros(1, 12, 3, dtype=torch.float64)
        x[:, :, 1] = -499999999.9
        old, corrected = simple_model(), simple_model("bool")
        with torch.no_grad():
            old_output, corrected_output = old(x), corrected(x)
            future_parent = next(index for index in old.last_sup_idx[0, 0, 4].tolist() if index > 0)
            future_position = future_parent * 4 + 1
            changed = x.clone()
            changed[0, future_position, 2] = 10
            before = (old(changed)[0, 4] - old_output[0, 4]).abs().max().item()
            after = (corrected(changed)[0, 4] - corrected_output[0, 4]).abs().max().item()
        self.assertGreater(before, 0.1)
        self.assertEqual(after, 0)
        finite_backward(self, corrected, x)

    def test_boolean_softmax_all_invalid_is_zero_with_finite_zero_gradients(self):
        logits = torch.tensor([[-999999999.8, -999999999.8, -1e9, -1e9],
                               [-1e9, 1e30, float("nan"), float("inf")]],
                              dtype=torch.float64, requires_grad=True)
        live = torch.tensor([[True, True, False, False], [False, False, False, False]])
        weights = H._masked_softmax(logits, live)
        self.assertTrue(torch.equal(weights, torch.tensor([[0.5, 0.5, 0, 0], [0, 0, 0, 0]],
                                                         dtype=torch.float64)))
        (weights * torch.tensor([1., 2., 3., 4.])).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(4, dtype=torch.float64)))
        self.assertTrue(torch.equal(logits.grad[0, 2:], torch.zeros(2, dtype=torch.float64)))
        self.assertGreater(logits.grad[0, :2].abs().sum().item(), 0)

    def test_boolean_max_padding_preserves_negative_real_max_and_default(self):
        x = torch.tensor([[[[-5., -2.], [-4., -3.], [-6., -7.]]]])
        default, count = H._block_pool(x, 2, "max")
        corrected, corrected_count = H._block_pool(x, 2, "max", mask_mode="bool")
        self.assertEqual(count, corrected_count)
        self.assertEqual(count, 2)
        self.assertTrue(torch.equal(default[0, 0, -1], torch.zeros(2)))
        self.assertTrue(torch.equal(corrected[0, 0, -1], x[0, 0, -1]))
        mean, _ = H._block_pool(x, 2)
        bool_mean, _ = H._block_pool(x, 2, mask_mode="bool")
        self.assertTrue(torch.equal(mean, bool_mean))

    def test_all_invalid_coarse_and_fine_early_rows_stay_zero(self):
        x = torch.zeros(1, 12, 3, dtype=torch.float64)
        x[:, :, 2] = 7
        for branch in (0, 1):
            with self.subTest(branch=branch):
                model = simple_model("bool", branch=branch)
                output = finite_backward(self, model, x)
                self.assertTrue(torch.equal(output[0, :2], torch.zeros(2, 3, dtype=torch.float64)))
                self.assertGreater(output[0, 4, 0].item(), 0)

    def test_causal_boundaries_and_incomplete_padding_all_arms(self):
        for length, s1 in ((31, 2), (65, 2), (65, 12)):
            for arm in ARMS:
                with self.subTest(length=length, s1=s1, arm=arm):
                    torch.manual_seed(15)
                    model = H.HierSparseAttention(16, num_heads=2, sliding_window_size=8,
                                                   compress_block_size=4, selection_block_size=4,
                                                   num_selected_blocks=4, superblock_group=3,
                                                   num_selected_superblocks=s1,
                                                   mask_mode="bool", **arm)
                    x = torch.randn(1, length, 16)
                    with torch.no_grad():
                        reference = model(x)
                        for position in (1, 3, 4, 5, 11, 12, 13, length - 1):
                            changed = x.clone()
                            changed[0, position] += 5
                            self.assertTrue(torch.allclose(reference[:, :position], model(changed)[:, :position],
                                                           atol=1e-6, rtol=0), position)
                    finite_backward(self, model, x)

    def test_ordinary_scores_keep_routing_counts_and_close_outputs(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(7)
                cfg = dict(num_heads=4, compress_block_size=16, selection_block_size=16,
                           num_selected_blocks=4, superblock_group=4, num_selected_superblocks=2)
                old = H.HierSparseAttention(64, **cfg, **arm)
                corrected = H.HierSparseAttention(64, mask_mode="bool", **cfg, **arm)
                corrected.load_state_dict(old.state_dict())
                old.keep_routing = corrected.keep_routing = True
                x = torch.randn(1, 256, 64)
                old_output, corrected_output = old(x), corrected(x)
                self.assertTrue(torch.equal(old.last_sup_idx, corrected.last_sup_idx))
                self.assertTrue(torch.equal(old.last_fin_blk, corrected.last_fin_blk))
                self.assertEqual(old.coarse_pairs_per_query, corrected.coarse_pairs_per_query)
                self.assertEqual(old.dense_pairs_per_query, corrected.dense_pairs_per_query)
                self.assertTrue(torch.allclose(old_output, corrected_output, atol=1e-6, rtol=0))
                finite_backward(self, corrected, x)

    def test_cpu_half_default_overflow_and_opt_in_finite_forward_backward(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(20)
                cfg = dict(num_heads=2, compress_block_size=4, selection_block_size=4,
                           num_selected_blocks=4, superblock_group=3, num_selected_superblocks=2)
                old = H.HierSparseAttention(16, **cfg, **arm).half()
                corrected = H.HierSparseAttention(16, mask_mode="bool", **cfg, **arm).half()
                corrected.load_state_dict(old.state_dict())
                x = torch.randn(1, 32, 16, dtype=torch.float16)
                with self.assertRaisesRegex(RuntimeError, "overflow"):
                    old(x)
                output = finite_backward(self, corrected, x)
                self.assertEqual(output.dtype, torch.float16)

    def test_boolean_scores_remain_float32_under_cpu_half_autocast(self):
        query = torch.full((1, 1, 1, 2), 300., dtype=torch.float16)
        keys = query.clone()
        with torch.autocast("cpu", dtype=torch.float16):
            old = torch.einsum("bhld,bhsd->bhls", query, keys)
            corrected = H._boolean_scores("bhld,bhsd->bhls", query, keys, 1.0)
        self.assertTrue(torch.isinf(old).all())
        self.assertEqual(corrected.dtype, torch.float32)
        self.assertEqual(corrected.item(), 180000.)

    def test_cpu_autocast_boolean_forward_backward_and_router_score_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            for arm in ARMS:
                with self.subTest(dtype=dtype, arm=arm):
                    torch.manual_seed(21)
                    model = H.HierSparseAttention(16, num_heads=2, compress_block_size=4,
                                                   selection_block_size=4, num_selected_blocks=4,
                                                   superblock_group=3, num_selected_superblocks=2,
                                                   mask_mode="bool", **arm)
                    model.keep_routing = True
                    x = torch.randn(1, 32, 16, requires_grad=True)
                    with torch.autocast("cpu", dtype=dtype):
                        output = model(x)
                        loss = output.float().square().sum()
                    loss.backward()
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(model.last_router_scores.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(model.last_router_scores[0, :, -1, :2]).all())
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_mask_mode_is_rejected_and_default_stays_sentinel(self):
        self.assertEqual(H.HierSparseAttention(16).mask_mode, "sentinel")
        with self.assertRaises((ValueError, AssertionError)):
            H.HierSparseAttention(16, mask_mode="unknown")


class RouterTelemetryTests(unittest.TestCase):
    def model(self, enabled=True, mode="bool"):
        return H.HierSparseAttention(16, num_heads=2, compress_block_size=8,
                                     selection_block_size=8, num_selected_blocks=4,
                                     superblock_group=2, num_selected_superblocks=2,
                                     router_gate=True, mask_mode=mode, router_telemetry=enabled)

    def test_telemetry_changes_no_outputs_gradients_or_rng_all_arms_and_modes(self):
        for mode in ("sentinel", "bool"):
            for arm in ARMS:
                with self.subTest(mode=mode, arm=arm):
                    torch.manual_seed(22)
                    cfg = dict(num_heads=2, compress_block_size=8, selection_block_size=8,
                               num_selected_blocks=4, superblock_group=2, num_selected_superblocks=2,
                               mask_mode=mode, **arm)
                    off = H.HierSparseAttention(16, **cfg)
                    on = H.HierSparseAttention(16, router_telemetry=True, **cfg)
                    on.load_state_dict(off.state_dict())
                    x = torch.randn(2, 64, 16)
                    upstream = torch.randn_like(x)
                    xo, xn = x.clone().requires_grad_(), x.clone().requires_grad_()
                    state = torch.get_rng_state().clone()
                    yo = off(xo)
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    yn = on(xn)
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    (yo * upstream).sum().backward()
                    (yn * upstream).sum().backward()
                    self.assertTrue(torch.equal(yo, yn))
                    self.assertTrue(torch.equal(xo.grad, xn.grad))
                    for old_parameter, new_parameter in zip(off.parameters(), on.parameters()):
                        self.assertTrue(torch.equal(old_parameter.grad, new_parameter.grad))
                    self.assertIsNone(off.consume_router_telemetry())
                    self.assertIsNotNone(on.consume_router_telemetry())

    def test_populations_exclude_invalid_scores_and_quantiles_are_bounded_reset(self):
        torch.manual_seed(23)
        model = self.model()
        x = torch.randn(2, 64, 16)
        for _ in range(3):
            model(x)
        snapshot = model.consume_router_telemetry()
        self.assertEqual(snapshot["forwards"], 3)
        self.assertEqual(snapshot["sample_cap_per_metric"], 512)
        self.assertEqual(snapshot["method"], "deterministic-power-of-two-strided-stream-v1")
        metrics = snapshot["metrics"]
        scores = metrics["level1_valid_scores"]
        # Four supers at L64; each query's fully-past parent count is floor(t/16).
        expected = 3 * 2 * 2 * sum(position // 16 for position in range(64))
        self.assertEqual(scores["population_count"], expected)
        self.assertEqual(scores["finite_count"], expected)
        self.assertEqual(scores["nonfinite_count"], 0)
        for record in metrics.values():
            self.assertLessEqual(record["sample_count"], 512)
            self.assertLessEqual(record["sample_count"], record["finite_count"])
            if record["sample_count"]:
                self.assertEqual(set(record["quantiles"]), {"0", "0.01", "0.1", "0.5", "0.9", "0.99", "1"})
                quantiles = [record["quantiles"][key] for key in ("0", "0.01", "0.1", "0.5", "0.9", "0.99", "1")]
                self.assertEqual(quantiles, sorted(quantiles))
                self.assertTrue(all(math.isfinite(value) for value in quantiles))
        self.assertIsNone(model.consume_router_telemetry())
        model(x)
        self.assertEqual(model.consume_router_telemetry()["forwards"], 1)

    def test_eval_forwards_are_not_mixed_into_training_telemetry(self):
        model = self.model()
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 64, 16))
        self.assertIsNone(model.consume_router_telemetry())
        model.train()
        model(torch.randn(1, 64, 16))
        self.assertEqual(model.consume_router_telemetry()["forwards"], 1)

    def test_telemetry_exposes_threshold_crossing_on_false_zero_fixture(self):
        model = simple_model("sentinel", branch=1, dtype=torch.float32, telemetry=True)
        x = torch.zeros(1, 12, 3)
        x[:, :, 1] = -2e8
        x[:, :4, 2] = 1
        model(x)
        snapshot = model.consume_router_telemetry()
        metrics = snapshot["metrics"]
        self.assertGreater(metrics["fine_valid_logits"]["counts"]["below_NEG_half"], 0)
        for record in metrics.values():
            for name, count in record["counts"].items():
                self.assertLessEqual(count, record["finite_count"])
                if record["finite_count"]:
                    self.assertAlmostEqual(record["fractions"][name], count / record["finite_count"])

    def test_telemetry_nonfinite_and_invalid_observations_have_explicit_denominators(self):
        from hier_telemetry import RouterTelemetry
        collector = RouterTelemetry()
        values = torch.tensor([-1e9, -5e8, 0., 20., float("nan"), float("inf")])
        live = torch.tensor([True, False, True, True, True, True])
        state = torch.get_rng_state().clone()
        collector.forwards = 1
        collector.add("scores", values, live)
        metric = collector.consume()["metrics"]["scores"]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(metric["population_count"], 5)
        self.assertEqual(metric["finite_count"], 3)
        self.assertEqual(metric["nonfinite_count"], 2)
        self.assertEqual(metric["sample_count"], 3)
        self.assertEqual(metric["counts"]["near_NEG_half"], 0)
        self.assertEqual(metric["counts"]["below_NEG"], 1)
        self.assertAlmostEqual(metric["fractions"]["below_NEG"], 1 / 3)
        self.assertEqual(metric["quantiles"]["0"], -1e9)
        self.assertEqual(metric["quantiles"]["0.5"], 0)
        self.assertEqual(metric["quantiles"]["1"], 20)

    def test_float64_quantile_interpolation_stays_finite_for_extreme_endpoints(self):
        from hier_telemetry import RouterTelemetry
        collector = RouterTelemetry()
        collector.forwards = 1
        collector.add("scores", torch.tensor([-1e308, 1e308], dtype=torch.float64))
        quantiles = collector.consume()["metrics"]["scores"]["quantiles"]
        self.assertTrue(all(math.isfinite(value) for value in quantiles.values()))
        self.assertEqual(quantiles["0"], -1e308)
        self.assertEqual(quantiles["0.5"], 0)
        self.assertEqual(quantiles["1"], 1e308)


if __name__ == "__main__":
    unittest.main(verbosity=2)
