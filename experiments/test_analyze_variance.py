"""CPU-only, standard-library regressions for Amendment 2 rules 1--4.

    python experiments/test_analyze_variance.py

Fixtures are explicitly synthetic and live only in TemporaryDirectory. The archived
screen is read as evidence; this suite never writes to experiments/m4_results/.
"""
import copy
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analyze_variance import analyze


HERE = Path(__file__).resolve().parent
SCREEN = HERE / "m4_results" / "screen-20260930T055659Z.json"
PRIMARY = "PRIMARY_test_accuracy_at_best_validation"


def synthetic_run(score, seed=123, arm="baseline", s1=2, *, validation=0.80):
    """Retain the real legacy schema while replacing all synthetic measurements."""
    template = json.loads(SCREEN.read_text())[0]
    row = copy.deepcopy(template)
    dense = arm == "dense"
    run_id = (f"synthetic-m4-L2048-dense-seed{seed}" if dense else
              f"synthetic-m4-L2048-s{s1}-k4-{arm}-seed{seed}")
    row.update(arm=arm, seed=seed, returncode=0, run_dir=f"/synthetic/{run_id}")
    result, manifest = row["result"], row["manifest"]
    for record in (result, manifest):
        record.update(run_id=run_id, arm=arm, seed=seed,
                      s1=None if dense else s1, protocol_conformant=True)
    result.update(status="complete", epochs_run=60, early_stopped=False,
                  time_capped=False, split_overlap={
                      "heldout_in_train": 0, "heldout_in_validation": 0,
                      "validation_in_train": 0},
                  best_validation_accuracy=validation,
                  best_validation_epoch=0, final_validation_accuracy=validation,
                  test_accuracy_final=0.0 if score < 0.01 else score - 0.01,
                  final_five_validation_mean=validation)
    # Do not retain the old screen jump in the synthetic measurements.
    result["jump_epoch_validation_above_0.30"] = 0 if validation > 0.30 else None
    result[PRIMARY] = score
    manifest["protocol"].update(s1=None if dense else s1,
                                coarse_pairs_per_query=None if dense else 12 + 11 * s1)
    manifest["mixer"] = ("zoology.mixers.attention.MHA" if dense else
                         "hier_nsa.HierSparseAttention")
    manifest["arm_kwargs"] = ({"dropout": 0.1, "num_heads": 4} if dense else {})
    row["metrics"] = [{"epoch": epoch, "valid_accuracy": validation,
                       "valid_loss": 1.0, "lr": 0.00032,
                       "elapsed_s": float(epoch + 1)} for epoch in range(60)]
    return row


class AnalyzeVarianceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-variance-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.serial = 0

    def evaluate(self, rows, *other_paths):
        self.serial += 1
        path = self.folder / f"synthetic-{self.serial}.json"
        path.write_text(json.dumps(rows))
        return analyze([*other_paths, path])

    def baseline(self, scores, validations=None):
        validations = validations or [0.80] * len(scores)
        return [synthetic_run(score, seed, validation=valid)
                for score, seed, valid in zip(scores, (123, 231, 312), validations)]

    def budget(self, s2, s12, dense):
        rows = self.baseline([s2, s2, s2]) + [
            synthetic_run(s12, s1=12), synthetic_run(dense, arm="dense")]
        return self.evaluate(rows)["budget"]

    def test_existing_screen_keeps_exact_primary_scores_and_signed_deficits(self):
        verdict = analyze([SCREEN])
        self.assertEqual(len(verdict["rows"]), 4)
        self.assertTrue(all(row["eligible"] for row in verdict["rows"]))
        self.assertFalse(verdict["baseline_spread"]["complete_seed_set"])
        self.assertIsNone(verdict["baseline_spread"]["range"])
        archived = json.loads(SCREEN.read_text())
        baseline = archived[0]["result"][PRIMARY]
        for row in archived[1:]:
            self.assertAlmostEqual(verdict["m4"]["deficits"][row["arm"]],
                                   row["result"][PRIMARY] - baseline)
        self.assertEqual(verdict["budget"]["classification"], "insufficient_data")

    def test_existing_screen_combines_with_synthetic_variance_without_inventing_results(self):
        verdict = self.evaluate([synthetic_run(0.80, 231),
                                 synthetic_run(0.82, 312),
                                 synthetic_run(0.93, s1=12),
                                 synthetic_run(0.98, arm="dense")], SCREEN)
        spread = verdict["baseline_spread"]
        self.assertTrue(spread["complete_seed_set"])
        self.assertAlmostEqual(spread["range"], 0.02)
        self.assertEqual(spread["band"], "small")
        self.assertEqual(verdict["budget"]["classification"], "recovery_prioritise_curve")

    def test_summary_uses_all_primary_scores_including_non_transition(self):
        scores = [0.007, 0.80, 0.81]
        verdict = self.evaluate(self.baseline(scores, [0.30, 0.80, 0.81]))
        spread = verdict["baseline_spread"]
        self.assertTrue(spread["complete_seed_set"])
        self.assertAlmostEqual(spread["range"], 0.803)
        self.assertAlmostEqual(spread["mean"], statistics.mean(scores))
        self.assertAlmostEqual(spread["median"], 0.80)
        self.assertAlmostEqual(spread["sample_sd"], statistics.stdev(scores))
        self.assertEqual(spread["non_transition_count"], 1)
        self.assertEqual(spread["band"], "comparable_to_m4_deficits")

    def test_transition_requires_validation_strictly_above_point_30(self):
        verdict = self.evaluate(self.baseline([0.78, 0.79, 0.80],
                                              [0.30, 0.30000001, 0.80]))
        self.assertEqual(verdict["baseline_spread"]["non_transition_count"], 1)
        self.assertEqual(verdict["baseline_spread"]["band"], "small")
        self.assertNotIn("strengthens concern", verdict["m4"]["interpretation"].lower())

    def test_all_range_boundaries(self):
        cases = [(0.04999999, "small"), (0.05, "small"),
                 (0.05000001, "material"), (0.14999999, "material"),
                 (0.15, "comparable_to_m4_deficits"),
                 (0.15000001, "comparable_to_m4_deficits")]
        for width, band in cases:
            with self.subTest(width=width):
                verdict = self.evaluate(self.baseline([0.70, 0.72, 0.70 + width]))
                self.assertAlmostEqual(verdict["baseline_spread"]["range"], width)
                self.assertEqual(verdict["baseline_spread"]["band"], band)

    def test_small_all_transition_and_large_variation_preserve_causal_limits(self):
        small = self.evaluate(self.baseline([0.80, 0.81, 0.82]))
        text = small["m4"]["interpretation"].lower()
        self.assertIn("concern", text)
        self.assertIn("prove", text)
        large = self.evaluate(self.baseline([0.65, 0.75, 0.85]))
        text = large["m4"]["interpretation"].lower()
        self.assertIn("paired", text)
        self.assertIn("noise", text)

    def test_recovery_equality_qualifies_even_when_s12_misses_near_dense(self):
        budget = self.budget(0.80, 0.90, 0.99)
        self.assertAlmostEqual(budget["delta_s12_s2"], 0.10)
        self.assertTrue(budget["recovery_s12"])
        self.assertFalse(budget["near_dense_s2"])
        self.assertFalse(budget["near_dense_s12"])
        self.assertEqual(budget["classification"], "recovery_prioritise_curve")
        below = self.budget(0.80, 0.89999999, 0.99)
        self.assertFalse(below["recovery_s12"])
        self.assertEqual(below["classification"], "inconclusive")

    def test_near_dense_equality_is_inclusive_for_both_hier_points(self):
        equal = self.budget(0.90, 0.90, 0.92)
        self.assertTrue(equal["near_dense_s2"])
        self.assertTrue(equal["near_dense_s12"])
        below = self.budget(0.89999999, 0.89999999, 0.92)
        self.assertFalse(below["near_dense_s2"])
        self.assertFalse(below["near_dense_s12"])
        one = self.budget(0.80, 0.90, 0.92)
        self.assertFalse(one["near_dense_s2"])
        self.assertTrue(one["near_dense_s12"])

    def test_both_miss_dense_difference_strictly_below_point_05(self):
        below = self.budget(0.80, 0.84999999, 0.99)
        self.assertEqual(below["classification"], "budget_alone_not_supported")
        exact = self.budget(0.80, 0.85, 0.99)
        self.assertEqual(exact["classification"], "inconclusive")
        above = self.budget(0.80, 0.85000001, 0.99)
        self.assertEqual(above["classification"], "inconclusive")
        reversed_points = self.budget(0.85, 0.80, 0.99)
        self.assertEqual(reversed_points["classification"], "inconclusive")

    def test_both_near_dense_and_recovery_without_dense_are_not_budget_claims(self):
        self.assertEqual(self.budget(0.97, 0.99, 0.99)["classification"], "inconclusive")
        verdict = self.evaluate(self.baseline([0.80] * 3) + [synthetic_run(0.95, s1=12)])
        self.assertEqual(verdict["budget"]["classification"], "insufficient_data")

    def test_ineligible_runs_are_visible_and_excluded_from_seed_summary(self):
        mutations = {
            "incomplete": lambda row: row["result"].update(status="failed"),
            "capped": lambda row: row["result"].update(time_capped=True),
            "overlap": lambda row: row["result"]["split_overlap"].update(heldout_in_train=1),
            "invalid_protocol": lambda row: row["manifest"].update(protocol_conformant=False),
            "changed_recipe": lambda row: row["manifest"]["protocol"].update(seq_len=512),
            "truncated": lambda row: row.update(metrics=row["metrics"][:-1]),
            "warm_cache": lambda row: row["manifest"]["data_cache_hit_at_start"].update(train=True),
            "overridden": lambda row: row["manifest"].update(overrides={"epochs": 1}),
            "wrong_selection": lambda row: row["manifest"]["protocol"].update(selection_rule="last checkpoint"),
            "identity_conflict": lambda row: row["result"].update(seed=123),
        }
        for label, mutate in mutations.items():
            with self.subTest(reason=label):
                rows = self.baseline([0.80, 0.81, 0.99])
                mutate(rows[-1])
                verdict = self.evaluate(rows)
                excluded = [row for row in verdict["rows"] if not row["eligible"]]
                self.assertEqual(len(excluded), 1)
                self.assertTrue(excluded[0]["reasons"])
                self.assertFalse(verdict["baseline_spread"]["complete_seed_set"])
                self.assertIsNone(verdict["baseline_spread"]["range"])
                self.assertIsNone(verdict["baseline_spread"]["sample_sd"])

    def test_real_early_stop_qualifies_but_false_claim_does_not(self):
        rows = self.baseline([0.80, 0.81, 0.99])
        stopped = rows[-1]
        stopped["metrics"] = stopped["metrics"][:30]
        stopped["metrics"][-1]["valid_accuracy"] = 0.981
        stopped["result"].update(epochs_run=30, early_stopped=True,
                                  best_validation_accuracy=0.981)
        self.assertTrue(self.evaluate(rows)["baseline_spread"]["complete_seed_set"])
        stopped["metrics"][-1]["valid_accuracy"] = 0.98
        stopped["result"]["best_validation_accuracy"] = 0.98
        self.assertFalse(self.evaluate(rows)["baseline_spread"]["complete_seed_set"])

    def test_training_past_early_stop_threshold_is_not_protocol_complete(self):
        fixture = synthetic_run(0.81)
        fixture["metrics"][0]["valid_accuracy"] = 0.981
        fixture["result"].update(best_validation_accuracy=0.981, early_stopped=False)
        row = self.evaluate([fixture])["rows"][0]
        self.assertFalse(row["eligible"])
        self.assertIn("early_stop_rule_mismatch", row["reasons"])

    def test_nan_epochs_are_excluded_and_json_serializable(self):
        fixture = synthetic_run(0.81)
        fixture["result"]["epochs_run"] = float("nan")
        verdict = self.evaluate([fixture])
        row = verdict["rows"][0]
        self.assertFalse(row["eligible"])
        self.assertIn("incomplete_epoch_records", row["reasons"])
        self.assertIsNone(row["epochs_trained"])
        json.dumps(verdict, allow_nan=False)

    def test_missing_seeds_are_graceful_and_do_not_turn_screen_into_variance(self):
        verdict = self.evaluate([synthetic_run(0.80)])
        spread = verdict["baseline_spread"]
        self.assertFalse(spread["complete_seed_set"])
        for name in ("range", "mean", "median", "sample_sd"):
            self.assertIsNone(spread[name], name)
        self.assertEqual(spread["band"], "insufficient_data")
        self.assertEqual(verdict["budget"]["classification"], "insufficient_data")
        empty = self.evaluate([])
        self.assertFalse(empty["baseline_spread"]["complete_seed_set"])

    def test_duplicate_seed_cannot_silently_choose_the_better_score(self):
        verdict = self.evaluate(self.baseline([0.80, 0.81, 0.82]) + [synthetic_run(0.99)])
        self.assertFalse(verdict["baseline_spread"]["complete_seed_set"])
        self.assertIsNone(verdict["baseline_spread"]["range"])

    def test_caveats_preserve_training_pairing_coverage_and_test_reuse_limits(self):
        verdict = self.evaluate(self.baseline([0.80] * 3) + [
            synthetic_run(0.90, s1=12), synthetic_run(0.98, arm="dense")])
        caveats = " ".join(verdict["caveats"]).lower()
        self.assertIn("noise", caveats)
        self.assertIn("coverage", caveats)
        self.assertIn("123", caveats)
        self.assertIn("three", caveats)

    def test_each_score_curve_final_score_epochs_and_jump_are_reported(self):
        fixture = synthetic_run(0.81)
        fixture["metrics"][0]["valid_accuracy"] = 0.30
        fixture["metrics"][1]["valid_accuracy"] = 0.30000001
        row = self.evaluate([fixture])["rows"][0]
        self.assertEqual(row["primary"], 0.81)
        self.assertEqual(row["final_score"], 0.80)
        self.assertEqual(row["epochs_trained"], 60)
        self.assertEqual(row["jump_epoch"], 1)
        self.assertEqual(row["validation_curve"],
                         [metric["valid_accuracy"] for metric in fixture["metrics"]])

    def test_invalid_primary_is_excluded_and_json_remains_valid(self):
        fixture = synthetic_run(float("nan"))
        verdict = self.evaluate([fixture])
        self.assertFalse(verdict["rows"][0]["eligible"])
        self.assertIn("invalid_primary_score", verdict["rows"][0]["reasons"])
        self.assertIsNone(verdict["rows"][0]["primary"])
        self.assertIsNone(verdict["rows"][0]["final_score"])
        json.dumps(verdict, allow_nan=False)

    def test_cli_prints_table_and_writes_matching_json_to_temp_output(self):
        path = self.folder / "synthetic-cli.json"
        path.write_text(json.dumps(self.baseline([0.80, 0.81, 0.82])))
        output = self.folder / "verdict.json"
        command = [sys.executable, str(HERE / "analyze_variance.py"),
                   str(path), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("primary", completed.stdout)
        self.assertIn("Baseline spread:", completed.stdout)
        self.assertEqual(json.loads(output.read_text()), analyze([path]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
