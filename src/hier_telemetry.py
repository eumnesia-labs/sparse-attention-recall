"""Bounded deterministic observations of router scores; no autograd or RNG calls."""
import math

import torch


class RouterTelemetry:
    """Exact population counts, approximate quantiles on a fixed-size stream sample.

    Retain every 2**k-th finite observation, increasing k when needed to keep
    at most sample_cap values per metric. This samples across the whole epoch;
    retained observations and their global indices are the only growing state.
    """

    def __init__(self, sample_cap=512, neg=-1e9):
        self.sample_cap = sample_cap
        self.neg = neg
        self.reset()

    def reset(self):
        self.forwards = 0
        self.metrics = {}
        self.counts = {}

    def count(self, name, amount):
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def add(self, name, tensor, valid=None):
        flat = tensor.detach().reshape(-1)
        if valid is not None:
            flat = flat[valid.expand_as(tensor).reshape(-1)]
        population = flat.numel()
        values = flat[torch.isfinite(flat)]
        finite = values.numel()
        state = self.metrics.setdefault(name, {
            "population_count": 0, "finite_count": 0, "nonfinite_count": 0,
            "stride": 1, "samples": [], "counts": {}})
        prior_finite = state["finite_count"]
        state["population_count"] += population
        state["finite_count"] += finite
        state["nonfinite_count"] += population - finite
        tests = {
            "below_NEG_half": values <= self.neg / 2,
            "below_NEG": values <= self.neg,
            "near_NEG_half": (values - self.neg / 2).abs() <= abs(self.neg) * .01,
            "near_NEG": (values - self.neg).abs() <= abs(self.neg) * .01,
            "positive_saturation": values >= 20,
            "log_gate_near_one": (values >= -1e-6) & (values <= 0),
            "log_gate_near_zero": values <= -20,
        }
        for key, selected in tests.items():
            state["counts"][key] = state["counts"].get(key, 0) + int(selected.sum().item())
        stride = state["stride"]
        while math.ceil(state["finite_count"] / stride) > self.sample_cap:
            stride *= 2
        if stride != state["stride"]:
            state["samples"] = [(i, v) for i, v in state["samples"] if i % stride == 0]
            state["stride"] = stride
        first = (-prior_finite) % stride
        if first < finite:
            indices = torch.arange(first, finite, stride, device=values.device)
            observed = values[indices].to(device="cpu", dtype=torch.float64).tolist()
            state["samples"].extend((prior_finite + first + j * stride, value)
                                    for j, value in enumerate(observed))
        assert len(state["samples"]) <= self.sample_cap

    def consume(self):
        if not self.forwards:
            return None
        metrics = {}
        for name, state in self.metrics.items():
            sample = sorted(value for _, value in state["samples"])

            def quantile(q):
                if not sample:
                    return None
                position = q * (len(sample) - 1)
                low, high = math.floor(position), math.ceil(position)
                fraction = position - low
                return sample[low] * (1 - fraction) + sample[high] * fraction

            metrics[name] = {
                "population_count": state["population_count"],
                "finite_count": state["finite_count"],
                "nonfinite_count": state["nonfinite_count"],
                "sample_count": len(sample), "sample_stride": state["stride"],
                "quantiles": {str(q): quantile(q) for q in (0, .01, .1, .5, .9, .99, 1)},
                "counts": dict(state["counts"]),
                "fractions": {key: value / state["finite_count"] if state["finite_count"] else None
                              for key, value in state["counts"].items()},
            }
        result = {
            "method": "deterministic-power-of-two-strided-stream-v1",
            "quantiles_are_approximate": True,
            "fraction_denominator": "finite valid observations; nonfinite counts reported separately",
            "sample_cap_per_metric": self.sample_cap, "forwards": self.forwards,
            "counts": dict(self.counts), "metrics": metrics,
            "thresholds": {"NEG": self.neg, "NEG_half": self.neg / 2,
                           "below_comparison": "<=", "near_absolute_distance": abs(self.neg) * .01,
                           "positive_score_saturation": 20, "log_gate_near_one": -1e-6,
                           "log_gate_near_zero": -20},
        }
        self.reset()
        return result
