"""Opt-in MQAR data isolation and deterministic tensor fingerprints.

Import after the pinned zoology checkout is on sys.path. The legacy harness uses
the upstream MQARConfig unchanged; this subclass is only selected explicitly.
"""
import hashlib
import os

import torch
from zoology.data.multiquery_ar import MQARConfig


ISOLATED_CACHE_NAMESPACE = "mqar-isolated-cpu-v1"


class SeededMQARConfig(MQARConfig):
    """Seed each segment's noise without advancing model or CUDA RNG streams."""

    noise_rng_version: str = ISOLATED_CACHE_NAMESPACE

    def build(self, seed):
        # torch.manual_seed also touches CUDA generators. Only the CPU generator
        # is used by upstream MQAR's noise randint, and only that state is saved.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return super().build(seed=seed)


def cache_directory(base, convention):
    if convention == "legacy":
        return base
    if convention == "isolated":
        return os.path.join(base, ISOLATED_CACHE_NAMESPACE)
    raise ValueError(f"unknown data RNG convention: {convention}")


def tensor_fingerprint(tensor):
    """SHA-256 of contiguous CPU tensor bytes, with shape and dtype recorded.

    Byte views support every torch dtype and avoid copying a full MQAR split
    into a Python bytes object. This inspection performs no random operations.
    """
    cpu = tensor.detach().cpu().contiguous()
    byte_array = cpu.reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(byte_array)).hexdigest()
    return {"sha256": digest, "dtype": str(cpu.dtype), "shape": list(cpu.shape)}


def segment_fingerprints(segment):
    return {"inputs": tensor_fingerprint(segment.inputs),
            "targets": tensor_fingerprint(segment.labels)}
