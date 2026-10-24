"""Print a portable fresh-run command from a recorded M5 manifest; never execute it."""
import argparse
import json
from pathlib import Path
import shlex
import sys

from m4_routing_arms import ARMS, DENSE, PROTOCOL


def command(manifest, out_root, cache_dir, device="cuda"):
    m = manifest
    if m.get("task") != "mqar" or m.get("arm") not in ("baseline", "dense"):
        raise ValueError("Only recorded M5 MQAR baseline/dense recipes are supported")
    precision = m.get("precision")
    if precision not in ("fp32", "tf32"):
        raise ValueError("Missing or unsupported precision")
    if m.get("overrides") != ({"precision": "tf32"} if precision == "tf32" else {}):
        raise ValueError("Unsupported protocol overrides")
    for key, value in PROTOCOL.items():
        if key != "s1" and m.get("protocol", {}).get(key) != value:
            raise ValueError(f"Unsupported protocol value: {key}")
    dense = m["arm"] == "dense"
    if m.get("arm_kwargs") != (DENSE if dense else ARMS[m["arm"]]):
        raise ValueError("Unsupported mixer configuration")
    if m.get("max_hours") != 0 or m.get("data_rng") not in ("legacy", "isolated"):
        raise ValueError("Unsupported time cap or data RNG")
    if type(m.get("seed")) is not int:
        raise ValueError("Missing integer seed")
    if not dense and (m.get("s1") not in (2, 4, 6, 12) or m.get("mask_mode") != "sentinel"):
        raise ValueError("Unsupported selector configuration")
    args = [sys.executable, str(Path(__file__).with_name("m4_routing_arms.py").resolve()),
            "--arm", m["arm"], "--seed", str(m["seed"]), "--task", "mqar",
            "--precision", precision, "--data-rng", m["data_rng"],
            "--out-root", str(Path(out_root).resolve()),
            "--cache-dir", str(Path(cache_dir).resolve()), "--device", device]
    if not dense:
        args += ["--s1", str(m["s1"]), "--mask-mode", "sentinel"]
        if m.get("router_telemetry"):
            args.append("--router-telemetry")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--cache-dir", required=True, help="Use a fresh directory, especially for legacy RNG")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = parser.parse_args()
    try:
        args = command(json.loads(a.manifest.read_text()), a.out_root, a.cache_dir, a.device)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(shlex.join(args))


if __name__ == "__main__":
    main()
