"""CPU telemetry integration and static launcher option checks; never import Modal."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_m4_harness import run
import torch

ROOT = Path(__file__).resolve().parents[1]


def launcher_helpers():
    tree = ast.parse((ROOT / 'experiments/modal_m4_arms.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('configure_specs', 'harness_options')]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pure launcher helpers>', 'exec'), scope)
    return scope['configure_specs'], scope['harness_options']


class TelemetryIntegrationTests(unittest.TestCase):
    def test_launcher_defaults_and_explicit_forwarding_without_modal(self):
        configure, options = launcher_helpers()
        specs = [dict(arm='baseline', s1=2, seed=123, label='baseline-s2-seed123'),
                 dict(arm='dense', s1=None, seed=123, label='dense-seed123')]
        self.assertEqual(configure(specs), specs)
        self.assertEqual(options(specs[0]), [])
        selected = configure(specs, 'isolated', 'bool', True)
        self.assertEqual(specs[0]['label'], 'baseline-s2-seed123')
        self.assertEqual(options(selected[0]), ['--data-rng', 'isolated', '--mask-mode', 'bool', '--router-telemetry'])
        self.assertEqual(options(selected[1]), ['--data-rng', 'isolated'])
        with self.assertRaises(ValueError): configure(specs, 'unseeded')

    def test_epoch_telemetry_records_and_does_not_change_cpu_training(self):
        with tempfile.TemporaryDirectory(prefix='m4-cpu-telemetry-') as temp:
            root = Path(temp)
            runs = []
            for enabled in (False, True):
                out, cache = root / str(enabled), root / ('cache-' + str(enabled))
                args = ['--arm', 'baseline', '--epochs', '2', '--n-train', '8',
                        '--n-val', '8', '--n-test', '8', '--data-rng', 'isolated', '--mask-mode', 'bool']
                if enabled: args.append('--router-telemetry')
                proc = run(str(out), str(cache), *args)
                self.assertEqual(proc.returncode, 0, proc.stdout[-1000:] + proc.stderr[-1000:])
                folder = next(out.iterdir())
                manifest = json.loads((folder / 'manifest.json').read_text())
                metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
                state = torch.load(folder / 'final.pt', weights_only=True)
                runs.append((manifest, metrics, state))
            self.assertEqual(runs[0][0]['split_hashes'], runs[1][0]['split_hashes'])
            self.assertFalse(runs[0][0]['router_telemetry'])
            self.assertTrue(runs[1][0]['router_telemetry'])
            self.assertEqual(runs[1][0]['mask_mode'], 'bool')
            self.assertTrue(all('router_telemetry' not in row for row in runs[0][1]))
            self.assertEqual(len(runs[1][1]), 2)
            for row in runs[1][1]:
                self.assertEqual(len(row['router_telemetry']), 2)
                self.assertTrue(all(snapshot is not None for snapshot in row['router_telemetry'].values()))
                self.assertTrue(all(snapshot['forwards'] == 1 for snapshot in row['router_telemetry'].values()))
            for left, right in zip(runs[0][1], runs[1][1]):
                for key in ('valid_accuracy', 'valid_loss', 'lr'): self.assertEqual(left[key], right[key])
            self.assertEqual(set(runs[0][2]), set(runs[1][2]))
            for key in runs[0][2]: self.assertTrue(torch.equal(runs[0][2][key], runs[1][2][key]), key)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
