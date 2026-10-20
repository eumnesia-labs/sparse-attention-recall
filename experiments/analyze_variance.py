"""Apply M4 Amendment 2 rules 1--4 offline; never fetch or launch GPU jobs.

Usage: python experiments/analyze_variance.py screen.json [variance.json ...]
       --output /tmp/variance-verdict.json
Missing/ambiguous cells remain unknown. Never pick a better duplicate run.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

EXPECTED = dict(seq_len=2048, num_kv_pairs=128, vocab_size=8192, d_model=128,
                n_layers=2, learning_rate=3.2e-4, weight_decay=0.1,
                early_stopping_threshold=0.98, k_sel=4, compress_block_size=16,
                sliding_window_size=32, num_heads=4, micro_batch=8, grad_accum=2,
                grad_clip=0.0, epochs=60, n_train=20000, n_val=2000, n_test=2000)
ARM_KWARGS = {'baseline': {}, 'router': {'router_gate': True},
              'maxpool': {'superblock_pool': 'max'},
              'router+maxpool': {'router_gate': True, 'superblock_pool': 'max'},
              'dense': {'dropout': 0.1, 'num_heads': 4}}
SEEDS = (123, 231, 312)
TOL = 1e-12  # recognize registered decimal boundaries after binary subtraction


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def ge(a, b):
    return a > b or math.isclose(a, b, rel_tol=0, abs_tol=TOL)


def summarize_run(record, source):
    result, manifest = record.get('result') or {}, record.get('manifest') or {}
    protocol = manifest.get('protocol') or {}
    arm = result.get('arm', manifest.get('arm', record.get('arm')))
    seed = result.get('seed', manifest.get('seed', record.get('seed')))
    s1 = None if arm == 'dense' else result.get('s1', manifest.get('s1', protocol.get('s1')))
    reasons = []
    if not isinstance(seed, int) or isinstance(seed, bool): reasons.append('invalid_seed')
    if result.get('status') != 'complete': reasons.append('status_not_complete')
    if result.get('time_capped') is not False: reasons.append('time_capped_or_unknown')
    if result.get('protocol_conformant') is not True or manifest.get('protocol_conformant') is not True:
        reasons.append('protocol_not_certified')
    overlap = result.get('split_overlap') or {}
    if any(overlap.get(k) != 0 for k in ('heldout_in_train', 'heldout_in_validation', 'validation_in_train')):
        reasons.append('split_overlap_or_unknown')
    mismatches = [k for k, v in EXPECTED.items() if protocol.get(k) != v]
    if mismatches: reasons.append('protocol_mismatch:' + ','.join(mismatches))
    if manifest.get('data_rng', 'legacy') != 'legacy': reasons.append('data_convention_not_legacy')
    cache = manifest.get('data_cache_hit_at_start') or {}
    if any(cache.get(k) is not False for k in ('train', 'validation', 'heldout_test')):
        reasons.append('data_cache_not_all_cold')
    if manifest.get('overrides') != {}: reasons.append('protocol_overrides')
    if protocol.get('selection_rule') != 'best validation accuracy, earliest epoch on ties':
        reasons.append('selection_rule_mismatch')
    if protocol.get('primary_metric') != 'held-out test accuracy of the best-validation checkpoint':
        reasons.append('primary_metric_mismatch')
    if arm not in ARM_KWARGS or manifest.get('arm_kwargs') != ARM_KWARGS.get(arm):
        reasons.append('arm_or_kwargs_mismatch')
    if manifest.get('arm') != arm or manifest.get('seed') != seed:
        reasons.append('identity_mismatch')
    if any(obj.get('arm', arm) != arm or obj.get('seed', seed) != seed for obj in (record, result)):
        reasons.append('identity_mismatch')
    if any(obj.get('s1', s1) != s1 for obj in (record, result, manifest, protocol)):
        reasons.append('s1_mismatch')
    mixer = manifest.get('mixer')
    if arm == 'dense':
        if mixer != 'zoology.mixers.attention.MHA': reasons.append('dense_mixer_mismatch')
    elif mixer not in (None, 'hier_nsa.HierSparseAttention'):
        reasons.append('hier_mixer_mismatch')
    if arm != 'dense' and (not isinstance(s1, int) or isinstance(s1, bool) or s1 < 1):
        reasons.append('invalid_s1')
    if arm != 'dense' and (protocol.get('superblock_group') != 11 or protocol.get('n_superblocks') != 12):
        reasons.append('selector_layout_mismatch')
    if arm != 'dense' and isinstance(s1, int) and protocol.get('coarse_pairs_per_query') != 12 + min(s1, 12) * 11:
        reasons.append('selector_count_mismatch')
    metric_rows = record.get('metrics') or []
    curve = [m.get('valid_accuracy') for m in metric_rows]
    epochs = result.get('epochs_run')
    valid_curve = bool(curve) and all(finite(v) and 0 <= v <= 1 for v in curve)
    if not valid_curve: reasons.append('invalid_or_missing_validation_curve')
    if (not isinstance(epochs, int) or epochs < 1 or epochs > 60 or len(curve) != epochs
            or [m.get('epoch') for m in metric_rows] != list(range(len(metric_rows)))):
        reasons.append('incomplete_epoch_records')
    if epochs != 60 and not (result.get('early_stopped') is True and valid_curve and curve[-1] > .98):
        reasons.append('training_not_protocol_complete')
    first_stop = next((i for i, v in enumerate(curve) if finite(v) and v > .98), None)
    if ((first_stop is not None and (first_stop != len(curve) - 1 or result.get('early_stopped') is not True))
            or (result.get('early_stopped') is True and first_stop is None)):
        reasons.append('early_stop_rule_mismatch')
    primary = result.get('PRIMARY_test_accuracy_at_best_validation')
    if not finite(primary) or not 0 <= primary <= 1: reasons.append('invalid_primary_score')
    final = result.get('test_accuracy_final')
    if not finite(final) or not 0 <= final <= 1: reasons.append('invalid_final_score')
    jump = next((i for i, v in enumerate(curve) if finite(v) and v > .30), None)
    return dict(source=source, run_id=result.get('run_id', manifest.get('run_id')),
                run_folder=manifest.get('run_folder'), arm=arm,
                s1=s1 if isinstance(s1, int) else None, seed=seed if isinstance(seed, int) else None,
                eligible=not reasons, reasons=reasons, primary=primary if finite(primary) else None,
                final_score=final if finite(final) else None,
                epochs_trained=epochs if isinstance(epochs, int) and not isinstance(epochs, bool) else None,
                jump_epoch=jump, non_transition=valid_curve and jump is None,
                validation_curve=[v if finite(v) else None for v in curve],
                data_convention=manifest.get('data_rng', 'legacy'))


def analyze(paths):
    rows, identities = [], set()
    for path in paths:
        payload = json.loads(Path(path).read_text())
        records = payload if isinstance(payload, list) else payload.get('runs', [payload])
        if not isinstance(records, list): raise ValueError(f'{path}: expected a run list')
        for record in records:
            if not isinstance(record, dict): raise ValueError(f'{path}: expected per-run objects')
            # Same imported run in two input files is one observation; conflicting repeats
            # retain distinct rows and become ambiguous rather than best-of-repeats selection.
            identity = json.dumps(record, sort_keys=True)
            if identity in identities: continue
            identities.add(identity)
            rows.append(summarize_run(record, str(path)))
    cells = {}
    for row in rows:
        if row['eligible']:
            cells.setdefault((row['arm'], row['s1'], row['seed']), []).append(row)
    ambiguous = [dict(arm=a, s1=s, seed=n, count=len(v))
                 for (a, s, n), v in cells.items() if len(v) > 1]

    def cell(arm, s1, seed):
        found = cells.get((arm, s1, seed), [])
        return found[0] if len(found) == 1 else None

    bases = [cell('baseline', 2, seed) for seed in SEEDS]
    complete = all(row is not None for row in bases)
    spread = dict(complete_seed_set=complete, seeds=list(SEEDS),
                  available_scores={str(seed): row['primary'] for seed, row in zip(SEEDS, bases) if row},
                  range=None, mean=None, median=None, sample_sd=None,
                  non_transition_count=None, band='insufficient_data')
    if complete:
        scores = [row['primary'] for row in bases]
        r = max(scores) - min(scores)
        band = ('small' if ge(.05, r) else
                'comparable_to_m4_deficits' if ge(r, .15) else 'material')
        spread.update(range=r, mean=statistics.mean(scores), median=statistics.median(scores),
                      sample_sd=statistics.stdev(scores),
                      non_transition_count=sum(row['non_transition'] for row in bases), band=band)
    s2, s12, dense = cell('baseline', 2, 123), cell('baseline', 12, 123), cell('dense', None, 123)
    deficits = {}
    for arm in ('router', 'maxpool', 'router+maxpool'):
        row = cell(arm, 2, 123)
        deficits[arm] = row['primary'] - s2['primary'] if row and s2 else None
    interpretation = 'insufficient_data'
    if complete:
        if spread['band'] == 'comparable_to_m4_deficits':
            interpretation = 'one-seed paired observations amid large variability; not declared noise'
        elif spread['band'] == 'small' and spread['non_transition_count'] == 0:
            interpretation = 'M4 deficits exceed observed baseline spread; strengthens concern, does not prove harm'
        else:
            interpretation = 'inconclusive; baseline range is not the distribution of paired arm differences'
    budget = dict(classification='insufficient_data', delta_s12_s2=None, recovery_s12=None,
                  near_dense_s2=None, near_dense_s12=None,
                  interpretation_of_rule4='S1=12 recovers means >=+0.10; S1=2 does not means it misses the near-dense margin, as in the archived review adopted by Amendment 2')
    if s2 and s12:
        delta = s12['primary'] - s2['primary']
        budget.update(delta_s12_s2=delta, recovery_s12=ge(delta, .10))
    if dense:
        for key, row in (('near_dense_s2', s2), ('near_dense_s12', s12)):
            if row: budget[key] = ge(row['primary'], dense['primary'] - .02)
    if s2 and s12 and dense:
        classification = 'inconclusive'
        if budget['recovery_s12'] and not budget['near_dense_s2']:
            classification = 'recovery_prioritise_curve'
        elif (not budget['near_dense_s2'] and not budget['near_dense_s12']
              and abs(budget['delta_s12_s2']) < .05
              and not math.isclose(abs(budget['delta_s12_s2']), .05, rel_tol=0, abs_tol=TOL)):
            classification = 'budget_alone_not_supported'
        budget['classification'] = classification
    return dict(protocol='M4 Amendment 2, 2026-10-01, rules 1-4', rows=rows,
                ambiguous_cells=ambiguous, baseline_spread=spread,
                m4=dict(deficits=deficits, interpretation=interpretation), budget=budget,
                caveats=['dense_not_paired_on_training_noise', 's12_budget_plus_candidate_coverage',
                         'seed123_test_reuse_exploratory', 'three_seeds_not_stable_variance',
                         'baseline_range_not_paired_arm_difference_distribution'],
                sources={'protocol': 'docs/protocol-m4-routing-arms.md:120',
                         'rule4_clarification': 'audits/sol-2026-10-01/REPORT.md:118'})


def print_table(verdict):
    print('arm                 S1   seed   primary   final     epochs jump eligible')
    for row in verdict['rows']:
        def score(value): return f'{value:.6f}' if finite(value) else '?'
        print(f"{str(row['arm']):19} {str(row['s1']):4} {str(row['seed']):6} "
              f"{score(row['primary']):9} {score(row['final_score']):9} "
              f"{str(row['epochs_trained']):6} {str(row['jump_epoch']):4} {row['eligible']}")
        if row['reasons']: print('  excluded: ' + '; '.join(row['reasons']))
    print('Baseline spread:', json.dumps(verdict['baseline_spread'], sort_keys=True))
    print('M4:', json.dumps(verdict['m4'], sort_keys=True))
    print('Budget:', json.dumps(verdict['budget'], sort_keys=True))
    print('Caveats:', '; '.join(verdict['caveats']))
    if verdict['ambiguous_cells']: print('Ambiguous cells:', json.dumps(verdict['ambiguous_cells']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('variance-verdict.json'))
    args = parser.parse_args()
    protected = Path(__file__).resolve().parent / 'm4_results'
    if args.output.resolve().is_relative_to(protected.resolve()):
        parser.error('write verdicts outside experiments/m4_results (collection-owned)')
    if args.output.resolve() in [p.resolve() for p in args.inputs]:
        parser.error('output must not overwrite an input')
    verdict = analyze(args.inputs)
    print_table(verdict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, allow_nan=False) + '\n')
    print('Verdict written:', args.output)


if __name__ == '__main__':
    main()
