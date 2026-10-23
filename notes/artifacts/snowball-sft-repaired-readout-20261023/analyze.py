"""Paired original/repaired sensitivity readout; missing rewards remain unknown."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPECTED = {'tb21': 88, 'swe': 100, 'tblite': 100}
VALID_EXPECTED = {'tb21': 88, 'swe': 100, 'tblite': 99}
PAIRS = {3: ('cfv2', 'cfr2'), 5: ('cv5', 'cr5'), 7: ('cv7', 'cr7')}
INVALID = {('tblite', 'git-repo-forensics')}
SEED = 20261022
BOOTSTRAPS = 10000


def key(row):
    return (row['model'], row['family'], int(row['rep']), row['task'])


def key_text(slot):
    return '/'.join(map(str, slot))


def pass_at_k(n, correct, k):
    return 1.0 if n - correct < k else 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def interval(values, indexes):
    return np.quantile(values[indexes].mean(axis=1), [0.025, 0.975]).tolist()


def metrics(data, vanilla, relay, family, reps, k, panel=None):
    task_sets = [{slot[3] for slot, row in data.items()
                  if slot[:3] == (tag, family, rep) and row['reward'] is not None
                  and (family, slot[3]) not in INVALID}
                 for tag in (vanilla, relay) for rep in reps]
    tasks = sorted(set.intersection(*task_sets) if panel is None else panel)
    assert tasks, (vanilla, relay, family, reps, k)
    values = {}
    for tag in (vanilla, relay):
        counts = [sum(data[(tag, family, rep, task)]['reward'] >= 1 for rep in reps)
                  for task in tasks]
        values[tag] = np.array([pass_at_k(len(reps), count, k) for count in counts])
    rng = np.random.default_rng(SEED)
    indexes = rng.integers(0, len(tasks), size=(BOOTSTRAPS, len(tasks)))
    difference = values[relay] - values[vanilla]
    return {'tasks': len(tasks), 'valid_tasks': VALID_EXPECTED[family],
            'scheduled_tasks': EXPECTED[family], 'panel': tasks,
            'vanilla': float(values[vanilla].mean()), 'relay': float(values[relay].mean()),
            'vanilla_ci95': interval(values[vanilla], indexes),
            'relay_ci95': interval(values[relay], indexes),
            'relay_minus_vanilla': float(difference.mean()),
            'difference_ci95': interval(difference, indexes)}


def write_csv(path, rows, fields=None):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output):
    supplied_hash = source['semantic_sha256']
    assert supplied_hash == hashlib.sha256(json.dumps(
        {k: v for k, v in source.items() if k != 'semantic_sha256'}, sort_keys=True).encode()).hexdigest()
    original = {key(row): row for row in source['canonical']}
    assert len(original) == len(source['canonical']) == 8064
    repaired = {slot: dict(row) for slot, row in original.items()}
    histories = collections.defaultdict(list)
    replacements = []
    invalid_replacements = []
    for candidate in source['candidates']:
        slot = key(candidate)
        assert slot in repaired and original[slot]['reward'] is None
        histories[slot].append(candidate)
        if candidate['reward'] is not None:
            assert repaired[slot]['reward'] is None, ('Duplicate numeric replacement', slot)
            assert math.isfinite(float(candidate['reward']))
            repaired[slot] = dict(candidate)
            (invalid_replacements if (slot[1], slot[3]) in INVALID
             else replacements).append(candidate)
        elif repaired[slot]['reward'] is None:
            repaired[slot] = dict(candidate)
    for slot, row in original.items():
        if row['reward'] is not None:
            assert repaired[slot] == row, ('Original numeric score changed', slot)
    expected_sets = {family: {slot[3] for slot in original if slot[:3] == ('cfv2', family, 1)}
                     for family in EXPECTED}
    for tag in {slot[0] for slot in original}:
        for family in EXPECTED:
            reps = sorted({slot[2] for slot in original if slot[:2] == (tag, family)})
            for rep in reps:
                tasks = {slot[3] for slot in original if slot[:3] == (tag, family, rep)}
                assert tasks == expected_sets[family], ('Task universe drift', tag, family, rep)
    rows = []
    for epoch, (vanilla, relay) in PAIRS.items():
        for family in EXPECTED:
            for k in (1, 3):
                original_metrics = metrics(original, vanilla, relay, family, (1, 2, 3), k)
                repaired_metrics = metrics(repaired, vanilla, relay, family, (1, 2, 3), k)
                same_panel = metrics(repaired, vanilla, relay, family, (1, 2, 3), k,
                                     panel=original_metrics['panel'])
                assert all(abs(same_panel[name] - original_metrics[name]) < 1e-12
                           for name in ('vanilla', 'relay', 'relay_minus_vanilla'))
                rows.append({'epoch': epoch, 'family': family, 'k': k, 'n_attempts': 3,
                             'vanilla_tag': vanilla, 'relay_tag': relay,
                             'original': original_metrics, 'repaired': repaired_metrics})
    for family in EXPECTED:
        rows.append({'epoch': 3, 'family': family, 'k': 4, 'n_attempts': 8,
                     'vanilla_tag': 'cfv2', 'relay_tag': 'cfr2',
                     'original': metrics(original, 'cfv2', 'cfr2', family, tuple(range(1, 9)), 4),
                     'repaired': metrics(repaired, 'cfv2', 'cfr2', family, tuple(range(1, 9)), 4)})
    # Independently compare all original-panel metrics against the previously completed reports.
    baseline_validations = []
    for row in rows:
        report_name = ('cf-only-fp32-capacity2-pass8-20261022' if row['k'] == 4
                       else 'cf2-epochs57-20261022')
        baseline = source['original_reports'][report_name]['data']['results']
        matches = [entry for entry in baseline if entry['x'] == row['relay_tag']
                   and entry['y'] == row['vanilla_tag'] and entry['family'] == row['family']
                   and entry['k'] == row['k']]
        assert len(matches) == 1, ('Missing or duplicate original report row', row)
        prior = matches[0]
        assert prior['tasks'] == row['original']['tasks']
        assert abs(prior['x_pass_at_k'] - row['original']['relay']) < 1e-12
        assert abs(prior['y_pass_at_k'] - row['original']['vanilla']) < 1e-12
        assert np.allclose(prior['diff_ci95'], row['original']['difference_ci95'], atol=1e-12, rtol=1e-12)
        baseline_validations.append({'epoch': row['epoch'], 'family': row['family'],
                                     'k': row['k'], 'matches_original_report': True})
    census = []
    unresolved = []
    errors = collections.Counter()
    for slot, row in repaired.items():
        if (slot[1], slot[3]) in INVALID or row['reward'] is not None:
            continue
        error = row['error'] or 'NoReward'
        errors[error] += 1
        unresolved.append({'model': slot[0], 'family': slot[1], 'rep': slot[2], 'task': slot[3],
                           'original_error': original[slot]['error'], 'latest_error': error,
                           'latest_root': row.get('root', 'canonical'),
                           'latest_result': row.get('result'), 'sha256': row.get('sha256'),
                           'original_result': original[slot]['result'],
                           'recovery_attempts': len(histories[slot])})
    for epoch, pair in PAIRS.items():
        for arm, tag in zip(('Vanilla', 'Relay'), pair):
            for family in EXPECTED:
                slots = [slot for slot in original if slot[:2] == (tag, family)
                         and (family, slot[3]) not in INVALID]
                before = [original[slot]['reward'] for slot in slots]
                after = [repaired[slot]['reward'] for slot in slots]
                numeric_before = [reward for reward in before if reward is not None]
                numeric_after = [reward for reward in after if reward is not None]
                census.append({'epoch': epoch, 'arm': arm, 'model': tag, 'family': family,
                               'attempts_per_task': 8 if epoch == 3 else 3,
                               'valid_expected_slots': len(slots),
                               'original_numeric': len(numeric_before),
                               'repaired_numeric': len(numeric_after),
                               'new_numeric': len(numeric_after) - len(numeric_before),
                               'unscored': len(slots) - len(numeric_after),
                               'original_raw_reward_mean': float(np.mean(numeric_before)),
                               'repaired_raw_reward_mean': float(np.mean(numeric_after)),
                               'repaired_partial_rewards': sum(0 < reward < 1 for reward in numeric_after)})
    invalid_rows = [{'key': key_text(slot), 'original_reward': original[slot]['reward'],
                     'raw_repaired_reward': row['reward'], 'raw_error': row['error'],
                     'raw_result': row.get('result'), 'root': row.get('root', 'canonical'),
                     'classification': 'invalid_missing_original_git_fixture'}
                    for slot, row in repaired.items() if (slot[1], slot[3]) in INVALID]
    assert len(invalid_rows) == 28
    valid_slots = [slot for slot in original if (slot[1], slot[3]) not in INVALID]
    summary = {'scheduled_slots': 8064, 'strict_valid_slots': len(valid_slots),
               'invalid_git_slots': len(invalid_rows),
               'original_valid_numeric': sum(original[s]['reward'] is not None for s in valid_slots),
               'repaired_valid_numeric': sum(repaired[s]['reward'] is not None for s in valid_slots),
               'valid_new_numeric': len(replacements), 'invalid_raw_new_numeric': len(invalid_replacements),
               'unscored_valid': len(unresolved),
               'raw_original_numeric': sum(r['reward'] is not None for r in original.values()),
               'raw_repaired_numeric': sum(r['reward'] is not None for r in repaired.values()),
               'all_recovery_roots_settled': source['all_roots_settled']}
    assert summary['strict_valid_slots'] == 8036
    assert summary['original_valid_numeric'] + summary['valid_new_numeric'] == summary['repaired_valid_numeric']
    assert summary['repaired_valid_numeric'] + summary['unscored_valid'] == 8036
    provenance = {'snapshot_utc': source['snapshot_utc'], 'source_snapshot_sha256': supplied_hash,
                  'validation': source['validation'], 'original_report_rows_verified': len(baseline_validations),
                  'roots': source['roots'], 'extractor': source['extractor'],
                  'original_report_sources': {name: {k: v for k, v in report.items() if k not in ('data', 'data_text', 'markdown_text')}
                                              for name, report in source['original_reports'].items()},
                  'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    error_examples = [{'error': error, 'count': count,
                       'top_tasks': collections.Counter(r['task'] for r in unresolved
                                                        if r['latest_error'] == error).most_common(6),
                       'examples': [r for r in unresolved if r['latest_error'] == error][:3]}
                      for error, count in errors.most_common()]
    report = {'snapshot_utc': source['snapshot_utc'], 'summary': summary,
              'method': 'Corrected FP32/capacity2 canonical scores plus settled repaired-stack retries of '
                        'originally unscored slots only. Original numeric scores immutable; no unknown-as-zero '
                        'imputation. Raw fractional rewards retained; full solve means reward>=1. '
                        'Paired complete-case tasks per epoch/benchmark across both arms and all selected reps; '
                        'same panel for pass@1/pass@3. Epoch3 pass@4 uses eight attempts and '
                        '1-C(8-c,4)/C(8,4). Paired task bootstrap,10000 draws,seed20261022,percentile95%CI; '
                        'exploratory/no multiplicity correction. Task filtering may be selective; repairs '
                        'expand panels. Panels differ between epochs. Epoch5 is intermediate on a7-epoch '
                        'schedule; epoch3 used its original3-epoch schedule. Git-forensics excluded from '
                        'strict metrics because its original repository fixture is absent.',
              'comparisons': rows, 'coverage': census, 'unresolved_errors': error_examples,
              'invalid_task_raw_scores': invalid_rows,
              'provenance': provenance}
    output.mkdir(parents=True, exist_ok=True)
    for name, baseline in source['original_reports'].items():
        prefix = 'original_pass8' if 'pass8' in name else 'original_epochs57'
        assert hashlib.sha256(baseline['data_text'].encode()).hexdigest() == baseline['sha256']
        assert hashlib.sha256(baseline['markdown_text'].encode()).hexdigest() == baseline['markdown_sha256']
        (output / f'{prefix}.json').write_text(baseline['data_text'])
        (output / f'{prefix}.md').write_text(baseline['markdown_text'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'unresolved.json').write_text(json.dumps(unresolved, indent=2) + '\n')
    (output / 'error_examples.json').write_text(json.dumps({'examples': source.get('error_examples', [])}, indent=2) + '\n')
    (output / 'invalid_task_raw_scores.json').write_text(json.dumps(invalid_rows, indent=2) + '\n')
    (output / 'replacements.json').write_text(json.dumps(replacements, indent=2) + '\n')
    write_csv(output / 'coverage.csv', census)
    write_csv(output / 'unresolved.csv', unresolved)
    flat = []
    for row in rows:
        entry = {k: row[k] for k in ('epoch', 'family', 'k', 'n_attempts')}
        for view in ('original', 'repaired'):
            metric = row[view]
            for name in ('tasks', 'valid_tasks', 'vanilla', 'relay', 'relay_minus_vanilla'):
                entry[f'{view}_{name}'] = metric[name]
            entry[f'{view}_delta_ci95_lower'] = metric['difference_ci95'][0]
            entry[f'{view}_delta_ci95_upper'] = metric['difference_ci95'][1]
        flat.append(entry)
    write_csv(output / 'comparisons.csv', flat)
    state = 'FINAL_READY' if source['all_roots_settled'] else 'INTERIM_WAITING_FOR_RECOVERY'
    pending = [r['root'] for r in source['roots'] if not r['settled']]
    text = [f'# Corrected-stack evaluation: {state}', '',
            f"Snapshot: {source['snapshot_utc']}.", '',
            f"Strict valid numeric coverage: {summary['repaired_valid_numeric']}/{summary['strict_valid_slots']} "
            f"({100*summary['repaired_valid_numeric']/summary['strict_valid_slots']:.2f}%), up from "
            f"{summary['original_valid_numeric']}/{summary['strict_valid_slots']}. "
            f"Recovered {summary['valid_new_numeric']} valid scores; {summary['unscored_valid']} remain unknown. "
            f"The 28 Git-forensics slots are excluded as an invalid fixture; raw scores are saved separately.", '',
            'Original numeric results are unchanged. Repaired values are a separate sensitivity analysis; '
            'coverage expands through fresh retries on the repaired harness. Each row uses the same task '
            'panel for Vanilla/Relay and both pass@1/pass@3; panels can differ across epochs.', '']
    if pending:
        text += ['Pending recovery roots (all their partial overlays excluded): ' + ', '.join(pending) + '.', '']
    for k in (1, 3, 4):
        text += [f'## Pass@{k}' + (' estimated from eight epoch-three attempts' if k == 4 else ' from repetitions 1–3'), '',
                 '| Epoch | Benchmark | Original V / R | Original tasks | Repaired V / R | Repaired tasks | R−V pp [95% CI] |',
                 '|---:|---|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['k'] != k:
                continue
            before, after = row['original'], row['repaired']
            lo, hi = after['difference_ci95']
            text.append(f"| {row['epoch']} | {row['family']} | "
                        f"{100*before['vanilla']:.1f}% / {100*before['relay']:.1f}% | "
                        f"{before['tasks']}/{before['valid_tasks']} | "
                        f"{100*after['vanilla']:.1f}% / {100*after['relay']:.1f}% | "
                        f"{after['tasks']}/{after['valid_tasks']} | "
                        f"{100*after['relay_minus_vanilla']:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] |")
        text += ['']
    text += ['## Remaining unknown scores', '', '| Latest error | Count | Most frequent tasks |',
             '|---|---:|---|']
    for row in error_examples:
        text.append(f"| {row['error']} | {row['count']} | " + ', '.join(f'{task} ({n})' for task, n in row['top_tasks']) + ' |')
    text += ['', report['method'], '',
             f"Source snapshot SHA256: `{supplied_hash}`. Original report validation: "
             f"{len(baseline_validations)}/21 rows match; source-file hashes checked: "
             f"{source['validation']['original_files_sha256_verified']} original and "
             f"{source['validation']['recovery_files_sha256_verified']} recovery files. "
             'Full source pointers and manifests are in provenance.json/data_snapshot.json.', '']
    (output / 'report.md').write_text('\n'.join(text))
    (output / 'STATUS').write_text(f"{source['snapshot_utc']} {state}; source_sha256={supplied_hash}\n")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).with_name('data_snapshot.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = analyze(json.loads(args.snapshot.read_text()), args.output)
    print(json.dumps({'snapshot_utc': result['snapshot_utc'], 'summary': result['summary']}))
