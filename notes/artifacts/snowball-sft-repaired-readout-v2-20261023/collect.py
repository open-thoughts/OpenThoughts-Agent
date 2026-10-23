"""Collect immutable canonical scores and settled repair overlays, read-only."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

BASE = Path('/scratch/11584/lukedhlee/experiments')
EVAL = BASE / 'sft_eval'
OTA = Path('/home1/11584/lukedhlee/snowball/ota-fp32-20261020')
sys.path.insert(0, str(OTA / 'data/relay/pilot'))
from readout import read_outcome  # noqa: E402

SPECS = {'cfv2': (8, 'cf2_20261022'), 'cfr2': (8, 'cf2_20261022'),
         **{tag: (3, 'cf2e57_20261022') for tag in ('cv5', 'cr5', 'cv7', 'cr7')}}
FAMILIES = {'tb21': ('tb21',), 'swe': ('swe_s0', 'swe_s1'),
            'tblite': ('tblite_s0', 'tblite_s1')}
EXPECTED = {'tb21': 88, 'swe': 100, 'tblite': 100}
ROOTS = ['cf2-missing-score-recovery2-20261022', 'cf2-harness-fixes-20261022',
         'cf2-verified-repair-20261023', 'cf2-image-query-repair-20261023',
         'cf2-remaining-repair-20261023', 'cf2-swe-repair-20261023',
         'cf2-cli-repair-20261023', 'cf2-late-repair-20261023',
         'cf2-startup-repair-20261023']
PREVIOUS = BASE / 'cf2-repaired-readout-20261023'
PREVIOUS_SNAPSHOT_SHA256 = '3a8188bca6d000e29446964a73caa1bf577ea52b78412857de96c01077eb6593'


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def slot_key(slot):
    return (slot['model'], slot['family'], int(slot['rep']), slot['task'])


def key_text(key):
    return '/'.join(map(str, key))


def same_reward(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), abs_tol=1e-12, rel_tol=1e-12)


def collect():
    # Verify the sealed first revision before reading any new recovery results.
    previous_sums = json.loads((PREVIOUS / 'SHA256SUMS.json').read_text())
    for name, digest in previous_sums.items():
        assert '/' not in name and '..' not in name
        assert sha_file(PREVIOUS / name) == digest, ('Previous revision mutated', name)
    previous = json.loads((PREVIOUS / 'data_snapshot.json').read_text())
    assert previous['semantic_sha256'] == PREVIOUS_SNAPSHOT_SHA256
    assert hashlib.sha256(json.dumps({k: v for k, v in previous.items()
                                    if k != 'semantic_sha256'}, sort_keys=True).encode()).hexdigest() == PREVIOUS_SNAPSHOT_SHA256
    previous_numeric = {slot_key(row): row for row in previous['canonical'] if row['reward'] is not None}
    previous_view = {slot_key(row): row for row in previous['canonical']}
    for row in previous['candidates']:
        if previous_view[slot_key(row)]['reward'] is None:
            previous_view[slot_key(row)] = row
        if row['reward'] is not None:
            previous_numeric[slot_key(row)] = row
    canonical = {}
    run_receipts = []
    reports = {}
    for root in ('cf2-epochs57-20261022', 'cf-only-fp32-capacity2-pass8-20261022'):
        directory = BASE / root
        status = (directory / 'COMPARISON_STATUS').read_text()
        assert 'RESULTS_READY' in status, (root, status)
        name = 'passk8.json' if 'pass8' in root else 'passk.json'
        markdown_name = 'report8.md' if 'pass8' in root else 'report.md'
        data_text = (directory / name).read_text()
        reports[root] = {'status': status, 'path': str(directory / name),
                         'sha256': sha_file(directory / name), 'data_text': data_text,
                         'markdown_path': str(directory / markdown_name),
                         'markdown_sha256': sha_file(directory / markdown_name),
                         'markdown_text': (directory / markdown_name).read_text(),
                         'data': json.loads(data_text)}
    for tag, (reps, day) in SPECS.items():
        for family, shards in FAMILIES.items():
            for rep in range(1, reps + 1):
                count = 0
                for shard in shards:
                    run = EVAL / 'tb2_jobs' / f'{shard}_{tag}_r{rep}_{day}'
                    config = EVAL / 'runs' / (run.name + '.yaml')
                    driver = EVAL / 'state' / run.name / 'driver.out'
                    run_receipts.append({'run': str(run), 'source_config': str(config),
                                         'source_config_sha256': sha_file(config),
                                         'driver': str(driver)})
                    for path in sorted(run.glob('*/result.json')):
                        task, reward, error = read_outcome(str(path))
                        assert task is not None and (reward is None or math.isfinite(float(reward)))
                        key = (tag, family, rep, task)
                        assert key not in canonical, ('Duplicate canonical slot', key)
                        info = path.stat()
                        canonical[key] = {'model': tag, 'family': family, 'rep': rep,
                                          'task': task, 'reward': reward, 'error': error,
                                          'result': str(path), 'size': info.st_size,
                                          'mtime_ns': info.st_mtime_ns}
                        count += 1
                assert count == EXPECTED[family], (tag, family, rep, count)
    assert len(canonical) == 8064
    original_hashes = {}
    recovery_hashes = {}
    root_receipts = []
    candidates = []
    numeric_keys = {}
    for name in ROOTS:
        root = BASE / name
        status_path = root / 'STATUS'
        status = status_path.read_text() if status_path.exists() else ''
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest_slots = manifest['slots']
        manifest_by_key = {slot_key(slot): slot for slot in manifest_slots}
        assert len(manifest_by_key) == len(manifest_slots), ('Duplicate manifest slot', name)
        metadata = {'root': name, 'path': str(root), 'status': status,
                    'manifest_sha256': sha_file(manifest_path),
                    'manifest_slots': len(manifest_slots), 'harbor_commit': manifest.get('harbor_commit'),
                    'settled': False, 'driver_overlays': [], 'candidates_included': 0}
        if name == ROOTS[0]:
            settled = 'RECOVERY_FINISHED' in status
            overlay_paths = [root / 'recovery_overlay.json']
            prior_path = root / 'prior_recovery_overlay.json'
            prior = json.loads(prior_path.read_text())
            prior_keys = {slot_key(slot) for slot in prior['slots']}
            assert set(manifest_by_key) <= prior_keys
            expected_keys = prior_keys | set(manifest_by_key)
        elif name == ROOTS[1]:
            settled = 'PILOT_FINISHED' in status
            overlay_paths = [root / 'fix1_overlay.json']
            expected_keys = set(manifest_by_key)
        else:
            expected_runs = {slot['recovery_run'] for slot in manifest_slots}
            expected_paths = {root / (run + '.overlay.json') for run in expected_runs}
            available = set(root.glob('*.overlay.json'))
            assert available <= expected_paths, ('Unexpected driver overlay', name, available - expected_paths)
            metadata.update(expected_drivers=len(expected_runs), finished_drivers=len(available),
                            missing_drivers=sorted(p.name for p in expected_paths - available))
            settled = 'RECOVERY_FINISHED' in status and available == expected_paths
            overlay_paths = sorted(available)
            expected_keys = set(manifest_by_key)
        seen = set()
        slots = []
        for path in overlay_paths:
            if not path.exists():
                assert not settled, ('Settled root lacks overlay', path)
                continue
            overlay = json.loads(path.read_text())
            rows = overlay['slots']
            keys = {slot_key(slot) for slot in rows}
            assert len(keys) == len(rows) and not (seen & keys), ('Duplicate overlay slots', path)
            seen |= keys
            if name not in ROOTS[:2]:
                run = path.name.removesuffix('.overlay.json')
                expected_group = {slot_key(s) for s in manifest_slots if s['recovery_run'] == run}
                assert keys == expected_group, ('Driver/manifest disagreement', path)
                for slot in rows:
                    original = manifest_by_key[slot_key(slot)]
                    for field in ('original_result', 'original_sha256', 'recovery_run'):
                        assert slot[field] == original[field], (path, field, slot_key(slot))
                assert 'completed_utc' in overlay and 'driver_exit' in overlay
            metadata['driver_overlays'].append({'path': str(path), 'sha256': sha_file(path),
                                                'slots': len(rows), 'driver_exit': overlay.get('driver_exit')})
            slots.extend(rows)
        assert seen <= expected_keys, ('Unknown overlay slot', name)
        metadata.update(settled=settled, overlay_slots=len(slots),
                        provisional_numeric=sum(s.get('recovery', {}).get('reward') is not None for s in slots))
        if not settled:
            root_receipts.append(metadata)
            continue
        assert seen == expected_keys, ('Incomplete settled overlay', name)
        if name not in ROOTS[:2]:
            summary_path = root / 'recovery_overlay.json'
            summary = json.loads(summary_path.read_text())
            summary_slots = {slot_key(s): s['recovery'] for s in summary['slots']}
            assert len(summary_slots) == len(summary['slots'])
            assert summary_slots == {slot_key(s): s['recovery'] for s in slots}, ('Summary/driver mismatch', name)
            metadata['summary_complete_metadata'] = summary.get('complete')
            metadata['summary_sha256'] = sha_file(summary_path)
        for slot in slots:
            key = slot_key(slot)
            assert key in canonical, ('Unknown canonical slot', name, key)
            if name == ROOTS[-1]:
                assert key not in previous_numeric, ('Refuse retry of scored revision-one outcome', key)
                prior = previous_view[key]
                assert slot['prior_recovery'] == {field: prior[field] for field in ('result', 'reward', 'error')}
                assert sha_file(Path(prior['result'])) == slot['prior_recovery_sha256'], ('Prior recovery SHA mismatch', key)
            base = canonical[key]
            assert base['reward'] is None, ('Refuse replacement of scored original', name, key)
            path = Path(slot['original_result'])
            assert str(path) == base['result'], (name, key, 'Original path mismatch')
            if str(path) not in original_hashes:
                original_hashes[str(path)] = sha_file(path)
            assert original_hashes[str(path)] == slot['original_sha256'], ('Original SHA mismatch', name, key)
            base['sha256'] = original_hashes[str(path)]
            config = Path(slot.get('source_config') or str(EVAL / 'runs' / (Path(slot['source_run']).name + '.yaml')))
            assert sha_file(config) == slot['source_config_sha256'], ('Source config SHA mismatch', name, key)
            recovery = slot.get('recovery', {})
            result_path = Path(recovery['result']) if recovery.get('result') else None
            if result_path is not None:
                assert result_path.exists(), ('Missing recovery result', name, result_path)
                task, reward, error = read_outcome(str(result_path))
                assert task == key[3] and same_reward(reward, recovery.get('reward')), ('Recovery reward mismatch', name, key)
                assert error == recovery.get('error'), ('Recovery error mismatch', name, key, error, recovery.get('error'))
                recovery_hashes[str(result_path)] = sha_file(result_path)
                result_sha = recovery_hashes[str(result_path)]
            else:
                assert recovery.get('reward') is None
                reward, error, result_sha = None, recovery.get('error') or 'NoRecoveryResult', None
            record = {'root': name, 'key': key_text(key), 'model': key[0], 'family': key[1],
                      'rep': key[2], 'task': key[3], 'reward': reward, 'error': error,
                      'result': str(result_path) if result_path else None, 'sha256': result_sha,
                      'original_result': str(path), 'original_sha256': original_hashes[str(path)],
                      'recovery_run': slot.get('recovery_run')}
            if reward is not None:
                assert key not in numeric_keys, ('Duplicate numeric replacement', key, numeric_keys.get(key), name)
                numeric_keys[key] = name
            candidates.append(record)
        metadata['candidates_included'] = len(slots)
        root_receipts.append(metadata)
    prior_candidates = [row for row in candidates if row['root'] != ROOTS[-1]]
    assert prior_candidates == previous['candidates'], 'Previous recovery records mutated'
    current_numeric = {key: row for key, row in canonical.items() if row['reward'] is not None}
    current_numeric.update({slot_key(row): row for row in candidates if row['reward'] is not None})
    for key, before in previous_numeric.items():
        assert current_numeric[key] == before, ('Previous numeric outcome mutated', key)
    diagnostic_view = dict(canonical)
    for candidate in candidates:
        item = slot_key(candidate)
        if diagnostic_view[item]['reward'] is None:
            diagnostic_view[item] = candidate
    unresolved = [row for item, row in diagnostic_view.items()
                  if row['reward'] is None and (item[1], item[3]) != ('tblite', 'git-repo-forensics')]
    error_examples = []
    for error, _ in collections.Counter(row['error'] or 'NoReward' for row in unresolved).most_common():
        tasks_seen = set()
        for row in unresolved:
            if (row['error'] or 'NoReward') != error or row['task'] in tasks_seen:
                continue
            tasks_seen.add(row['task'])
            example = dict(row)
            path = Path(row['result']) if row.get('result') else None
            example['message'] = 'No result path; inspect settled driver overlay/driver.out.'
            if path is not None and path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 2**18))
                    text = stream.read().decode(errors='replace')
                marker = text.rfind('"exception_info":')
                if marker >= 0:
                    try:
                        exception, _ = json.JSONDecoder().raw_decode(text[marker:].split(':', 1)[1].lstrip())
                        if exception:
                            example['message'] = exception.get('exception_message', '')[:1800]
                    except ValueError:
                        example['message'] = 'Exception exceeds bounded diagnostic read; inspect original artifact.'
            error_examples.append(example)
            if len(tasks_seen) >= 3:
                break
    payload = {'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'revision': 2,
               'previous_revision': {
                   'path': str(PREVIOUS), 'source_snapshot_sha256': PREVIOUS_SNAPSHOT_SHA256,
                   'sealed_hashes': previous_sums, 'numeric_outcomes_preserved': len(previous_numeric),
                   'report_json_text': (PREVIOUS / 'report.json').read_text(),
                   'report_markdown_text': (PREVIOUS / 'report.md').read_text()},
               'canonical': [canonical[k] for k in sorted(canonical)], 'candidates': candidates,
               'error_examples': error_examples,
               'roots': root_receipts, 'run_receipts': run_receipts,
               'original_reports': reports,
               'validation': {'canonical_slots': len(canonical),
                              'original_files_sha256_verified': len(original_hashes),
                              'recovery_files_sha256_verified': len(recovery_hashes),
                              'numeric_replacement_slots': len(numeric_keys),
                              'duplicate_numeric_replacements': 0,
                              'original_numeric_replacements': 0},
               'extractor': {'path': str(OTA / 'data/relay/pilot/readout.py'),
                             'sha256': sha_file(OTA / 'data/relay/pilot/readout.py')},
               'all_roots_settled': all(root['settled'] for root in root_receipts)}
    payload['semantic_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


if __name__ == '__main__':
    print(json.dumps(collect(), separators=(',', ':')))
