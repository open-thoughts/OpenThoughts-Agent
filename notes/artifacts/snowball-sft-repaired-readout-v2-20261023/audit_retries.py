"""Check recovery retries for a numeric verifier outcome before the selected trial."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, '/home1/11584/lukedhlee/snowball/ota-fp32-20261020/data/relay/pilot')
from readout import read_outcome  # noqa: E402


def main():
    root = Path(__file__).resolve().parent
    source = json.loads((root / 'data_snapshot.json').read_text())
    groups = {}
    for candidate in source['candidates']:
        if not candidate.get('result'):
            continue
        selected = Path(candidate['result'])
        groups[str(selected)] = candidate
    multiple = []
    findings = []
    for selected_text, candidate in sorted(groups.items()):
        selected = Path(selected_text)
        with selected.open('rb') as stream:
            head = stream.read(4096).decode()
        match = re.search(r'"trial_relpath"\s*:\s*("(?:[^"\\]|\\.)*")', head)
        assert match, selected
        selected_attempt = Path(json.loads(match[1])).name
        assert selected_attempt.isdigit(), (selected, selected_attempt)
        prior_paths = sorted(path for path in selected.parent.glob('attempts/*/result.json')
                             if int(path.parent.name) < int(selected_attempt))
        if not prior_paths:
            continue
        paths = []
        for path in [*prior_paths, selected]:
            task, reward, error = read_outcome(str(path))
            assert task == candidate['task']
            paths.append({'result': str(path), 'reward': reward, 'error': error})
        earlier_numeric = [row for row in paths[:-1] if row['reward'] is not None]
        record = {'key': candidate['key'], 'root': candidate['root'],
                  'selected': candidate['result'], 'trials': paths,
                  'earlier_numeric': earlier_numeric}
        multiple.append(record)
        if earlier_numeric:
            findings.append(record)
    print(json.dumps({'source_snapshot_sha256': source['semantic_sha256'],
                      'groups_checked': len(groups), 'multi_attempt_groups': len(multiple),
                      'earlier_numeric_findings': len(findings),
                      'attempts': multiple, 'findings': findings}, indent=2))
    assert not findings, 'An earlier numeric verifier result needs explicit adjudication.'


if __name__ == '__main__':
    main()
