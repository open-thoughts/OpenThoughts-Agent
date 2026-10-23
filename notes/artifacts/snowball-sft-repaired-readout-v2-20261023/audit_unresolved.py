"""Bounded verifier/setup diagnostics for every unresolved revision-two slot."""
import json
from pathlib import Path
import re


def tail(path, limit):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode(errors='replace')


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / 'report.json').read_text())
    rows = json.loads((root / 'unresolved.json').read_text())
    for row in rows:
        path = Path(row['latest_result']) if row.get('latest_result') else None
        if path is None or not path.exists():
            continue
        text = tail(path, 2**18)
        marker = text.rfind('"exception_info":')
        if marker >= 0:
            exception, _ = json.JSONDecoder().raw_decode(text[marker:].split(':', 1)[1].lstrip())
            row['exception_message'] = (exception or {}).get('exception_message', '')[-6000:]
        with path.open('rb') as stream:
            head = stream.read(4096).decode()
        task_marker = head.find('"task_id":')
        if task_marker >= 0:
            task_id, _ = json.JSONDecoder().raw_decode(head[task_marker:].split(':', 1)[1].lstrip())
            row['task_path'] = task_id.get('path')
        match = re.search(r'"trial_relpath"\s*:\s*("(?:[^"\\]|\\.)*")', head)
        selected = path.parent.parent / json.loads(match[1]) if match else path.parent
        for filename in ('test-stdout.txt', 'test-stderr.txt'):
            log = selected / 'verifier' / filename
            if log.exists():
                row[filename] = tail(log, 6000)
    print(json.dumps({'source_snapshot_sha256': report['provenance']['source_snapshot_sha256'],
                      'unresolved_slots': len(rows), 'diagnostics': rows}, indent=2))


if __name__ == '__main__':
    main()
