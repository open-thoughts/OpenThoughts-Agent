"""Refresh validated repaired comparisons until every recovery root has settled."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
DEADLINE = datetime.datetime(2026, 10, 23, 17, 45, tzinfo=datetime.timezone.utc)


def main():
    lock = (ROOT / 'watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = 0
    while datetime.datetime.now(datetime.timezone.utc) < DEADLINE:
        attempt += 1
        try:
            staging = ROOT / 'refresh.tmp'
            staging.mkdir(exist_ok=True)
            with (staging / 'data_snapshot.json').open('w') as output:
                subprocess.run([sys.executable, str(ROOT / 'collect.py')], stdout=output,
                               check=True, timeout=240)
            subprocess.run([sys.executable, str(ROOT / 'analyze.py'), '--snapshot',
                            str(staging / 'data_snapshot.json'), '--output', str(staging)],
                           check=True, timeout=180)
            status = (staging / 'STATUS').read_text()
            sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in staging.iterdir() if p.is_file() and p.name not in ('STATUS', 'SHA256SUMS.json')}
            sums['collect.py'] = hashlib.sha256((ROOT / 'collect.py').read_bytes()).hexdigest()
            sums['analyze.py'] = hashlib.sha256((ROOT / 'analyze.py').read_bytes()).hexdigest()
            (staging / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2)+'\n')
            for path in staging.iterdir():
                if path.is_file() and path.name != 'STATUS':
                    path.replace(ROOT / path.name)
            (staging / 'STATUS').replace(ROOT / 'STATUS')
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'refresh', attempt, status.strip(), flush=True)
            if 'FINAL_READY' in status:
                shutil.rmtree(staging)
                return
        except Exception:
            error = traceback.format_exc()
            (ROOT / 'last_refresh_error.txt').write_text(error)
            print(error, flush=True)
        # This runs in a detached tmux session, leaving operator tools responsive.
        time.sleep(900)
    (ROOT / 'DEADLINE_STATUS').write_text('Deadline reached; inspect STATUS and watch.out.\n')


if __name__ == '__main__':
    main()
