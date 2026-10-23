"""Retrieve and checksum the final small repaired readout onto the Mac."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
REMOTE = '/scratch/11584/lukedhlee/experiments/cf2-repaired-readout-20261023'
MEMORY = Path('/Users/lukedhlee/OpenThoughts-Agent/ai_memory/notes/snowball_cf2_repaired_readout_20261023.md')
DEADLINE = datetime.datetime(2026, 10, 23, 18, 0, tzinfo=datetime.timezone.utc)


def main():
    lock = (ROOT / 'pull.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while datetime.datetime.now(datetime.timezone.utc) < DEADLINE:
        try:
            status = subprocess.check_output(['ssh', 'horizon', f'cat {REMOTE}/STATUS'],
                                             text=True, timeout=45)
            if 'FINAL_READY' in status:
                staging = ROOT / 'final_pull.tmp'
                staging.mkdir(exist_ok=True)
                subprocess.run(['scp', f'horizon:{REMOTE}/SHA256SUMS.json', str(staging)],
                               check=True, timeout=60)
                sums = json.loads((staging / 'SHA256SUMS.json').read_text())
                # Source code is already local; fetch only the sealed result artifacts.
                names = [name for name in sums if not name.endswith('.py')]
                assert all('/' not in name and '..' not in name for name in names)
                subprocess.run(['scp', *[f'horizon:{REMOTE}/{name}' for name in names], str(staging)],
                               check=True, timeout=120)
                for name in names:
                    assert hashlib.sha256((staging / name).read_bytes()).hexdigest() == sums[name], name
                for path in staging.iterdir():
                    path.replace(ROOT / path.name)
                (ROOT / 'STATUS').write_text(status)
                report = json.loads((ROOT / 'report.json').read_text())
                receipt = {'pulled_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           'remote': REMOTE, 'files_checked': len(names),
                           'source_snapshot_sha256': report['provenance']['source_snapshot_sha256'],
                           'summary': report['summary']}
                (ROOT / 'FINAL_PULL_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
                MEMORY.write_text(
                    '# Snowball corrected-stack repaired readout\n\n'
                    'Final validated results are ready locally and on Horizon. Original scores remain '
                    'unchanged; repaired scores are a separate sensitivity analysis. Git-forensics is '
                    'excluded from strict panels because the original fixture is absent.\n\n'
                    f"Snapshot: {report['snapshot_utc']}. Valid numeric coverage "
                    f"{report['summary']['repaired_valid_numeric']}/{report['summary']['strict_valid_slots']}; "
                    f"{report['summary']['unscored_valid']} remain unscored.\n\n"
                    f'Artifacts: `{ROOT}` and `{REMOTE}`. Read report.md/report.json, '
                    'comparisons.csv, coverage.csv, unresolved.csv, provenance.json.\n\n'
                    f"Source SHA256: `{report['provenance']['source_snapshot_sha256']}`. "
                    f"Final pull verified {len(names)} file checksums.\n")
                shutil.rmtree(staging)
                print(json.dumps(receipt), flush=True)
                return
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), status.strip(), flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
        time.sleep(900)
    (ROOT / 'PULL_DEADLINE_STATUS').write_text('Deadline reached; inspect pull.out/remote STATUS.\n')


if __name__ == '__main__':
    main()
