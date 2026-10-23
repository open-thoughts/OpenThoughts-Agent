"""Copy the Requests task and optionally repair its verifier's network fixtures.

The tests assume TCP port 1 is closed and 10.255.255.1:80 silently drops SYNs.
Cloud egress often drops the former and rejects the latter. Enforce those two
fixtures in the sandbox's network namespace only while the original grader runs.
No agent files, test assertions, reward parser, or time budgets are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

WRAPPER = r'''#!/bin/bash
set -euo pipefail
# Verification only: this script is uploaded after the agent has finished.
# Keep the original Requests tests/parser byte-identical in test-original.sh.
if ! command -v iptables >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y iptables
fi
chain="CF2_REQUESTS_$$"
iptables -N "$chain"
cleanup() {
    iptables -D OUTPUT -j "$chain" 2>/dev/null || true
    iptables -F "$chain" 2>/dev/null || true
    iptables -X "$chain" 2>/dev/null || true
}
trap cleanup EXIT
# The timeout tests explicitly use this address as a nonresponsive tarpit.
iptables -A "$chain" -p tcp -d 10.255.255.1 --dport 80 -j DROP
# The connection-error test explicitly expects no service on port 1.
iptables -A "$chain" -p tcp --dport 1 -j REJECT --reject-with tcp-reset
iptables -I OUTPUT 1 -j "$chain"
bash /tests/test-original.sh
'''


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--network-fixture', action='store_true',
                        help='Enable the verifier-only closed-port/tarpit fixture')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    assert source.name == destination.name == 'psf__requests-2317'
    assert not destination.exists(), 'Refuse to overwrite a task copy'
    datum = json.loads((source / 'tests/config.json').read_text())
    assert datum['instance_id'] == 'psf__requests-2317'
    assert not (source / 'tests/test-original.sh').exists()
    before = hashes(source)
    shutil.copytree(source, destination, symlinks=True)
    assert hashes(destination) == before, 'Task copy must be byte-identical before repair'
    if args.network_fixture:
        (destination / 'tests/test-original.sh').write_bytes((source / 'tests/test.sh').read_bytes())
        (destination / 'tests/test.sh').write_text(WRAPPER)
    after = hashes(destination)
    changed = sorted(k for k in before if before[k] != after[k])
    added = sorted(set(after) - set(before))
    assert changed == (['tests/test.sh'] if args.network_fixture else [])
    assert added == (['tests/test-original.sh'] if args.network_fixture else [])
    assert hashes(source) == before, 'Source task must remain immutable'
    print(json.dumps(dict(source=str(source), destination=str(destination),
                          network_fixture=args.network_fixture, changed=changed, added=added,
                          original_sha256=before, repaired_sha256=after), indent=2))


if __name__ == '__main__':
    main()
