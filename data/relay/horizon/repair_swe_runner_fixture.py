"""Restore SWE-bench's non-errexit evaluation flow in copied Harbor tasks.

SWE-bench 4.0.3 TestSpec.eval_script intentionally omits shell errexit. The
converted Harbor wrapper accidentally enables it, preventing the official test
command and parser from evaluating agent-corrupted projects. Official parsing
and resolution predicates stay unchanged; only its completed report writes the
Harbor reward, so a failed parser process does not become a fabricated zero.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HEADER = "#!/bin/bash\nset -euo pipefail\n"
REPORT = 'json.dump(report_map, open("/logs/verifier/report.json", "w"), indent=4)'
REWARD = """# Reward logging for Harbor.
mkdir -p /logs/verifier
if [ "${exit_code}" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def repaired_script(original: str) -> str:
    if not original.startswith(HEADER):
        raise ValueError("Expected converted SWE-bench errexit header")
    if original.count(REPORT) != 1 or original.count(REWARD) != 1:
        raise ValueError("Expected one pinned SWE report and Harbor reward block")
    if 'dependencies = ["swebench==4.0.3"' not in original:
        raise ValueError("Only SWE-bench 4.0.3 converted fixtures are supported")
    result = original.replace(HEADER, "#!/bin/bash\nset +e\nset -uo pipefail\n", 1)
    result = result.replace(
        REPORT,
        REPORT + '\nwith open("/logs/verifier/reward.txt", "w") as reward:\n'
        '    reward.write("1\\n" if report_map[instance_id]["resolved"] else "0\\n")',
        1,
    )
    result = result.replace(REWARD, "", 1)
    result = result.replace(
        "# Run parser and capture its exit code.",
        "mkdir -p /logs/verifier\n\n# Run parser and capture its exit code.",
        1,
    )
    return result


def repair(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    original = (source / "tests/test.sh").read_text()
    output = repaired_script(original)
    shutil.copytree(source, destination)
    (destination / "tests/test.sh").write_text(output)
    receipt = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "original_test_sh_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "repaired_test_sh_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "official_parser_version": "swebench==4.0.3",
        "official_test_commands_and_resolution_predicates_unchanged": True,
        "corrupted_project_files_not_restored": True,
        "reward_requires_completed_official_report": True,
    }
    (destination / "swe_runner_repair.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    repair(args.source, args.destination)
