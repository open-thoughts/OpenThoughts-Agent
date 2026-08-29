"""Deterministic ToolScale task construction.

ToolScale v3 was reconstructed from TaskTrove v3.42 and the pinned NVIDIA
source.  It selected the 4,048 source rows with at least one action and embedded
a domain-wide tool catalog, but its runtime synthesized every tool result from
the same per-task ``communicate_info`` strings.  It also had no discovery path
for identifiers that appeared only in the reference actions.

This module retains the v3 source pin and row-selection rule while defining the
v4 execution contract.  The catalog remains domain-wide, ``inspect`` exposes
candidate argument values without naming or grouping the reference calls, and
``call`` accepts only fixture-backed calls.  The runtime and verifier share the
same canonical call representation.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

SOURCE_REPO = "nvidia/ToolScale"
SOURCE_REVISION = "1ff421d53450d22fc2779ff19ffc502ce5582dc8"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
V3_TASKTROVE_REVISION = "3e96fe6464ce5ab6209e98801caab29b4a1fe87a"


def json_value(value: Any) -> Any:
    """Convert Arrow values into stable JSON values."""
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def clean_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop null Arrow fields from one ToolScale action."""
    return dict(json_value(arguments or {}))


def selected_by_v3(row: Mapping[str, Any]) -> bool:
    """Return whether the exact v3 generator retained a source row."""
    criteria = row.get("evaluation_criteria") or {}
    return bool(criteria.get("actions"))


def source_actions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return canonical reference calls from a ToolScale source row."""
    criteria = row.get("evaluation_criteria") or {}
    calls = []
    for action in criteria.get("actions") or []:
        calls.append({"name": action["name"], "arguments": clean_arguments(action.get("arguments"))})
    return calls


def source_assertions(row: Mapping[str, Any]) -> list[str]:
    """Return nonempty source assertions used as deterministic result evidence."""
    criteria = row.get("evaluation_criteria") or {}
    return [str(item).strip() for item in criteria.get("nl_assertions") or [] if str(item).strip()]


def task_domain(row: Mapping[str, Any]) -> str:
    scenario = row.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    return str(instructions.get("domain") or "general")


def build_domain_catalog(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Build domain-wide tool schemas without exposing per-task reference calls."""
    catalog: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        if not selected_by_v3(row):
            continue
        domain = task_domain(row)
        domain_tools = catalog.setdefault(domain, {})
        for call in source_actions(row):
            domain_tools.setdefault(call["name"], set()).update(call["arguments"])
    return {
        domain: {name: sorted(arguments) for name, arguments in sorted(tools.items())}
        for domain, tools in sorted(catalog.items())
    }


def canonical(value: Any) -> str:
    return json.dumps(json_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_id(task_id: str, tool: str, arguments: Mapping[str, Any]) -> str:
    payload = canonical({"arguments": arguments, "task_id": task_id, "tool": tool})
    return "evidence-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def expected_fixture(row: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    calls = source_actions(row)
    return {
        "assertions": source_assertions(row),
        "calls": [
            {
                **call,
                "evidence_id": evidence_id(task_id, call["name"], call["arguments"]),
            }
            for call in calls
        ],
        "domain": task_domain(row),
        "task_id": task_id,
    }


def discovery_candidates(calls: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Flatten argument values without retaining their call grouping."""
    candidates: dict[str, list[Any]] = {}
    for call in calls:
        for key, value in call["arguments"].items():
            values = candidates.setdefault(key, [])
            if value not in values:
                values.append(value)
    return candidates


def render_instruction(row: Mapping[str, Any], task_id: str) -> str:
    scenario = row.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    criteria = row.get("evaluation_criteria") or {}
    lines = ["# ToolScale deterministic tool-use task", "", f"- **Domain:** {task_domain(row)}"]
    if row.get("id"):
        lines.append(f"- **Scenario ID:** {row['id']}")
    headings = (
        ("Task Instructions", "task_instructions"),
        ("Reason for Request", "reason_for_call"),
        ("Known Information", "known_info"),
        ("Unknown Information", "unknown_info"),
    )
    for heading, key in headings:
        value = str(instructions.get(key) or "").strip()
        if value:
            lines.extend(("", f"## {heading}", "", value))
    assertions = source_assertions(row)
    if assertions:
        lines.extend(("", "## Success Criteria", ""))
        lines.extend(f"- {assertion}" for assertion in assertions)
    communicate_info = [str(item) for item in criteria.get("communicate_info") or []]
    if communicate_info:
        lines.extend(("", "## Information to Surface", ""))
        lines.extend(f"- {item}" for item in communicate_info)
    lines.extend(
        (
            "",
            "## Available tools",
            "",
            f"Inspect the {task_domain(row)} tool catalog with:",
            "",
            f"`toolscale catalog --domain {task_domain(row)}`",
            "",
            "Discover scenario-specific candidate argument values with:",
            "",
            f"`toolscale inspect --task-id {task_id}`",
            "",
            "The inspection output does not identify or group the required calls. Use the scenario and catalog to choose tools.",
            "",
            "Invoke a tool with:",
            "",
            f"`toolscale call --task-id {task_id} --tool TOOL_NAME --arguments 'JSON_OBJECT'`",
            "",
            "Only fixture-backed calls succeed. Successful calls return deterministic, task-specific evidence.",
            "",
            "## Deliverable",
            "",
            "Write a concise client-ready answer to `/app/response.md`. Include each relevant returned fact and the `evidence_id` from every tool result you rely on. The verifier checks executed calls and the response; a call plan without execution does not pass.",
        )
    )
    return "\n".join(lines).strip() + "\n"


def render_runtime(fixture: Mapping[str, Any], domain_catalog: Mapping[str, Sequence[str]]) -> str:
    fixture_json = json.dumps(fixture, ensure_ascii=False, sort_keys=True)
    catalog_json = json.dumps(dict(domain_catalog), ensure_ascii=False, sort_keys=True)
    candidates_json = json.dumps(discovery_candidates(fixture["calls"]), ensure_ascii=False, sort_keys=True)
    return f'''#!/usr/bin/env python3
"""Offline runtime for one ToolScale task."""

import argparse
import hashlib
import json
import os
from pathlib import Path

FIXTURE = {fixture_json}
CATALOG = {catalog_json}
CANDIDATES = {candidates_json}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_id(task_id, tool, arguments):
    payload = canonical({{"arguments": arguments, "task_id": task_id, "tool": tool}})
    return "evidence-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def catalog_command(domain):
    if domain != FIXTURE["domain"]:
        raise SystemExit(f"catalog unavailable for domain: {{domain}}")
    print(json.dumps({{name: {{"arguments": args}} for name, args in CATALOG.items()}}, indent=2, sort_keys=True))


def inspect_command(task_id):
    if task_id != FIXTURE["task_id"]:
        raise SystemExit(f"unknown task id: {{task_id}}")
    print(json.dumps({{"argument_candidates": CANDIDATES, "domain": FIXTURE["domain"], "task_id": task_id}}, indent=2, sort_keys=True))


def call_command(task_id, tool, arguments_text):
    if task_id != FIXTURE["task_id"]:
        raise SystemExit(f"unknown task id: {{task_id}}")
    arguments = json.loads(arguments_text)
    if not isinstance(arguments, dict):
        raise SystemExit("--arguments must decode to a JSON object")
    matches = [call for call in FIXTURE["calls"] if call["name"] == tool and call["arguments"] == arguments]
    if not matches:
        raise SystemExit("no fixture-backed result for that tool and argument object")
    call = matches[0]
    assert call["evidence_id"] == evidence_id(task_id, tool, arguments)
    index = FIXTURE["calls"].index(call)
    result = {{
        "arguments": arguments,
        "evidence_id": call["evidence_id"],
        "facts": {{"source_assertion": FIXTURE["assertions"][index]}} if index < len(FIXTURE["assertions"]) else {{}},
        "operation": tool,
        "status": "completed",
        "task_id": task_id,
    }}
    log_path = Path(os.environ.get("TOOL_SCALE_LOG", "/app/toolscale_calls.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(result) + "\\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    catalog_parser = subparsers.add_parser("catalog")
    catalog_parser.add_argument("--domain", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--task-id", required=True)
    call_parser = subparsers.add_parser("call")
    call_parser.add_argument("--task-id", required=True)
    call_parser.add_argument("--tool", required=True)
    call_parser.add_argument("--arguments", required=True)
    args = parser.parse_args()
    if args.command == "catalog":
        catalog_command(args.domain)
    elif args.command == "inspect":
        inspect_command(args.task_id)
    else:
        call_command(args.task_id, args.tool, args.arguments)


if __name__ == "__main__":
    main()
'''


def render_check() -> str:
    return '''#!/usr/bin/env python3
import json
import os
from pathlib import Path

tests_dir = Path(os.environ.get("TOOL_SCALE_TESTS_DIR", "/tests"))
app_dir = Path(os.environ.get("APP_DIR", "/app"))
expected = json.loads((tests_dir / "expected.json").read_text())
log_path = app_dir / "toolscale_calls.jsonl"
response_path = app_dir / "response.md"
assert log_path.is_file(), "no tool calls were executed"
assert response_path.is_file(), "missing /app/response.md"
logs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
for wanted in expected["calls"]:
    matches = [
        item for item in logs
        if item.get("task_id") == expected["task_id"]
        and item.get("operation") == wanted["name"]
        and item.get("arguments") == wanted["arguments"]
        and item.get("evidence_id") == wanted["evidence_id"]
    ]
    assert matches, f"required fixture-backed call was not executed: {wanted['name']}"
response = response_path.read_text().casefold()
for wanted in expected["calls"]:
    assert wanted["evidence_id"].casefold() in response, f"response omits {wanted['evidence_id']}"
for assertion in expected["assertions"]:
    assert assertion.casefold() in response, f"response omits source assertion: {assertion}"
print(f"validated {len(expected['calls'])} fixture-backed calls")
'''


def render_test_sh() -> str:
    return '''#!/bin/bash
set -uo pipefail
TESTS_DIR="${TOOL_SCALE_TESTS_DIR:-/tests}"
LOGS_DIR="${TOOL_SCALE_LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS_DIR"
echo 0 > "$LOGS_DIR/reward.txt"
if python3 "$TESTS_DIR/check.py" 2>&1 | tee "$LOGS_DIR/test_output.txt"; then
    echo 1 > "$LOGS_DIR/reward.txt"
    exit 0
fi
exit 1
'''


def render_solution(fixture: Mapping[str, Any]) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'APP_DIR="${APP_DIR:-/app}"',
        'mkdir -p "$APP_DIR"',
        'rm -f "$APP_DIR/toolscale_calls.jsonl"',
        'response="$APP_DIR/response.md"',
        ': > "$response"',
        f"printf '%s\\n' {shlex.quote('# ToolScale result for ' + fixture['task_id'])} >> \"$response\"",
    ]
    for call in fixture["calls"]:
        args = canonical(call["arguments"])
        command = (
            f"toolscale call --task-id {shlex.quote(fixture['task_id'])} "
            f"--tool {shlex.quote(call['name'])} --arguments {shlex.quote(args)}"
        )
        lines.append(command + " > /tmp/toolscale-result.json")
        lines.append(f"printf '%s\\n' {shlex.quote('- Evidence: ' + call['evidence_id'])} >> \"$response\"")
    for assertion in fixture["assertions"]:
        lines.append(f"printf '%s\\n' {shlex.quote('- ' + assertion)} >> \"$response\"")
    return "\n".join(lines) + "\n"


DOCKERFILE = '''FROM python:3.12-slim
WORKDIR /app
COPY toolscale_runtime.py /usr/local/bin/toolscale
RUN chmod +x /usr/local/bin/toolscale
CMD ["/bin/bash"]
'''


TASK_TOML = '''version = "1.0"

[agent]
timeout_sec = 900.0

[metadata]
author_name = "OpenThoughts-Agent"
author_email = ""
difficulty = "medium"
category = "tool-use"
tags = ["sandbox", "tool-use", "deterministic", "fixture-backed"]

[verifier]
restart_environment = false
timeout_sec = 720.0
'''
