"""Calendar verifier with exact event-set and scheduling constraints."""

VERIFIER_PY = r'''#!/usr/bin/env python3
"""Validate a complete calendar written to /app/answer.txt."""
from __future__ import annotations

import json
import pathlib
import re
import sys
import unicodedata

REWARD = pathlib.Path("/logs/verifier/reward.txt")
ANSWER = pathlib.Path("/app/answer.txt")
DATA = pathlib.Path("/tests/verifier_data.json")


def _write_reward(score: int) -> None:
    REWARD.parent.mkdir(parents=True, exist_ok=True)
    REWARD.write_text("1" if score else "0")


def _normalized_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    return normalized or None


def _parse_time(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d{2}):(\d{2})", value.strip())
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _clock_minutes(hour: str, minute: str | None, ampm: str | None) -> int | None:
    h = int(hour)
    m = int(minute or 0)
    if not (0 <= m <= 59):
        return None
    if ampm is None:
        if not (0 <= h <= 23):
            return None
        return h * 60 + m
    if not (1 <= h <= 12):
        return None
    return (h % 12 + (12 if ampm == "pm" else 0)) * 60 + m


def _check_constraint(constraint: object, start: int, end: int) -> bool:
    if constraint is None or constraint == "":
        return True
    if not isinstance(constraint, str):
        return False
    text = " ".join(constraint.strip().lower().split())
    clock = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    match = re.fullmatch(rf"before\s+{clock}", text)
    if match is not None:
        limit = _clock_minutes(*match.groups())
        return limit is not None and end <= limit
    match = re.fullmatch(rf"after\s+{clock}", text)
    if match is not None:
        limit = _clock_minutes(*match.groups())
        return limit is not None and start >= limit
    match = re.fullmatch(rf"at\s+{clock}", text)
    if match is not None:
        exact = _clock_minutes(*match.groups())
        return exact is not None and start == exact
    match = re.fullmatch(rf"between\s+{clock}\s+and\s+{clock}", text)
    if match is not None:
        groups = match.groups()
        lower = _clock_minutes(*groups[:3])
        upper = _clock_minutes(*groups[3:])
        return lower is not None and upper is not None and start >= lower and end <= upper
    return False


def evaluate_calendar(expected: object, events: object) -> tuple[bool, list[str]]:
    """Evaluate the externally visible calendar contract."""
    errors: list[str] = []
    if not isinstance(expected, dict) or not expected:
        return False, ["expected_events missing"]
    if not isinstance(events, list):
        return False, ["answer must be a JSON list"]

    actual_by_id: dict[int, dict] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event at index {index} is not an object")
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            errors.append(f"event at index {index} has invalid event_id")
            continue
        if event_id in actual_by_id:
            errors.append(f"duplicate event_id {event_id}")
            continue
        actual_by_id[event_id] = event

    expected_ids: set[int] = set()
    for key, spec in expected.items():
        if not isinstance(spec, dict):
            errors.append(f"expected event {key!r} is not an object")
            continue
        try:
            event_id = int(key)
        except (TypeError, ValueError):
            errors.append(f"expected event key {key!r} is invalid")
            continue
        if spec.get("event_id") != event_id:
            errors.append(f"expected event {event_id} has inconsistent event_id")
        expected_ids.add(event_id)

    for event_id in sorted(expected_ids - actual_by_id.keys()):
        errors.append(f"missing event_id {event_id}")
    for event_id in sorted(actual_by_id.keys() - expected_ids):
        errors.append(f"unexpected event_id {event_id}")

    intervals: list[tuple[int, int, int]] = []
    for event_id in sorted(expected_ids & actual_by_id.keys()):
        spec = expected[str(event_id)] if str(event_id) in expected else expected[event_id]
        actual = actual_by_id[event_id]
        expected_name = _normalized_name(spec.get("event_name"))
        actual_name = _normalized_name(actual.get("event_name"))
        if expected_name is None or actual_name != expected_name:
            errors.append(f"event {event_id} name mismatch")

        duration = actual.get("duration")
        expected_duration = spec.get("duration")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration <= 0
            or duration != expected_duration
        ):
            errors.append(f"event {event_id} duration mismatch")
            continue
        start = _parse_time(actual.get("start_time"))
        minimum = _parse_time(spec.get("min_time"))
        maximum = _parse_time(spec.get("max_time"))
        if start is None or minimum is None or maximum is None:
            errors.append(f"event {event_id} has invalid time data")
            continue
        end = start + duration
        if start < minimum or end > maximum:
            errors.append(f"event {event_id} is outside its allowed window")
        if not _check_constraint(spec.get("constraint"), start, end):
            errors.append(f"event {event_id} violates its declared constraint")
        intervals.append((start, end, event_id))

    intervals.sort()
    for index, previous in enumerate(intervals):
        for current in intervals[index + 1 :]:
            if current[0] >= previous[1]:
                break
            previous_start, previous_end, previous_id = previous
            current_start, current_end, current_id = current
            errors.append(
                f"events {previous_id} "
                f"[{previous_start // 60:02d}:{previous_start % 60:02d}, "
                f"{previous_end // 60:02d}:{previous_end % 60:02d}) and "
                f"{current_id} "
                f"[{current_start // 60:02d}:{current_start % 60:02d}, "
                f"{current_end // 60:02d}:{current_end % 60:02d}) overlap"
            )
    return not errors, errors


def _answer_json(raw: str) -> object:
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    return json.loads(fence.group(1) if fence else raw)


def main() -> int:
    data = json.loads(DATA.read_text())
    expected = data.get("expected_events")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("expected_events missing from verifier data")
    if not ANSWER.exists():
        print("/app/answer.txt missing", file=sys.stderr)
        _write_reward(0)
        return 0
    try:
        events = _answer_json(ANSWER.read_text(errors="replace"))
    except json.JSONDecodeError as error:
        print(f"answer parse error: {error}", file=sys.stderr)
        _write_reward(0)
        return 0
    valid, errors = evaluate_calendar(expected, events)
    for error in errors:
        print(error)
    score = int(valid)
    _write_reward(score)
    return score


if __name__ == "__main__":
    main()
'''
