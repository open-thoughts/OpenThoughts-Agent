"""Regression coverage for Iris capability-token duration enforcement."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hpc.iris.capability_tokens import (
    controller_max_endpoint_token_ttl_seconds,
    infer_harbor_agent,
    persist_token_duration_policy,
    resolve_token_duration_policy,
)


def test_resolver_reads_the_controller_owned_constant():
    module = SimpleNamespace(MAX_ENDPOINT_TOKEN_TTL_SECONDS=345)
    assert (
        controller_max_endpoint_token_ttl_seconds(module_loader=lambda _: module) == 345
    )


def test_terminus_two_keeps_unlimited_timeout_and_needs_no_controller_lookup():
    policy = resolve_token_duration_policy(
        agent="terminus-2",
        timeout_seconds=0,
        max_ttl_resolver=lambda: pytest.fail("terminus-2 must not resolve token TTL"),
    )
    assert not policy.token_required
    assert policy.effective_timeout_seconds == 0
    assert policy.controller_max_ttl_seconds is None


def test_token_required_agent_defaults_timeout_to_controller_maximum():
    policy = resolve_token_duration_policy(
        agent="opencode", timeout_seconds=0, max_ttl_resolver=lambda: 345
    )
    assert policy.token_required
    assert policy.effective_timeout_seconds == 345
    assert policy.effective_token_ttl_seconds == 345


def test_shorter_requested_token_bounds_the_job_too():
    policy = resolve_token_duration_policy(
        agent="openhands",
        timeout_seconds=0,
        requested_token_ttl_seconds=200,
        max_ttl_resolver=lambda: 345,
    )
    assert policy.effective_timeout_seconds == 200
    with pytest.raises(ValueError, match="cannot outlive"):
        resolve_token_duration_policy(
            agent="openhands",
            timeout_seconds=201,
            requested_token_ttl_seconds=200,
            max_ttl_resolver=lambda: 345,
        )


def test_rejects_token_ttl_above_controller_maximum():
    with pytest.raises(ValueError, match="exceeds Marin controller maximum"):
        resolve_token_duration_policy(
            agent="opencode",
            timeout_seconds=0,
            requested_token_ttl_seconds=346,
            max_ttl_resolver=lambda: 345,
        )


def test_infers_harbor_agent_and_persists_only_secret_free_policy(tmp_path):
    harbor = tmp_path / "harbor.yaml"
    harbor.write_text("agents:\n  - name: opencode\n")
    assert infer_harbor_agent(None, harbor) == "opencode"
    policy = resolve_token_duration_policy(
        agent="opencode", timeout_seconds=0, max_ttl_resolver=lambda: 345
    )
    saved = persist_token_duration_policy(job_name="job", policy=policy, root=tmp_path)
    payload = json.loads(saved.read_text())
    assert payload["effective_timeout_seconds"] == 345
    assert not {"token", "api_base", "endpoint", "url"} & set(payload)
