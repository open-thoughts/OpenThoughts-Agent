"""Harbor-aware resume policy manager for ``hpc.launch`` (datagen + eval).

Replaces the silent ``experiments/<name>_2`` dedup in
``hpc/launch_utils.py:setup_experiments_dir`` with explicit, flag-driven
policy on whether to resume, mutate, wipe, or bail when an existing job
dir is detected. SFT / RL / consolidate job types fall through this
module entirely (they don't go through Harbor's trial-level resume).

Decision matrix (after the existing dir's drift has been classified):

| ``--force_mutate`` | ``--allow_overwrite`` | Behavior on drift           |
|--------------------|-----------------------|-----------------------------|
| true               | true                  | Mutate, fall back to wipe   |
| true               | false                 | Mutate, bail on plan fail   |
| false              | true                  | Wipe + warn                 |
| false              | false                 | **Default.** Bail with diff |
| (any combo, drift absent)                  | Resume clean               |

Public entrypoints:

  - ``resolve_resume_policy_for_launch(exp_args, *, job_name)`` — top-level
    seam called from ``hpc/launch_utils.py:resolve_job_and_paths``. Returns
    ``None`` when the manager declines to engage (no prior dir, non-Harbor
    job type), or a :class:`ResumePolicyResult` describing the action taken.
    Raises :class:`ResumeBail` if the matrix says bail.
  - ``inspect_resume`` / ``plan_mutation`` / ``apply_mutation`` / ``wipe_job_dir``
    / ``render_bail_message`` — read-only / planning / side-effecting
    primitives, exposed for the standalone CLI and tests.

The manager works at the **dict level** (loaded JSON), not against
Harbor's Pydantic models, to avoid version drift across Harbor branches.
The contract with Harbor is the on-disk shape of ``config.json`` and
per-trial ``config.json``, which has been stable across recent versions.
"""

from __future__ import annotations

import copy
import fcntl
import json
import shutil
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Public enums / dataclasses / exceptions
# -----------------------------------------------------------------------------


class ResumeState(str, Enum):
    NO_PRIOR = "NO_PRIOR"
    CLEAN_RESUME = "CLEAN_RESUME"
    MUTABLE_DRIFT = "MUTABLE_DRIFT"
    FATAL_DRIFT = "FATAL_DRIFT"
    CORRUPT = "CORRUPT"
    ALREADY_COMPLETE = "ALREADY_COMPLETE"


class ResumeAction(str, Enum):
    NO_ENGAGEMENT = "NO_ENGAGEMENT"
    CLEAN_RESUME = "CLEAN_RESUME"
    MUTATE_AND_RESUME = "MUTATE_AND_RESUME"
    WIPE_AND_FRESH = "WIPE_AND_FRESH"


@dataclass
class ConfigFieldDrift:
    """A single field-level diff between prior on-disk and new planned config."""
    path: Tuple[Any, ...]
    old: Any
    new: Any
    note: str = ""  # human-readable annotation (e.g. "synthetic vLLM ID rotation")


@dataclass
class TrialConfigDrift:
    trial_name: str
    drifts: List[ConfigFieldDrift]
    fatal: bool = False
    orphan: bool = False   # existing trial whose agent.name no longer in plan


@dataclass
class CoverageStats:
    n_existing: int = 0                       # total existing trial dirs
    n_with_result: int = 0
    n_without_result: int = 0
    n_orphans: int = 0
    by_exception_type: Dict[str, int] = field(default_factory=dict)
    reward_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class ResumeReport:
    job_dir: Path
    state: ResumeState
    coverage: Optional[CoverageStats] = None
    job_config_drift: List[ConfigFieldDrift] = field(default_factory=list)
    trial_drifts: List[TrialConfigDrift] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    corrupt_trial_dirs: List[Path] = field(default_factory=list)
    has_fatal_drift: bool = False


@dataclass
class MutationPlan:
    new_top_level_config_json: str
    trial_config_rewrites: List[Tuple[Path, str]] = field(default_factory=list)
    trial_dirs_to_wipe: List[Path] = field(default_factory=list)
    trial_results_to_quarantine: List[Path] = field(default_factory=list)

    @property
    def estimated_changes(self) -> int:
        return (
            1
            + len(self.trial_config_rewrites)
            + len(self.trial_dirs_to_wipe)
            + len(self.trial_results_to_quarantine)
        )


@dataclass
class ResumePolicyResult:
    action: ResumeAction
    report: ResumeReport
    plan: Optional[MutationPlan] = None


class ResumeBail(Exception):
    """Raised when the resume manager's policy matrix says bail.

    Carries the pre-rendered actionable message for the top-level
    launcher handler to print before exiting.
    """

    def __init__(self, report: ResumeReport, message: str):
        super().__init__(message)
        self.report = report
        self.message = message


# -----------------------------------------------------------------------------
# Mutation allowlist
# -----------------------------------------------------------------------------

# Top-level JobConfig fields that can be patched in place. Anything else is
# fatal. Note: ``orchestrator.n_concurrent_trials`` (legacy nested) and the
# flat ``n_concurrent_trials`` are both recognized — Harbor's
# ``_migrate_orchestrator_config`` validator normalizes them on load.
_MUTABLE_JOB_TOP_LEVEL = {
    "n_concurrent_trials",
    "quiet",
    "debug",
    "timeout_multiplier",
    "agent_timeout_multiplier",
    "verifier_timeout_multiplier",
    "agent_setup_timeout_multiplier",
    "environment_build_timeout_multiplier",
    "retry",
}
_MUTABLE_JOB_NESTED_ROOTS = {
    ("orchestrator",),     # whole orchestrator subtree
    ("retry",),            # whole retry subtree (mirror)
}

# Per-trial config.json: path prefixes that may be rewritten on mutation.
_MUTABLE_TRIAL_PREFIXES = {
    ("agent", "kwargs", "api_base"),
    ("agent", "kwargs", "api_key"),
    ("agent", "kwargs", "metrics_endpoint"),
    ("agent", "kwargs", "base_url"),
    ("timeout_multiplier",),
    ("agent_timeout_multiplier",),
    ("verifier_timeout_multiplier",),
    ("agent_setup_timeout_multiplier",),
    ("environment_build_timeout_multiplier",),
}

_HOSTED_VLLM_PREFIX = "hosted_vllm/"


def _is_synthetic_vllm_id_rotation(drift: ConfigFieldDrift) -> bool:
    """A ``model_name`` drift is mutable when EITHER side is ``hosted_vllm/<digits>``.

    ``hpc.launch_utils.generate_served_model_id(job_name=...)`` emits a
    deterministic-per-job ID (so chain-restarts produce the same synthetic
    ID), but legacy artifacts from older launches contain the prior
    timestamp-based IDs, AND prior resume_manager mutations may have
    rewritten the on-disk ``config.json`` with the resolved canonical HF
    name. Either directionality represents "same serving identity" since
    Harbor resolves at runtime regardless of which form is on disk:

      - OLD=synthetic, NEW=synthetic: ID rotation across launches
        (timestamp legacy) or noop after determinism land.
      - OLD=real, NEW=synthetic: prior resume_manager mutation rewrote
        config.json to the canonical HF name; the new launch's YAML has
        the synthetic alias. This is the common failure mode the
        deterministic-ID change fixes — keep the predicate symmetric so
        legacy on-disk state still resolves cleanly.
      - OLD=synthetic, NEW=real: planner falls back to the HF name (rare
        — happens when ``requires_vllm`` is False at materialization
        time).

    A real model swap (BOTH sides are real HF repo names) stays fatal —
    that's an actual semantic change the user should acknowledge.
    """
    if not drift.path or drift.path[-1] != "model_name":
        return False
    # Accept both per-trial shape ("agent", "model_name") and top-level shape
    # ("agents", <idx>, "model_name").
    old, new = drift.old, drift.new
    if not (isinstance(old, str) and isinstance(new, str)):
        return False

    def _is_synthetic(s: str) -> bool:
        if not s.startswith(_HOSTED_VLLM_PREFIX):
            return False
        return s[len(_HOSTED_VLLM_PREFIX):].isdigit()

    return _is_synthetic(old) or _is_synthetic(new)


# Predicates that override the allowlist on a per-drift basis. Each returns
# True iff the drift is structurally safe to patch (not a real semantic change).
_SPECIAL_CASE_MUTABLE_PREDICATES: List[Callable[[ConfigFieldDrift], bool]] = [
    _is_synthetic_vllm_id_rotation,
]


_MUTABLE_AGENT_KWARG_KEYS = {"api_base", "api_key", "metrics_endpoint", "base_url"}


def _is_mutable_top_level(drift: ConfigFieldDrift) -> bool:
    if not drift.path:
        return False
    head = drift.path[0]
    if head in _MUTABLE_JOB_TOP_LEVEL:
        return True
    if (head,) in _MUTABLE_JOB_NESTED_ROOTS:
        return True
    # ``agents[<idx>].kwargs.<allowlisted>`` is mutable at the top level too,
    # since the resume manager rewrites the whole config.json with the planned
    # config on mutation and patches each trial's agent.kwargs to match.
    if (
        head == "agents"
        and len(drift.path) >= 4
        and drift.path[2] == "kwargs"
        and drift.path[3] in _MUTABLE_AGENT_KWARG_KEYS
    ):
        return True
    for pred in _SPECIAL_CASE_MUTABLE_PREDICATES:
        if pred(drift):
            return True
    return False


def _is_mutable_trial_field(drift: ConfigFieldDrift) -> bool:
    for prefix in _MUTABLE_TRIAL_PREFIXES:
        if len(drift.path) >= len(prefix) and tuple(drift.path[: len(prefix)]) == prefix:
            return True
    for pred in _SPECIAL_CASE_MUTABLE_PREDICATES:
        if pred(drift):
            return True
    return False


# -----------------------------------------------------------------------------
# Dict diff utilities
# -----------------------------------------------------------------------------


def _is_effectively_empty(v: Any) -> bool:
    """True for values semantically equivalent to "key absent / default".

    Used to suppress spurious diffs when one config has the key with a
    falsy default (``None``, ``[]``, ``{}``, ``""``, ``False``) and the
    other simply omits the key. These pairs are not real semantic
    drifts and would otherwise dominate the bail message for
    materialized-from-CLI vs. on-disk Harbor configs.
    """
    if v is None:
        return True
    if isinstance(v, (list, dict, str)) and len(v) == 0:
        return True
    return False


# Top-level config paths whose list values should be compared as SETS, not
# ordered sequences. Harbor's retry exception filters are membership-based —
# their on-disk ordering is incidental and frequently differs from the
# Pydantic enum order in the planned config.
_ORDER_INSENSITIVE_LIST_PATHS: set[tuple[str, ...]] = {
    ("orchestrator", "retry", "exclude_exceptions"),
    ("orchestrator", "retry", "include_exceptions"),
    ("orchestrator", "retry", "mask_exceptions"),
    ("orchestrator", "retry", "passthrough_exceptions"),
    # Legacy / flat shape (some configs have these at the root).
    ("retry", "exclude_exceptions"),
    ("retry", "include_exceptions"),
}


def _diff_dicts(
    old: Any,
    new: Any,
    path: Tuple[Any, ...] = (),
) -> List[ConfigFieldDrift]:
    """Recursive dict/list/scalar diff, emitting leaf-level drifts.

    Compares ``old`` against ``new`` and yields one :class:`ConfigFieldDrift`
    per differing leaf. Lists are compared element-wise (or as sets if the
    path is in ``_ORDER_INSENSITIVE_LIST_PATHS``); length mismatches are
    reported as a single drift at the list path. Pairs where one side is
    missing and the other is effectively empty (None/[]/{}/"") are
    suppressed — they're not real semantic drifts.
    """
    drifts: List[ConfigFieldDrift] = []
    if type(old) is not type(new):
        if old != new:
            if _is_effectively_empty(old) and _is_effectively_empty(new):
                return drifts
            drifts.append(ConfigFieldDrift(path=path, old=old, new=new))
        return drifts
    if isinstance(old, dict):
        keys = set(old.keys()) | set(new.keys())
        for key in sorted(keys, key=lambda k: str(k)):
            sub_old = old.get(key, _MISSING)
            sub_new = new.get(key, _MISSING)
            if sub_old is _MISSING or sub_new is _MISSING:
                present = sub_new if sub_old is _MISSING else sub_old
                # Suppress: key absent on one side, falsy default on the other.
                if _is_effectively_empty(present):
                    continue
                # Suppress: planned config omits the key entirely
                # (sub_new is _MISSING) but on-disk has a value. Harbor
                # preserves the on-disk value on resume; the planned config
                # simply doesn't constrain it. Symmetric case
                # (sub_old is _MISSING and sub_new has a value) is a real
                # introduction and stays as a drift.
                if sub_new is _MISSING:
                    continue
                drifts.append(
                    ConfigFieldDrift(
                        path=path + (key,),
                        old=None if sub_old is _MISSING else sub_old,
                        new=None if sub_new is _MISSING else sub_new,
                    )
                )
            else:
                # Special case: planned config explicitly omitted (None) a
                # field that on-disk has set. The on-disk value is preserved
                # at runtime (Harbor reads its own config), so this is a
                # no-op semantically. Only suppress when planned-is-None —
                # None on disk vs a planned non-None is still a real drift.
                if sub_new is None and sub_old is not None:
                    continue
                drifts.extend(_diff_dicts(sub_old, sub_new, path + (key,)))
        return drifts
    if isinstance(old, list):
        # Order-insensitive comparison for known set-semantics lists.
        if path in _ORDER_INSENSITIVE_LIST_PATHS:
            try:
                if set(old) == set(new):
                    return drifts
            except TypeError:
                pass  # unhashable elements — fall through to ordered diff
        if len(old) != len(new):
            drifts.append(ConfigFieldDrift(path=path, old=old, new=new))
            return drifts
        for idx, (a, b) in enumerate(zip(old, new)):
            drifts.extend(_diff_dicts(a, b, path + (idx,)))
        return drifts
    if old != new:
        drifts.append(ConfigFieldDrift(path=path, old=old, new=new))
    return drifts


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def _annotate_drift(drift: ConfigFieldDrift) -> ConfigFieldDrift:
    """Attach human-readable notes to special-case drifts."""
    if _is_synthetic_vllm_id_rotation(drift):
        return ConfigFieldDrift(
            path=drift.path,
            old=drift.old,
            new=drift.new,
            note="synthetic vLLM ID rotation — same serving identity",
        )
    return drift


# -----------------------------------------------------------------------------
# Disk readers (defensive)
# -----------------------------------------------------------------------------


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read + parse a JSON file. Returns None on missing or unparseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _enumerate_trial_dirs(job_dir: Path) -> List[Path]:
    if not job_dir.exists() or not job_dir.is_dir():
        return []
    out: List[Path] = []
    for entry in sorted(job_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Trial dirs have a config.json inside; skip pure result/log directories.
        if (entry / "config.json").exists() or (entry / "result.json").exists():
            out.append(entry)
    return out


def _trial_result_summary(result: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Extract (exception_type, reward) from a parsed trial result blob."""
    exc = (result.get("exception_info") or {}).get("exception_type")
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    try:
        reward = float(reward) if reward is not None else None
    except (TypeError, ValueError):
        reward = None
    return exc, reward


def _reward_bucket(reward: Optional[float]) -> str:
    if reward is None:
        return "no_reward"
    if reward == 0.0:
        return "0.0"
    if reward == 1.0:
        return "1.0"
    return "(0.0, 1.0)"


# -----------------------------------------------------------------------------
# Inspection
# -----------------------------------------------------------------------------


def inspect_resume(
    job_dir: Path,
    planned_top_level_config: Dict[str, Any],
) -> ResumeReport:
    """Read disk + compare against the planned config. Side-effect-free.

    ``planned_top_level_config`` is the merged Harbor JobConfig dict that
    a fresh launch would produce (typically via
    ``hpc.harbor_utils.merge_harbor_config``).
    """
    job_dir = Path(job_dir)

    if not job_dir.exists():
        return ResumeReport(job_dir=job_dir, state=ResumeState.NO_PRIOR)

    config_path = job_dir / "config.json"
    prior_config = _safe_read_json(config_path)
    if prior_config is None and not config_path.exists():
        # Directory exists but Harbor never wrote a config.json. Treat as
        # leftover artifact, no resumable state.
        return ResumeReport(job_dir=job_dir, state=ResumeState.NO_PRIOR)
    if prior_config is None:
        # config.json exists but is corrupt.
        return ResumeReport(
            job_dir=job_dir,
            state=ResumeState.CORRUPT,
            corrupt_trial_dirs=[config_path],
        )

    # Build coverage stats from trial dirs.
    coverage = CoverageStats()
    trial_dirs = _enumerate_trial_dirs(job_dir)
    corrupt: List[Path] = []
    trial_drifts: List[TrialConfigDrift] = []
    has_fatal = False

    planned_agent_names = {
        agent.get("name")
        for agent in planned_top_level_config.get("agents", [])
        if isinstance(agent, dict)
    }
    planned_first_agent_kwargs = _first_agent_kwargs(planned_top_level_config)

    for trial_dir in trial_dirs:
        coverage.n_existing += 1
        trial_config = _safe_read_json(trial_dir / "config.json")
        trial_result_path = trial_dir / "result.json"
        trial_result = _safe_read_json(trial_result_path)

        if trial_result_path.exists() and trial_result is None:
            corrupt.append(trial_result_path)
            coverage.n_without_result += 1
        elif trial_result is None:
            coverage.n_without_result += 1
        else:
            coverage.n_with_result += 1
            exc, reward = _trial_result_summary(trial_result)
            if exc:
                coverage.by_exception_type[exc] = coverage.by_exception_type.get(exc, 0) + 1
            coverage.reward_distribution[_reward_bucket(reward)] = (
                coverage.reward_distribution.get(_reward_bucket(reward), 0) + 1
            )

        if trial_config is None:
            continue

        trial_agent_name = (trial_config.get("agent") or {}).get("name")
        if planned_agent_names and trial_agent_name not in planned_agent_names:
            coverage.n_orphans += 1
            trial_drifts.append(
                TrialConfigDrift(
                    trial_name=trial_dir.name,
                    drifts=[],
                    fatal=True,
                    orphan=True,
                )
            )
            has_fatal = True
            continue

        # Compare each allowlisted trial field against the planned top-level
        # equivalent. We only emit drifts for fields we know how to mutate
        # (api_base, model_name via the synthetic-ID predicate, timeouts);
        # other drifts are not tracked at the trial level since they can't
        # be safely patched.
        drifts = _diff_trial_against_plan(trial_config, planned_top_level_config, planned_first_agent_kwargs)
        if drifts:
            annotated = [_annotate_drift(d) for d in drifts]
            fatal = any(not _is_mutable_trial_field(d) for d in annotated)
            trial_drifts.append(
                TrialConfigDrift(
                    trial_name=trial_dir.name,
                    drifts=annotated,
                    fatal=fatal,
                )
            )
            if fatal:
                has_fatal = True

    # Top-level config drift.
    job_drifts_raw = _diff_dicts(prior_config, planned_top_level_config)
    job_drifts = [_annotate_drift(d) for d in job_drifts_raw]
    for d in job_drifts:
        if not _is_mutable_top_level(d):
            has_fatal = True

    # Already-complete detection: a job_result.json exists with finished_at
    # set and no trials are missing.
    state = _classify_state(
        prior_config=prior_config,
        job_dir=job_dir,
        has_fatal=has_fatal,
        job_drifts=job_drifts,
        trial_drifts=trial_drifts,
        coverage=coverage,
    )

    ambiguities = _detect_ambiguities(planned_top_level_config, trial_drifts)

    return ResumeReport(
        job_dir=job_dir,
        state=state,
        coverage=coverage,
        job_config_drift=job_drifts,
        trial_drifts=trial_drifts,
        ambiguities=ambiguities,
        corrupt_trial_dirs=corrupt,
        has_fatal_drift=has_fatal,
    )


def _first_agent_kwargs(top_level: Dict[str, Any]) -> Dict[str, Any]:
    agents = top_level.get("agents") or []
    if not agents:
        return {}
    first = agents[0]
    if not isinstance(first, dict):
        return {}
    return first.get("kwargs") or {}


def _first_agent_model_name(top_level: Dict[str, Any]) -> Optional[str]:
    agents = top_level.get("agents") or []
    if not agents:
        return None
    first = agents[0]
    if not isinstance(first, dict):
        return None
    return first.get("model_name")


def _diff_trial_against_plan(
    trial_config: Dict[str, Any],
    planned_top_level: Dict[str, Any],
    planned_agent_kwargs: Dict[str, Any],
) -> List[ConfigFieldDrift]:
    """Compare a single trial's config.json against the planned top-level config.

    Only fields covered by the allowlist (api_base / api_key / metrics_endpoint
    / base_url / timeouts / model_name via predicate) are surfaced — other
    structural drift is handled at the top-level config level.
    """
    drifts: List[ConfigFieldDrift] = []

    trial_agent = trial_config.get("agent") or {}
    trial_kwargs = trial_agent.get("kwargs") or {}

    # Allowlisted agent.kwargs scalars.
    for kw_key in ("api_base", "api_key", "metrics_endpoint", "base_url"):
        if kw_key in trial_kwargs or kw_key in planned_agent_kwargs:
            old = trial_kwargs.get(kw_key)
            new = planned_agent_kwargs.get(kw_key)
            if old != new:
                drifts.append(
                    ConfigFieldDrift(
                        path=("agent", "kwargs", kw_key),
                        old=old,
                        new=new,
                    )
                )

    # model_name (only mutable when the synthetic-vLLM-ID predicate fires).
    trial_model = trial_agent.get("model_name")
    planned_model = _first_agent_model_name(planned_top_level)
    if trial_model != planned_model:
        drifts.append(
            ConfigFieldDrift(
                path=("agent", "model_name"),
                old=trial_model,
                new=planned_model,
            )
        )

    # Timeout multipliers at the trial top level.
    for tk in (
        "timeout_multiplier",
        "agent_timeout_multiplier",
        "verifier_timeout_multiplier",
        "agent_setup_timeout_multiplier",
        "environment_build_timeout_multiplier",
    ):
        old = trial_config.get(tk)
        new = planned_top_level.get(tk)
        if old != new:
            drifts.append(ConfigFieldDrift(path=(tk,), old=old, new=new))

    return drifts


def _classify_state(
    *,
    prior_config: Dict[str, Any],
    job_dir: Path,
    has_fatal: bool,
    job_drifts: List[ConfigFieldDrift],
    trial_drifts: List[TrialConfigDrift],
    coverage: CoverageStats,
) -> ResumeState:
    if has_fatal:
        return ResumeState.FATAL_DRIFT
    has_mutable = bool(job_drifts) or any(td.drifts for td in trial_drifts)
    if has_mutable:
        return ResumeState.MUTABLE_DRIFT

    # No drift detected. Distinguish CLEAN_RESUME from ALREADY_COMPLETE.
    job_result = _safe_read_json(job_dir / "result.json")
    if job_result is not None:
        finished_at = job_result.get("finished_at")
        n_total = job_result.get("n_total_trials")
        if finished_at and n_total is not None:
            if coverage.n_with_result >= int(n_total):
                return ResumeState.ALREADY_COMPLETE
    return ResumeState.CLEAN_RESUME


def _detect_ambiguities(
    planned_top_level: Dict[str, Any],
    trial_drifts: List[TrialConfigDrift],
) -> List[str]:
    """Flag scenarios where resume's first-unmatched-wins matching is risky."""
    ambiguities: List[str] = []
    n_attempts = planned_top_level.get("n_attempts")
    if isinstance(n_attempts, int) and n_attempts > 1:
        ambiguities.append(
            f"n_attempts={n_attempts}: Harbor matches existing trials by Pydantic "
            "equality with first-unmatched-wins ordering. If only some attempts "
            "completed, the surviving attempt that gets re-attempted depends on "
            "iteration order."
        )
    if any(td.orphan for td in trial_drifts):
        n_orphans = sum(1 for td in trial_drifts if td.orphan)
        ambiguities.append(
            f"{n_orphans} existing trial dir(s) reference agent names not in the "
            "new config (orphans). Will be wiped on mutation."
        )
    return ambiguities


# -----------------------------------------------------------------------------
# Planning
# -----------------------------------------------------------------------------


def plan_mutation(
    report: ResumeReport,
    planned_top_level_config: Dict[str, Any],
    planned_agent_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[MutationPlan]:
    """Build a mutation plan, or return None if drift cannot be reconciled.

    Returns None when:
      - state == FATAL_DRIFT (some drift is outside the mutable allowlist)
      - state == NO_PRIOR (nothing to mutate)
      - state == CORRUPT (top-level config.json is unreadable)
    """
    if report.state in {ResumeState.NO_PRIOR, ResumeState.CORRUPT}:
        return None
    if report.has_fatal_drift:
        return None

    if planned_agent_kwargs is None:
        planned_agent_kwargs = _first_agent_kwargs(planned_top_level_config)

    # Top-level config is always rewritten to the planned shape on mutation.
    new_top_level_json = json.dumps(planned_top_level_config, indent=2, sort_keys=False)

    trial_rewrites: List[Tuple[Path, str]] = []
    trial_wipes: List[Path] = []

    for td in report.trial_drifts:
        trial_dir = report.job_dir / td.trial_name
        if td.orphan:
            trial_wipes.append(trial_dir)
            continue
        if not td.drifts:
            continue
        trial_config = _safe_read_json(trial_dir / "config.json")
        if trial_config is None:
            # Couldn't read this trial's config; safest is to wipe so Harbor
            # re-creates it.
            trial_wipes.append(trial_dir)
            continue
        patched = _apply_trial_mutations(trial_config, td.drifts, planned_top_level_config, planned_agent_kwargs)
        trial_rewrites.append(
            (trial_dir / "config.json", json.dumps(patched, indent=2, sort_keys=False))
        )

    return MutationPlan(
        new_top_level_config_json=new_top_level_json,
        trial_config_rewrites=trial_rewrites,
        trial_dirs_to_wipe=trial_wipes,
        trial_results_to_quarantine=list(report.corrupt_trial_dirs),
    )


def _apply_trial_mutations(
    trial_config: Dict[str, Any],
    drifts: List[ConfigFieldDrift],
    planned_top_level: Dict[str, Any],
    planned_agent_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of ``trial_config`` with allowlisted drifts patched to plan."""
    patched = copy.deepcopy(trial_config)
    planned_model = _first_agent_model_name(planned_top_level)
    agent_block = patched.setdefault("agent", {})
    kwargs_block = agent_block.setdefault("kwargs", {})

    for drift in drifts:
        path = drift.path
        if not path:
            continue
        head = path[0]
        if head == "agent" and len(path) >= 2 and path[1] == "kwargs" and len(path) == 3:
            field_name = path[2]
            new_val = planned_agent_kwargs.get(field_name)
            if new_val is None and field_name in kwargs_block:
                kwargs_block.pop(field_name, None)
            elif new_val is not None:
                kwargs_block[field_name] = new_val
        elif head == "agent" and len(path) >= 2 and path[1] == "model_name":
            agent_block["model_name"] = planned_model
        elif head in {
            "timeout_multiplier",
            "agent_timeout_multiplier",
            "verifier_timeout_multiplier",
            "agent_setup_timeout_multiplier",
            "environment_build_timeout_multiplier",
        }:
            new_val = planned_top_level.get(head)
            if new_val is None:
                patched.pop(head, None)
            else:
                patched[head] = new_val

    return patched


# -----------------------------------------------------------------------------
# Applying
# -----------------------------------------------------------------------------


def apply_mutation(plan: MutationPlan, *, job_dir: Path) -> None:
    """Apply a mutation plan to disk. Idempotent.

    Acquires an fcntl lock on ``<job_dir>/.resume_manager.lock`` for the
    duration of the apply to prevent concurrent resume operations from
    clobbering each other.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    lock_path = job_dir / ".resume_manager.lock"
    with _resume_lock(lock_path):
        # Quarantine corrupt result.json files first so subsequent reads see
        # the clean state.
        for corrupt_path in plan.trial_results_to_quarantine:
            if not corrupt_path.exists():
                continue
            quarantine = corrupt_path.with_suffix(corrupt_path.suffix + ".corrupt")
            corrupt_path.rename(quarantine)

        # Wipe orphan trial dirs.
        for trial_dir in plan.trial_dirs_to_wipe:
            if trial_dir.exists():
                shutil.rmtree(trial_dir)

        # Rewrite per-trial configs.
        for trial_config_path, new_json in plan.trial_config_rewrites:
            trial_config_path.parent.mkdir(parents=True, exist_ok=True)
            trial_config_path.write_text(new_json)

        # Rewrite top-level config.json.
        (job_dir / "config.json").write_text(plan.new_top_level_config_json)


def wipe_job_dir(job_dir: Path) -> None:
    """Remove all contents of ``job_dir`` (the dir itself stays so callers
    can land at the same path)."""
    job_dir = Path(job_dir)
    if not job_dir.exists():
        return
    for entry in job_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


class _resume_lock:
    """Context manager: acquire/release an fcntl lock on a file."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._fh = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"Another resume operation appears to be in flight at {self._path.parent}. "
                "Wait for it to finish or remove .resume_manager.lock if you're sure it's stale."
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


# -----------------------------------------------------------------------------
# Bail message rendering
# -----------------------------------------------------------------------------


def _format_path(path: Tuple[Any, ...]) -> str:
    if not path:
        return "<root>"
    parts: List[str] = []
    for p in path:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            parts.append(f".{p}" if parts else str(p))
    return "".join(parts)


def _format_drift_line(drift: ConfigFieldDrift) -> str:
    line = f"  {_format_path(drift.path)}: {drift.old!r} → {drift.new!r}"
    if drift.note:
        line += f"  ({drift.note})"
    return line


def render_bail_message(
    report: ResumeReport,
    *,
    force_mutate: bool,
    allow_overwrite: bool,
) -> str:
    """Pre-render an actionable bail message for the operator."""
    lines: List[str] = [
        f"[resume_manager] Prior job dir at {report.job_dir} cannot be auto-resumed:",
        "",
    ]

    if report.state == ResumeState.ALREADY_COMPLETE:
        lines.append("  Status: job is ALREADY COMPLETE on disk (finished_at set, all trials done).")
        if report.coverage:
            lines.append(
                f"           {report.coverage.n_with_result} trials with result.json, "
                f"{report.coverage.n_without_result} without."
            )
        lines.append("")
        lines.append("  Choose one:")
        lines.append("    --allow_overwrite                     wipe + re-run from scratch")
        lines.append(
            "    --experiments_dir <fresh_path>        land at a fresh path; leave existing dir alone"
        )
        return "\n".join(lines)

    if report.state == ResumeState.CORRUPT:
        lines.append("  Status: prior config.json is CORRUPT (unparseable).")
        lines.append("")
        lines.append("  Choose one:")
        lines.append("    --allow_overwrite                     wipe existing dir, start fresh")
        lines.append("    --experiments_dir <fresh_path>        land at a fresh path")
        return "\n".join(lines)

    # Coverage block
    if report.coverage:
        cov = report.coverage
        n_total_seen = cov.n_existing
        lines.append("  Coverage:")
        lines.append(f"    on disk: {n_total_seen} trial dirs")
        lines.append(
            f"             {cov.n_with_result} with result.json, "
            f"{cov.n_without_result} without, {cov.n_orphans} orphans"
        )
        if cov.by_exception_type:
            lines.append("    by exception type:")
            for exc, n in sorted(cov.by_exception_type.items(), key=lambda kv: -kv[1]):
                lines.append(f"      {exc}: {n}")
        if cov.reward_distribution:
            buckets = ", ".join(
                f"{k}={v}" for k, v in sorted(cov.reward_distribution.items())
            )
            lines.append(f"    rewards: {buckets}")
        lines.append("")

    # Drift block
    fatal_drifts = [d for d in report.job_config_drift if not _is_mutable_top_level(d)]
    mutable_drifts = [d for d in report.job_config_drift if _is_mutable_top_level(d)]
    fatal_trial_drifts = [
        td for td in report.trial_drifts if td.fatal and not td.orphan
    ]
    orphans = [td for td in report.trial_drifts if td.orphan]
    mutable_trial_count = sum(
        1 for td in report.trial_drifts if td.drifts and not td.fatal
    )

    if fatal_drifts:
        lines.append("  Job config drift (fatal):")
        for d in fatal_drifts:
            lines.append(_format_drift_line(d))
        lines.append("")
    if mutable_drifts:
        lines.append("  Job config drift (mutable):")
        for d in mutable_drifts:
            lines.append(_format_drift_line(d))
        lines.append("")
    if fatal_trial_drifts:
        lines.append(f"  Per-trial fatal drift: {len(fatal_trial_drifts)} trial(s)")
        sample = fatal_trial_drifts[0]
        lines.append(f"    e.g. {sample.trial_name}:")
        for d in sample.drifts[:3]:
            lines.append(f"    {_format_drift_line(d)}")
        if len(sample.drifts) > 3:
            lines.append(f"      ... and {len(sample.drifts) - 3} more")
        lines.append("")
    if mutable_trial_count and not fatal_trial_drifts:
        lines.append(
            f"  Per-trial mutable drift: {mutable_trial_count} trial(s) — would be patched."
        )
        lines.append("")
    if orphans:
        lines.append(
            f"  Orphan trials (agent name no longer in config): {len(orphans)} — would be wiped."
        )
        lines.append("")
    if report.ambiguities:
        lines.append("  Notes:")
        for note in report.ambiguities:
            lines.append(f"    - {note}")
        lines.append("")

    # Action menu
    lines.append("  Choose one:")
    is_fully_mutable = not report.has_fatal_drift and (
        mutable_drifts or mutable_trial_count or orphans
    )
    mutate_note = (
        ""
        if is_fully_mutable
        else "  (NOT applicable: drift includes fatal fields)"
    )
    lines.append(f"    --force_mutate                        patch existing dir + resume{mutate_note}")
    lines.append("    --allow_overwrite                     wipe existing dir, start fresh")
    lines.append(
        "    --experiments_dir <path>              land at a fresh path; leave existing dir alone"
    )
    lines.append("")
    lines.append("  Combine --force_mutate --allow_overwrite to mutate where possible, wipe on fatal.")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Policy resolution
# -----------------------------------------------------------------------------


def resolve_resume_policy(
    job_dir: Path,
    planned_top_level_config: Dict[str, Any],
    *,
    force_mutate: bool,
    allow_overwrite: bool,
) -> ResumePolicyResult:
    """Inspect + plan + decide per the matrix. Raises ``ResumeBail`` on bail.

    Side effects: applies mutation or wipes the dir as decided. The caller
    only needs to suppress dedup (so the launcher lands at the same path).
    """
    report = inspect_resume(job_dir, planned_top_level_config)

    if report.state == ResumeState.NO_PRIOR:
        return ResumePolicyResult(action=ResumeAction.NO_ENGAGEMENT, report=report)

    if report.state == ResumeState.CLEAN_RESUME:
        return ResumePolicyResult(action=ResumeAction.CLEAN_RESUME, report=report)

    if report.state == ResumeState.ALREADY_COMPLETE:
        if allow_overwrite:
            wipe_job_dir(job_dir)
            return ResumePolicyResult(action=ResumeAction.WIPE_AND_FRESH, report=report)
        raise ResumeBail(
            report,
            render_bail_message(report, force_mutate=force_mutate, allow_overwrite=allow_overwrite),
        )

    if report.state == ResumeState.CORRUPT:
        if allow_overwrite:
            wipe_job_dir(job_dir)
            return ResumePolicyResult(action=ResumeAction.WIPE_AND_FRESH, report=report)
        raise ResumeBail(
            report,
            render_bail_message(report, force_mutate=force_mutate, allow_overwrite=allow_overwrite),
        )

    # Drift cases: MUTABLE_DRIFT or FATAL_DRIFT.
    plan: Optional[MutationPlan] = None
    if force_mutate and not report.has_fatal_drift:
        plan = plan_mutation(report, planned_top_level_config)

    if plan is not None:
        apply_mutation(plan, job_dir=job_dir)
        return ResumePolicyResult(
            action=ResumeAction.MUTATE_AND_RESUME, report=report, plan=plan
        )

    # Mutation not attempted or not possible.
    if allow_overwrite:
        wipe_job_dir(job_dir)
        return ResumePolicyResult(action=ResumeAction.WIPE_AND_FRESH, report=report)

    raise ResumeBail(
        report,
        render_bail_message(report, force_mutate=force_mutate, allow_overwrite=allow_overwrite),
    )


# -----------------------------------------------------------------------------
# Top-level seam used by hpc/launch_utils.py:resolve_job_and_paths
# -----------------------------------------------------------------------------


def _is_harbor_backed_job_type(job_type: Optional[str]) -> bool:
    return str(job_type or "").lower() in {"datagen", "eval"}


def _resolve_prior_job_dir(exp_args: Dict[str, Any], job_name: str) -> Optional[Path]:
    """Compute the path where a prior job's Harbor artifacts would live.

    On disk Harbor writes ``config.json`` + trial dirs under
    ``<experiments_dir>/trace_jobs/<job_name>_traces/``. The ``<job_name>_traces``
    suffix is added by ``hpc/datagen_launch_utils.py`` when it constructs the
    chunk job name passed down to Harbor as the trace-job name.

    ``experiments_dir`` mirrors ``hpc/launch_utils.py:setup_experiments_dir``:
    explicit ``--experiments_dir`` is used verbatim; otherwise it defaults to
    ``experiments/<job_name>`` (the un-deduped path the launcher lands at
    before its ``_2`` suffix kicks in).
    """
    if not job_name:
        return None
    from hpc.launch_utils import resolve_workspace_path
    experiments_dir = exp_args.get("experiments_dir")
    if experiments_dir:
        experiments_root = resolve_workspace_path(str(experiments_dir))
    else:
        experiments_root = resolve_workspace_path(f"experiments/{job_name}")
    return experiments_root / "trace_jobs" / f"{job_name}_traces"


def _materialize_planned_config(exp_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort materialization of the merged Harbor config dict.

    Returns None when we can't build a meaningful planned config from the
    available CLI args (e.g. no harbor config path). In that case the
    resume manager declines to engage and the caller falls through to
    normal dedup.
    """
    from hpc.harbor_utils import load_harbor_config, merge_harbor_config

    harbor_config_path = (
        exp_args.get("trace_harbor_config")
        or exp_args.get("harbor_config")
    )
    if not harbor_config_path:
        return None
    try:
        harbor_config_data = load_harbor_config(str(harbor_config_path))
    except (FileNotFoundError, OSError):
        return None
    except Exception:
        return None

    model_name = (
        exp_args.get("_harbor_model_name")
        or exp_args.get("model_name_or_path")
        or exp_args.get("trace_model")
        or exp_args.get("datagen_model")
        or ""
    )
    # Mirror the launcher's later `hosted_vllm_alias(generate_served_model_id(
    # job_name=...))` substitution so the planned model_name we diff against
    # the on-disk config.json matches what the launcher will write to
    # merged_harbor_config.yaml. Without this overlay, planned has the raw HF
    # repo name (e.g. ``cyankiwi/...``) and the on-disk JSON — rewritten by a
    # prior mutation — has either the same HF name or a stale synthetic; in
    # both cases the runtime YAML's deterministic ``hosted_vllm/<id>``
    # diverges from JSON at Harbor's equality check.
    job_name_for_alias = exp_args.get("job_name")
    if job_name_for_alias and model_name and not str(model_name).startswith(_HOSTED_VLLM_PREFIX):
        # Only rewrite when the launcher itself would (datagen/eval paths
        # that need a vLLM server). Detect by checking whether the resolved
        # exp_args carries a vllm-server config or the launcher has stashed
        # a `_harbor_model_name` override (which would have already produced
        # the alias and been caught by the fallback chain above).
        from hpc.launch_utils import hosted_vllm_alias, generate_served_model_id
        # Seed with the BARE job_name (not the ``_traces``-suffixed chunk
        # name) to match the launcher's own call at
        # ``hpc/datagen_launch_utils.py:_prepare_datagen_configuration``.
        # The launcher generates a
        # single synthetic per launch and reuses it across chunks; we
        # mirror that here so resume_manager's planned config matches
        # the YAML at the model_name field.
        synthetic_id = generate_served_model_id(job_name=job_name_for_alias)
        model_name = hosted_vllm_alias(synthetic_id)
    n_concurrent = exp_args.get("trace_n_concurrent") or exp_args.get("n_concurrent") or 4
    try:
        n_concurrent = int(n_concurrent)
    except (TypeError, ValueError):
        n_concurrent = 4

    planned = merge_harbor_config(
        harbor_config_data,
        agent_name=exp_args.get("trace_agent_name") or exp_args.get("harbor_agent_name"),
        model_name=str(model_name),
        n_concurrent=n_concurrent,
        endpoint_meta=exp_args.get("_harbor_endpoint_meta"),
        agent_kwarg_overrides=list(exp_args.get("agent_kwarg") or exp_args.get("agent_kwargs") or []),
        extra_agent_kwargs=exp_args.get("_harbor_extra_agent_kwargs"),
    )

    # Overlay the CLI-resolved runtime values that merge_harbor_config doesn't
    # know about. Without this, the planned config keeps the harbor-yaml
    # placeholder strings (`/replace/with/tasks/path`, `default-trace-job`,
    # `trace_jobs`) and every comparison against an on-disk config bails with
    # spurious "fatal drifts" at these paths.
    if not isinstance(planned, dict):
        return planned
    tasks_path = exp_args.get("tasks_input_path") or exp_args.get("trace_input_path")
    if tasks_path:
        datasets = planned.get("datasets")
        if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
            if datasets[0].get("path") in (None, "", "/replace/with/tasks/path"):
                datasets[0]["path"] = str(tasks_path)
    # Resolve job_name to the trace-job-name shape the launcher actually
    # passes to Harbor (chunk_job_name = "<base>_traces"; see
    # hpc/datagen_launch_utils.py:476).
    base_job_name = exp_args.get("job_name")
    if base_job_name and planned.get("job_name") in (None, "", "default-trace-job"):
        planned["job_name"] = f"{base_job_name}_traces"
    # jobs_dir is computed by the launcher as <experiments_dir>/trace_jobs.
    # Resolve to an absolute path so the comparison against the on-disk
    # absolute path doesn't trip on absolute-vs-relative shape alone.
    from hpc.launch_utils import resolve_workspace_path
    experiments_dir = exp_args.get("experiments_dir")
    if planned.get("jobs_dir") in (None, "", "trace_jobs"):
        if experiments_dir:
            planned["jobs_dir"] = str(resolve_workspace_path(str(experiments_dir)) / "trace_jobs")
        elif base_job_name:
            planned["jobs_dir"] = str(
                resolve_workspace_path(f"experiments/{base_job_name}") / "trace_jobs"
            )

    return planned


def resolve_resume_policy_for_launch(
    exp_args: Dict[str, Any],
    *,
    job_name: str,
) -> Optional[ResumePolicyResult]:
    """Top-level seam called from ``hpc/launch_utils.py:resolve_job_and_paths``.

    Returns ``None`` when the manager declines to engage:
      - non-Harbor job type (SFT, RL, consolidate, pretokenize)
      - prior job dir doesn't exist
      - planned config can't be materialized

    Returns a :class:`ResumePolicyResult` describing the action taken,
    or raises :class:`ResumeBail` if the matrix decided to bail.
    """
    job_type = exp_args.get("job_type")
    if not _is_harbor_backed_job_type(job_type):
        return None

    prior_dir = _resolve_prior_job_dir(exp_args, job_name)
    if prior_dir is None or not prior_dir.exists():
        return None

    planned = _materialize_planned_config(exp_args)
    if planned is None:
        return None

    force_mutate = bool(exp_args.get("force_mutate"))
    allow_overwrite = bool(exp_args.get("allow_overwrite"))

    result = resolve_resume_policy(
        prior_dir,
        planned,
        force_mutate=force_mutate,
        allow_overwrite=allow_overwrite,
    )

    if result.action == ResumeAction.NO_ENGAGEMENT:
        return None

    _emit_action_summary(result)
    return result


def _emit_action_summary(result: ResumePolicyResult) -> None:
    """Print a one-line operator summary of the resume action taken."""
    rpt = result.report
    cov = rpt.coverage or CoverageStats()
    if result.action == ResumeAction.CLEAN_RESUME:
        msg = (
            f"[resume_manager] Resuming at {rpt.job_dir} — {cov.n_with_result} trials done, "
            f"{cov.n_without_result} pending."
        )
    elif result.action == ResumeAction.MUTATE_AND_RESUME:
        plan = result.plan
        msg = (
            f"[resume_manager] Mutating + resuming at {rpt.job_dir} — "
            f"{cov.n_with_result} trials kept, "
            f"{len(plan.trial_config_rewrites) if plan else 0} trial configs patched, "
            f"{len(plan.trial_dirs_to_wipe) if plan else 0} orphans wiped."
        )
    elif result.action == ResumeAction.WIPE_AND_FRESH:
        msg = (
            f"[resume_manager] Wiped {rpt.job_dir} (--allow_overwrite). "
            f"Starting fresh."
        )
    else:
        return
    print(msg, file=sys.stderr)


# -----------------------------------------------------------------------------
# Standalone CLI: python -m hpc.resume_manager <subcommand>
# -----------------------------------------------------------------------------


def _cli_load_planned_from_config_arg(config_path: Path) -> Dict[str, Any]:
    """Load a planned config dict from a YAML or JSON file."""
    import yaml
    text = config_path.read_text()
    if config_path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m hpc.resume_manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    insp = sub.add_parser("inspect", help="Print ResumeReport for an existing job dir")
    insp.add_argument("job_dir", type=Path)
    insp.add_argument(
        "--config",
        type=Path,
        help="(optional) planned config YAML/JSON to diff against; omit for coverage-only.",
    )

    diff = sub.add_parser("diff", help="Print drift only")
    diff.add_argument("job_dir", type=Path)
    diff.add_argument("--config", type=Path, required=True)

    pln = sub.add_parser("plan", help="Print a mutation plan (no writes)")
    pln.add_argument("job_dir", type=Path)
    pln.add_argument("--config", type=Path, required=True)

    apl = sub.add_parser("apply", help="Apply a mutation plan")
    apl.add_argument("job_dir", type=Path)
    apl.add_argument("--config", type=Path, required=True)
    apl.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    wip = sub.add_parser("wipe", help="rmtree the contents of <job_dir>")
    wip.add_argument("job_dir", type=Path)
    wip.add_argument("--yes", action="store_true")

    args = p.parse_args()
    job_dir = Path(args.job_dir)

    if args.cmd == "inspect":
        planned = (
            _cli_load_planned_from_config_arg(args.config) if args.config else {}
        )
        report = inspect_resume(job_dir, planned)
        print(json.dumps(_report_to_dict(report), default=str, indent=2))
        return 0

    if args.cmd == "diff":
        planned = _cli_load_planned_from_config_arg(args.config)
        report = inspect_resume(job_dir, planned)
        if not (report.job_config_drift or any(td.drifts for td in report.trial_drifts)):
            print("No drift detected.")
            return 0
        for d in report.job_config_drift:
            print(_format_drift_line(d))
        for td in report.trial_drifts:
            if not td.drifts:
                continue
            print(f"  [{td.trial_name}]")
            for d in td.drifts:
                print(f"  {_format_drift_line(d)}")
        return 0

    if args.cmd == "plan":
        planned = _cli_load_planned_from_config_arg(args.config)
        report = inspect_resume(job_dir, planned)
        plan = plan_mutation(report, planned)
        if plan is None:
            print(f"Cannot plan mutation (state={report.state.value}, has_fatal={report.has_fatal_drift}).")
            return 1
        print(
            json.dumps(
                {
                    "estimated_changes": plan.estimated_changes,
                    "trial_config_rewrites": [str(p) for p, _ in plan.trial_config_rewrites],
                    "trial_dirs_to_wipe": [str(p) for p in plan.trial_dirs_to_wipe],
                    "trial_results_to_quarantine": [str(p) for p in plan.trial_results_to_quarantine],
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "apply":
        planned = _cli_load_planned_from_config_arg(args.config)
        report = inspect_resume(job_dir, planned)
        plan = plan_mutation(report, planned)
        if plan is None:
            print(f"Cannot plan mutation (state={report.state.value}, has_fatal={report.has_fatal_drift}).")
            return 1
        if not args.yes:
            print(
                f"About to apply mutation: {plan.estimated_changes} changes "
                f"({len(plan.trial_config_rewrites)} rewrites, "
                f"{len(plan.trial_dirs_to_wipe)} wipes, "
                f"{len(plan.trial_results_to_quarantine)} quarantines)."
            )
            resp = input("Proceed? [y/N] ").strip().lower()
            if resp not in {"y", "yes"}:
                print("Aborted.")
                return 1
        apply_mutation(plan, job_dir=job_dir)
        print("Applied.")
        return 0

    if args.cmd == "wipe":
        if not args.yes:
            resp = input(f"About to wipe contents of {job_dir}. Proceed? [y/N] ").strip().lower()
            if resp not in {"y", "yes"}:
                print("Aborted.")
                return 1
        wipe_job_dir(job_dir)
        print(f"Wiped {job_dir}.")
        return 0

    return 1


def _report_to_dict(report: ResumeReport) -> Dict[str, Any]:
    return {
        "job_dir": str(report.job_dir),
        "state": report.state.value,
        "has_fatal_drift": report.has_fatal_drift,
        "coverage": asdict(report.coverage) if report.coverage else None,
        "job_config_drift": [
            {"path": list(d.path), "old": d.old, "new": d.new, "note": d.note}
            for d in report.job_config_drift
        ],
        "trial_drifts": [
            {
                "trial_name": td.trial_name,
                "fatal": td.fatal,
                "orphan": td.orphan,
                "drifts": [
                    {"path": list(d.path), "old": d.old, "new": d.new, "note": d.note}
                    for d in td.drifts
                ],
            }
            for td in report.trial_drifts
        ],
        "ambiguities": report.ambiguities,
        "corrupt_trial_dirs": [str(p) for p in report.corrupt_trial_dirs],
    }


__all__ = [
    "ResumeState",
    "ResumeAction",
    "ConfigFieldDrift",
    "TrialConfigDrift",
    "CoverageStats",
    "ResumeReport",
    "MutationPlan",
    "ResumePolicyResult",
    "ResumeBail",
    "inspect_resume",
    "plan_mutation",
    "apply_mutation",
    "wipe_job_dir",
    "render_bail_message",
    "resolve_resume_policy",
    "resolve_resume_policy_for_launch",
]


if __name__ == "__main__":
    sys.exit(_cli())
