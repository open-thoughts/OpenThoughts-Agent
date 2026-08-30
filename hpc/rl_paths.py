"""Resolve durable RL state paths and validate checkpoint resume contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from hpc.artifact_store import ArtifactStorePaths, paths_for_run
from hpc.experiment_path_names import numbered_experiment_fork_pattern


CHECKPOINTS_SUBDIR = "checkpoints"
EXPORTS_SUBDIR = "exports"
TRACE_JOBS_SUBDIR = "trace_jobs"
LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
GLOBAL_STEP_PREFIX = "global_step_"
GLOBAL_STEP_PATTERN = re.compile(rf"{GLOBAL_STEP_PREFIX}(\d+)$")
SHARD_WORLD_SIZE_PATTERN = re.compile(
    r"(?:^|_)(?:model|optim|extra_state)_world_size_(\d+)_rank_\d+\.pt$"
)
HYDRA_NULL_VALUES = frozenset({"", "null", "None", "~"})
CKPT_PATH_KEY = "trainer.ckpt_path"
EXPORT_PATH_KEY = "trainer.export_path"
RESUME_MODE_KEY = "trainer.resume_mode"
RESUME_PATH_KEY = "trainer.resume_path"
TRIALS_DIR_KEY = "terminal_bench_config.trials_dir"


class CheckpointLayoutError(ValueError):
    """A checkpoint path cannot satisfy the requested resume contract."""


class AmbiguousCheckpointError(CheckpointLayoutError):
    """Multiple run roots contain the same highest checkpoint step."""


class CheckpointWorldSizeError(CheckpointLayoutError):
    """A selected checkpoint's sharded world size cannot satisfy the resolved placement."""


def checkpoint_component_world_sizes(checkpoint_path: Path) -> dict[str, int]:
    """Read the per-component world sizes recorded in FSDP2 shard file names.

    FSDP2 writes one shard per rank (``model_world_size_<W>_rank_<R>.pt``) under
    each component directory (``policy/``, ``ref/``, ``critic/``). The loader
    reads the shard for its own rank at the CURRENT world size and does not
    reshard, so the recorded world size is a hard resume constraint.
    """

    sizes: dict[str, int] = {}
    if not checkpoint_path.is_dir():
        return sizes
    for component_dir in checkpoint_path.iterdir():
        if not component_dir.is_dir():
            continue
        for shard in component_dir.iterdir():
            match = SHARD_WORLD_SIZE_PATTERN.search(shard.name)
            if match is not None:
                sizes[component_dir.name] = int(match.group(1))
                break
    return sizes


class RLResumeMode(StrEnum):
    NONE = "none"
    LATEST = "latest"
    FROM_PATH = "from_path"


class RLResumePolicy(StrEnum):
    FIXED = "fixed"
    AT_LINK_START = "at_link_start"


class RLLaunchIntent(StrEnum):
    AUTO = "auto"
    FRESH = "fresh"


@dataclass(frozen=True)
class CheckpointCandidate:
    state_root: Path
    checkpoint_path: Path
    step: int


@dataclass(frozen=True)
class RLRunPaths:
    """Resolved paths consumed by the launcher and SkyRL configuration."""

    job_name: str
    checkpoint_dir: Path
    export_dir: Path
    trials_dir: Path
    resume_mode: RLResumeMode
    resume_path: Path | None
    resume_policy: RLResumePolicy
    artifact_store: ArtifactStorePaths | None = None

    def describe(self) -> str:
        if self.resume_path is not None:
            action = f"RESUME from {self.resume_path}"
        else:
            action = "FRESH START (no checkpoint selected)"
        return (
            f"[rl_paths] {action}; resume_mode={self.resume_mode.value}; "
            f"checkpoints={self.checkpoint_dir}; exports={self.export_dir}; trials={self.trials_dir}"
        )


def hydra_override_values(overrides: Sequence[str]) -> dict[str, str]:
    """Return Hydra override values using its last-value-wins convention."""

    values: dict[str, str] = {}
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            continue
        key = key.lstrip("+").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _configured_path_values(
    trainer_config: Mapping[str, object],
    terminal_bench_config: Mapping[str, object],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for config_key, hydra_key in (
        ("ckpt_path", CKPT_PATH_KEY),
        ("export_path", EXPORT_PATH_KEY),
        ("resume_mode", RESUME_MODE_KEY),
        ("resume_path", RESUME_PATH_KEY),
    ):
        value = trainer_config.get(config_key)
        if value is not None:
            values[hydra_key] = str(value)

    trials_dir = terminal_bench_config.get("trials_dir")
    if trials_dir is not None:
        values[TRIALS_DIR_KEY] = str(trials_dir)
    return values


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _validate_checkpoint_world_size(
    checkpoint_path: Path,
    expected_world_sizes: Mapping[str, int] | None,
) -> None:
    """Fail at submit when a selected checkpoint cannot load at the resolved world size.

    FSDP2 writes one shard file per rank and the loader does not reshard, so a
    bank written at 16 ranks can never load into a 32-rank placement. Detecting
    the mismatch here turns a ~15-minute multi-node failure into a submit-time
    error that costs seconds.
    """

    if not expected_world_sizes:
        return
    recorded = checkpoint_component_world_sizes(checkpoint_path)
    for component, expected in expected_world_sizes.items():
        found = recorded.get(component)
        if found is None or found == expected:
            continue
        raise CheckpointWorldSizeError(
            f"Checkpoint {checkpoint_path} was written with {component} world size "
            f"{found}, but the resolved placement implies world size {expected}. "
            "FSDP2 loads one shard per rank and does not reshard. Run at the "
            f"original world size ({found}) or start fresh (set trainer.resume_mode=none "
            "or move the checkpoint bank aside)."
        )


class RLPathManager:
    """Own checkpoint discovery and all durable RL path resolution."""

    def __init__(
        self,
        job_name: str,
        canonical_root: Path,
        launch_root: Path,
        *,
        artifact_store_enabled: bool = False,
    ):
        self.job_name = job_name
        self.canonical_root = canonical_root.expanduser().resolve()
        self.launch_root = launch_root.expanduser().resolve()
        self.artifact_store_enabled = artifact_store_enabled

    def resolve(
        self,
        *,
        trainer_config: Mapping[str, object] | None = None,
        terminal_bench_config: Mapping[str, object] | None = None,
        skyrl_overrides: Sequence[str] = (),
        launch_intent: RLLaunchIntent = RLLaunchIntent.AUTO,
        expected_world_sizes: Mapping[str, int] | None = None,
    ) -> RLRunPaths:
        cli_values = hydra_override_values(skyrl_overrides)
        overrides = _configured_path_values(
            trainer_config or {}, terminal_bench_config or {}
        )
        overrides.update(cli_values)
        requested_mode, requested_resume_path = self._requested_resume(
            overrides, launch_intent
        )
        resume_policy = (
            RLResumePolicy.AT_LINK_START
            if launch_intent is RLLaunchIntent.AUTO and requested_mode is None
            else RLResumePolicy.FIXED
        )

        state_root = (
            self.launch_root
            if launch_intent is RLLaunchIntent.FRESH
            else self.canonical_root
        )
        checkpoint_dir = self._configured_checkpoint_dir(overrides, state_root)
        state_root = self._state_root_for_checkpoint_dir(
            checkpoint_dir, state_root, overrides
        )

        explicit = self._resolve_explicit_request(
            requested_mode,
            requested_resume_path,
            launch_intent,
            state_root,
            checkpoint_dir,
            overrides,
            resume_policy,
            expected_world_sizes,
        )
        if explicit is not None:
            return explicit

        if CKPT_PATH_KEY in overrides:
            candidate = self._checkpoint_candidate(state_root, checkpoint_dir)
            candidates = [candidate] if candidate is not None else []
        else:
            candidates = self._checkpoint_candidates()

        if not candidates:
            return self._resolved_paths(
                state_root,
                checkpoint_dir,
                overrides,
                resume_mode=RLResumeMode.NONE,
                resume_path=None,
                resume_policy=resume_policy,
            )

        highest_step = max(candidate.step for candidate in candidates)
        highest = [
            candidate for candidate in candidates if candidate.step == highest_step
        ]
        if len(highest) != 1:
            paths = ", ".join(str(candidate.checkpoint_path) for candidate in highest)
            raise AmbiguousCheckpointError(
                f"Multiple run roots contain global_step_{highest_step}: {paths}. "
                "Set trainer.resume_mode=from_path and trainer.resume_path explicitly."
            )

        selected = highest[0]
        _validate_checkpoint_world_size(selected.checkpoint_path, expected_world_sizes)
        return self._resolved_paths(
            selected.state_root,
            selected.checkpoint_path.parent,
            overrides,
            resume_mode=RLResumeMode.LATEST,
            resume_path=selected.checkpoint_path,
            resume_policy=resume_policy,
        )

    @staticmethod
    def _requested_resume(
        overrides: dict[str, str], launch_intent: RLLaunchIntent
    ) -> tuple[str | None, str | None]:
        requested_mode = overrides.get(RESUME_MODE_KEY)
        requested_resume_path = overrides.get(RESUME_PATH_KEY)
        if requested_mode in HYDRA_NULL_VALUES:
            requested_mode = None
        if requested_resume_path in HYDRA_NULL_VALUES:
            requested_resume_path = None

        if launch_intent is RLLaunchIntent.FRESH and requested_mode not in (
            None,
            RLResumeMode.NONE.value,
        ):
            raise CheckpointLayoutError(
                "A fresh start cannot also request checkpoint resume"
            )
        if requested_resume_path and requested_mode != RLResumeMode.FROM_PATH.value:
            raise CheckpointLayoutError(
                "trainer.resume_path requires trainer.resume_mode=from_path"
            )
        if requested_mode not in {None, *(mode.value for mode in RLResumeMode)}:
            raise CheckpointLayoutError(
                f"Unknown trainer.resume_mode: {requested_mode}"
            )
        return requested_mode, requested_resume_path

    def _resolve_explicit_request(
        self,
        requested_mode: str | None,
        requested_resume_path: str | None,
        launch_intent: RLLaunchIntent,
        state_root: Path,
        checkpoint_dir: Path,
        overrides: dict[str, str],
        resume_policy: RLResumePolicy,
        expected_world_sizes: Mapping[str, int] | None = None,
    ) -> RLRunPaths | None:
        """Resolve a user-selected mode, or return None for automatic discovery."""

        if (
            launch_intent is RLLaunchIntent.FRESH
            or requested_mode == RLResumeMode.NONE.value
        ):
            return self._resolved_paths(
                state_root,
                checkpoint_dir,
                overrides,
                resume_mode=RLResumeMode.NONE,
                resume_path=None,
                resume_policy=resume_policy,
            )
        if requested_mode == RLResumeMode.LATEST.value:
            candidate = self._required_checkpoint_candidate(state_root, checkpoint_dir)
            _validate_checkpoint_world_size(
                candidate.checkpoint_path, expected_world_sizes
            )
            return self._resolved_paths(
                state_root,
                checkpoint_dir,
                overrides,
                resume_mode=RLResumeMode.LATEST,
                resume_path=candidate.checkpoint_path,
                resume_policy=resume_policy,
            )
        if requested_mode != RLResumeMode.FROM_PATH.value:
            return None

        if not requested_resume_path:
            raise CheckpointLayoutError(
                "trainer.resume_mode=from_path requires trainer.resume_path"
            )
        resume_path = self._validate_explicit_resume_path(
            _absolute_path(requested_resume_path)
        )
        _validate_checkpoint_world_size(resume_path, expected_world_sizes)
        if CKPT_PATH_KEY in overrides and checkpoint_dir != resume_path.parent:
            raise CheckpointLayoutError(
                f"trainer.resume_path {resume_path} is not under trainer.ckpt_path {checkpoint_dir}"
            )
        checkpoint_dir = resume_path.parent
        state_root = self._state_root_for_checkpoint_dir(
            checkpoint_dir, state_root, overrides
        )
        return self._resolved_paths(
            state_root,
            checkpoint_dir,
            overrides,
            resume_mode=RLResumeMode.FROM_PATH,
            resume_path=resume_path,
            resume_policy=resume_policy,
        )

    def _configured_checkpoint_dir(
        self, overrides: dict[str, str], state_root: Path
    ) -> Path:
        configured = overrides.get(CKPT_PATH_KEY)
        if configured:
            return _absolute_path(configured)
        return state_root / self.job_name / CHECKPOINTS_SUBDIR

    def _state_root_for_checkpoint_dir(
        self, checkpoint_dir: Path, fallback: Path, overrides: dict[str, str]
    ) -> Path:
        if (
            checkpoint_dir.name == CHECKPOINTS_SUBDIR
            and checkpoint_dir.parent.name == self.job_name
        ):
            return checkpoint_dir.parent.parent
        missing = [
            key for key in (EXPORT_PATH_KEY, TRIALS_DIR_KEY) if key not in overrides
        ]
        if missing:
            raise CheckpointLayoutError(
                f"Nonstandard checkpoint directory {checkpoint_dir} requires explicit values for "
                f"{', '.join(missing)}"
            )
        return fallback

    def _checkpoint_candidates(self) -> list[CheckpointCandidate]:
        candidates: list[CheckpointCandidate] = []
        for state_root in self._candidate_state_roots():
            checkpoint_dir = state_root / self.job_name / CHECKPOINTS_SUBDIR
            candidate = self._checkpoint_candidate(state_root, checkpoint_dir)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _candidate_state_roots(self) -> list[Path]:
        roots = [self.canonical_root]
        parent = self.canonical_root.parent
        if not parent.is_dir():
            return roots
        fork_pattern = numbered_experiment_fork_pattern(self.canonical_root.name)
        roots.extend(
            path.resolve()
            for path in parent.iterdir()
            if path.is_dir() and fork_pattern.fullmatch(path.name)
        )
        return roots

    def _checkpoint_candidate(
        self,
        state_root: Path,
        checkpoint_dir: Path,
    ) -> CheckpointCandidate | None:
        marker_path = checkpoint_dir / LATEST_CHECKPOINT_FILE
        checkpoint_steps = self._checkpoint_steps(checkpoint_dir)
        if not marker_path.is_file():
            if checkpoint_steps:
                raise CheckpointLayoutError(
                    f"Checkpoint directories exist under {checkpoint_dir}, but {LATEST_CHECKPOINT_FILE} is missing"
                )
            return None

        try:
            step = int(marker_path.read_text().strip())
        except (OSError, ValueError) as error:
            raise CheckpointLayoutError(
                f"Invalid checkpoint marker: {marker_path}"
            ) from error

        checkpoint_path = checkpoint_dir / f"{GLOBAL_STEP_PREFIX}{step}"
        if not checkpoint_path.is_dir():
            raise CheckpointLayoutError(
                f"Checkpoint marker {marker_path} names missing {checkpoint_path.name}"
            )
        if checkpoint_steps and max(checkpoint_steps) != step:
            raise CheckpointLayoutError(
                f"Checkpoint marker {marker_path} names {GLOBAL_STEP_PREFIX}{step}, "
                f"but {GLOBAL_STEP_PREFIX}{max(checkpoint_steps)} also exists"
            )
        return CheckpointCandidate(state_root, checkpoint_path, step)

    def _required_checkpoint_candidate(
        self, state_root: Path, checkpoint_dir: Path
    ) -> CheckpointCandidate:
        candidate = self._checkpoint_candidate(state_root, checkpoint_dir)
        if candidate is None:
            raise CheckpointLayoutError(
                f"trainer.resume_mode=latest requires a usable checkpoint under {checkpoint_dir}"
            )
        return candidate

    @staticmethod
    def _checkpoint_steps(checkpoint_dir: Path) -> list[int]:
        if not checkpoint_dir.is_dir():
            return []
        steps: list[int] = []
        for path in checkpoint_dir.iterdir():
            match = GLOBAL_STEP_PATTERN.fullmatch(path.name)
            if path.is_dir() and match:
                steps.append(int(match.group(1)))
        return steps

    @staticmethod
    def _validate_explicit_resume_path(resume_path: Path) -> Path:
        if (
            not resume_path.is_dir()
            or GLOBAL_STEP_PATTERN.fullmatch(resume_path.name) is None
        ):
            raise CheckpointLayoutError(
                f"trainer.resume_path must be an existing global_step_<N> directory: {resume_path}"
            )
        return resume_path

    def _resolved_paths(
        self,
        state_root: Path,
        checkpoint_dir: Path,
        overrides: dict[str, str],
        *,
        resume_mode: RLResumeMode,
        resume_path: Path | None,
        resume_policy: RLResumePolicy,
    ) -> RLRunPaths:
        trainer_root = state_root / self.job_name
        export_dir = (
            _absolute_path(overrides[EXPORT_PATH_KEY])
            if EXPORT_PATH_KEY in overrides
            else trainer_root / EXPORTS_SUBDIR
        )
        artifact_store = None
        if self.artifact_store_enabled:
            if TRIALS_DIR_KEY in overrides:
                raise CheckpointLayoutError(
                    "container.artifact_store.enabled=true owns terminal_bench_config.trials_dir; "
                    "remove the explicit trials_dir override"
                )
            artifact_store = paths_for_run(trainer_root)
            trials_dir = artifact_store.trials
        else:
            trials_dir = (
                _absolute_path(overrides[TRIALS_DIR_KEY])
                if TRIALS_DIR_KEY in overrides
                else trainer_root / TRACE_JOBS_SUBDIR
            )
        return RLRunPaths(
            job_name=self.job_name,
            checkpoint_dir=checkpoint_dir,
            export_dir=export_dir,
            trials_dir=trials_dir,
            resume_mode=resume_mode,
            resume_path=resume_path,
            resume_policy=resume_policy,
            artifact_store=artifact_store,
        )
