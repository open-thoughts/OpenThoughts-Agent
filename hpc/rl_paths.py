"""Resolve durable RL state paths and validate checkpoint resume contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Sequence


LATEST_CHECKPOINT_FILE = "latest_ckpt_global_step.txt"
GLOBAL_STEP_PATTERN = re.compile(r"global_step_(\d+)$")


class CheckpointLayoutError(ValueError):
    """A checkpoint path cannot satisfy the requested resume contract."""


class AmbiguousCheckpointError(CheckpointLayoutError):
    """Multiple run roots contain the same highest checkpoint step."""


class RLResumeMode(StrEnum):
    NONE = "none"
    LATEST = "latest"
    FROM_PATH = "from_path"


class RLPathDecision(StrEnum):
    NEW = "new"
    RESUME = "resume"
    EXPLICIT_FRESH = "explicit-fresh"


@dataclass(frozen=True)
class CheckpointCandidate:
    state_root: Path
    checkpoint_dir: Path
    checkpoint_path: Path
    step: int


@dataclass(frozen=True)
class RLRunPaths:
    """Resolved paths consumed by the launcher and SkyRL configuration."""

    job_name: str
    state_root: Path
    checkpoint_dir: Path
    export_dir: Path
    trials_dir: Path
    resume_mode: RLResumeMode
    resume_path: Path | None
    decision: RLPathDecision

    def describe(self) -> str:
        if self.resume_path is not None:
            action = f"RESUME from {self.resume_path}"
        elif self.decision is RLPathDecision.EXPLICIT_FRESH:
            action = "EXPLICIT FRESH START"
        else:
            action = "NEW RUN (no checkpoint found)"
        return (
            f"[rl_paths] {action}; resume_mode={self.resume_mode.value}; "
            f"checkpoints={self.checkpoint_dir}; exports={self.export_dir}; trials={self.trials_dir}"
        )


def _override_values(overrides: Sequence[str]) -> dict[str, str]:
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


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


class RLPathManager:
    """Own checkpoint discovery and all durable RL path resolution."""

    def __init__(self, job_name: str, canonical_root: Path, launch_root: Path):
        self.job_name = job_name
        self.canonical_root = canonical_root.expanduser().resolve()
        self.launch_root = launch_root.expanduser().resolve()

    def resolve(
        self,
        *,
        skyrl_overrides: Sequence[str] = (),
        fresh_start_requested: bool = False,
    ) -> RLRunPaths:
        overrides = _override_values(skyrl_overrides)
        requested_mode = overrides.get("trainer.resume_mode")
        requested_resume_path = overrides.get("trainer.resume_path")
        if requested_mode in {"", "null", "None", "~"}:
            requested_mode = RLResumeMode.NONE.value
        if requested_resume_path in {"", "null", "None", "~"}:
            requested_resume_path = None

        if fresh_start_requested and requested_mode not in (None, RLResumeMode.NONE.value):
            raise CheckpointLayoutError("A fresh start cannot also request checkpoint resume")
        if requested_resume_path and requested_mode != RLResumeMode.FROM_PATH.value:
            raise CheckpointLayoutError("trainer.resume_path requires trainer.resume_mode=from_path")

        state_root = self.launch_root if fresh_start_requested else self.canonical_root
        checkpoint_dir = self._configured_checkpoint_dir(overrides, state_root)
        state_root = self._state_root_for_checkpoint_dir(checkpoint_dir, state_root)

        if fresh_start_requested or requested_mode == RLResumeMode.NONE.value:
            return self._resolved_paths(
                state_root,
                checkpoint_dir,
                overrides,
                resume_mode=RLResumeMode.NONE,
                resume_path=None,
                decision=RLPathDecision.EXPLICIT_FRESH,
            )

        if requested_mode == RLResumeMode.LATEST.value:
            candidate = self._checkpoint_candidate(state_root, checkpoint_dir, required=True)
            assert candidate is not None
            return self._resolved_paths(
                state_root,
                checkpoint_dir,
                overrides,
                resume_mode=RLResumeMode.LATEST,
                resume_path=candidate.checkpoint_path,
                decision=RLPathDecision.RESUME,
            )

        if requested_mode == RLResumeMode.FROM_PATH.value:
            if not requested_resume_path:
                raise CheckpointLayoutError("trainer.resume_mode=from_path requires trainer.resume_path")
            resume_path = self._validate_explicit_resume_path(_absolute_path(requested_resume_path))
            if "trainer.ckpt_path" in overrides and checkpoint_dir != resume_path.parent:
                raise CheckpointLayoutError(
                    f"trainer.resume_path {resume_path} is not under trainer.ckpt_path {checkpoint_dir}"
                )
            checkpoint_dir = resume_path.parent
            state_root = self._state_root_for_checkpoint_dir(checkpoint_dir, state_root)
            return self._resolved_paths(
                state_root,
                checkpoint_dir,
                overrides,
                resume_mode=RLResumeMode.FROM_PATH,
                resume_path=resume_path,
                decision=RLPathDecision.RESUME,
            )

        if requested_mode is not None:
            raise CheckpointLayoutError(f"Unknown trainer.resume_mode: {requested_mode}")

        if "trainer.ckpt_path" in overrides:
            candidate = self._checkpoint_candidate(state_root, checkpoint_dir, required=False)
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
                decision=RLPathDecision.NEW,
            )

        highest_step = max(candidate.step for candidate in candidates)
        highest = [candidate for candidate in candidates if candidate.step == highest_step]
        if len(highest) != 1:
            paths = ", ".join(str(candidate.checkpoint_path) for candidate in highest)
            raise AmbiguousCheckpointError(
                f"Multiple run roots contain global_step_{highest_step}: {paths}. "
                "Set trainer.resume_mode=from_path and trainer.resume_path explicitly."
            )

        selected = highest[0]
        return self._resolved_paths(
            selected.state_root,
            selected.checkpoint_dir,
            overrides,
            resume_mode=RLResumeMode.FROM_PATH,
            resume_path=selected.checkpoint_path,
            decision=RLPathDecision.RESUME,
        )

    def _configured_checkpoint_dir(self, overrides: dict[str, str], state_root: Path) -> Path:
        configured = overrides.get("trainer.ckpt_path")
        if configured:
            return _absolute_path(configured)
        return state_root / self.job_name / "checkpoints"

    def _state_root_for_checkpoint_dir(self, checkpoint_dir: Path, fallback: Path) -> Path:
        if checkpoint_dir.name == "checkpoints" and checkpoint_dir.parent.name == self.job_name:
            return checkpoint_dir.parent.parent
        return fallback

    def _checkpoint_candidates(self) -> list[CheckpointCandidate]:
        candidates: list[CheckpointCandidate] = []
        for state_root in self._candidate_state_roots():
            checkpoint_dir = state_root / self.job_name / "checkpoints"
            candidate = self._checkpoint_candidate(state_root, checkpoint_dir, required=False)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _candidate_state_roots(self) -> list[Path]:
        roots = [self.canonical_root]
        parent = self.canonical_root.parent
        if not parent.is_dir():
            return roots
        fork_pattern = re.compile(rf"{re.escape(self.canonical_root.name)}_(\d+)$")
        roots.extend(
            path.resolve() for path in parent.iterdir() if path.is_dir() and fork_pattern.fullmatch(path.name)
        )
        return roots

    def _checkpoint_candidate(
        self,
        state_root: Path,
        checkpoint_dir: Path,
        *,
        required: bool,
    ) -> CheckpointCandidate | None:
        marker_path = checkpoint_dir / LATEST_CHECKPOINT_FILE
        step_dirs = self._step_directories(checkpoint_dir)
        if not marker_path.is_file():
            if step_dirs:
                raise CheckpointLayoutError(
                    f"Checkpoint directories exist under {checkpoint_dir}, but {LATEST_CHECKPOINT_FILE} is missing"
                )
            if required:
                raise CheckpointLayoutError(
                    f"trainer.resume_mode=latest requires a usable checkpoint under {checkpoint_dir}"
                )
            return None

        try:
            step = int(marker_path.read_text().strip())
        except (OSError, ValueError) as error:
            raise CheckpointLayoutError(f"Invalid checkpoint marker: {marker_path}") from error

        checkpoint_path = checkpoint_dir / f"global_step_{step}"
        if not checkpoint_path.is_dir():
            raise CheckpointLayoutError(f"Checkpoint marker {marker_path} names missing {checkpoint_path.name}")
        if step_dirs and max(step_dirs) != step:
            raise CheckpointLayoutError(
                f"Checkpoint marker {marker_path} names global_step_{step}, "
                f"but global_step_{max(step_dirs)} also exists"
            )
        return CheckpointCandidate(state_root, checkpoint_dir, checkpoint_path, step)

    @staticmethod
    def _step_directories(checkpoint_dir: Path) -> list[int]:
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
        if not resume_path.is_dir() or GLOBAL_STEP_PATTERN.fullmatch(resume_path.name) is None:
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
        decision: RLPathDecision,
    ) -> RLRunPaths:
        trainer_root = state_root / self.job_name
        export_dir = (
            _absolute_path(overrides["trainer.export_path"])
            if "trainer.export_path" in overrides
            else trainer_root / "exports"
        )
        trials_dir = (
            _absolute_path(overrides["terminal_bench_config.trials_dir"])
            if "terminal_bench_config.trials_dir" in overrides
            else trainer_root / "trace_jobs"
        )
        return RLRunPaths(
            job_name=self.job_name,
            state_root=state_root,
            checkpoint_dir=checkpoint_dir,
            export_dir=export_dir,
            trials_dir=trials_dir,
            resume_mode=resume_mode,
            resume_path=resume_path,
            decision=decision,
        )
