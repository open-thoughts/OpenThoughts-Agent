"""Decorators for generation scripts."""

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def gpu_required(func: F) -> F:
    """Mark run_task_generation as requiring GPU resources.

    When applied, tooling can enforce that GPU-backed engines (e.g. local vLLM)
    are fully available before task generation starts.
    """
    setattr(func, "_gpu_required", True)
    return func


__all__ = ["gpu_required"]
