"""Resolved Iris accelerator policy for OT-Agent launchers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class IrisTopology(Protocol):
    chip_count: int
    vm_count: int


class IrisJobApi(Protocol):
    def build_tpu_alternatives(self, tpu: str) -> Sequence[str]:
        ...

    def get_tpu_topology(self, tpu: str) -> IrisTopology:
        ...

    def parse_gpu_spec(self, gpu: str) -> tuple[str, int]:
        ...

    def build_resources(self, tpu: str | None, gpu: str | None, *, cpu: float, memory: str, disk: str) -> Any:
        ...

    def resolve_multinode_defaults(self, tpu: str | None, gpu: str | None, replicas: int | None) -> tuple[int, Any]:
        ...


class MarinIrisJobApi:
    def build_tpu_alternatives(self, tpu: str) -> Sequence[str]:
        from iris.cli.job import build_tpu_alternatives

        return build_tpu_alternatives(tpu)

    def get_tpu_topology(self, tpu: str) -> IrisTopology:
        from iris.cluster.tpu_topology import get_tpu_topology

        return get_tpu_topology(tpu)

    def parse_gpu_spec(self, gpu: str) -> tuple[str, int]:
        from iris.cli.job import parse_gpu_spec

        return parse_gpu_spec(gpu)

    def build_resources(self, tpu: str | None, gpu: str | None, *, cpu: float, memory: str, disk: str) -> Any:
        from iris.cli.job import build_resources

        return build_resources(tpu, gpu, cpu=cpu, memory=memory, disk=disk)

    def resolve_multinode_defaults(self, tpu: str | None, gpu: str | None, replicas: int | None) -> tuple[int, Any]:
        from iris.cli.job import resolve_multinode_defaults

        return resolve_multinode_defaults(tpu, gpu, replicas)


DEFAULT_IRIS_JOB_API = MarinIrisJobApi()


@dataclass(frozen=True)
class ResolvedIrisAccelerator:
    """OT-Agent's resolved launch-time view of an Iris accelerator."""

    kind: str
    tpu: str | None = None
    gpu: str | None = None
    tpu_variants: tuple[str, ...] = ()
    tpu_topology: IrisTopology | None = None
    gpu_variant: str | None = None
    gpu_count: int | None = None
    iris_api: IrisJobApi = field(default=DEFAULT_IRIS_JOB_API, repr=False, compare=False)

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        default_tpu: str,
        iris_api: IrisJobApi = DEFAULT_IRIS_JOB_API,
    ) -> "ResolvedIrisAccelerator":
        """Resolve CLI default handling and parse the selected accelerator."""
        gpu = getattr(args, "gpu", None)
        tpu = getattr(args, "tpu", None) or (None if gpu else default_tpu)

        if gpu and tpu:
            raise SystemExit("--gpu and --tpu are mutually exclusive; pass only one accelerator.")
        if not gpu and not tpu:
            raise SystemExit("Pass either --tpu <variant> or --gpu <variant>.")

        if tpu:
            tpu_variants = tuple(iris_api.build_tpu_alternatives(tpu))
            topology = iris_api.get_tpu_topology(tpu_variants[0])
            return cls(
                kind="tpu",
                tpu=tpu,
                tpu_variants=tpu_variants,
                tpu_topology=topology,
                iris_api=iris_api,
            )

        gpu_variant, gpu_count = iris_api.parse_gpu_spec(gpu)
        return cls(
            kind="gpu",
            gpu=gpu,
            gpu_variant=gpu_variant,
            gpu_count=gpu_count,
            iris_api=iris_api,
        )

    @property
    def is_tpu(self) -> bool:
        return self.kind == "tpu"

    @property
    def is_gpu(self) -> bool:
        return self.kind == "gpu"

    @property
    def primary_tpu(self) -> str | None:
        return self.tpu_variants[0] if self.tpu_variants else None

    @property
    def vm_count(self) -> int:
        if not self.is_tpu:
            return 1
        return self.tpu_topology.vm_count

    @property
    def downstream_eval_device_count(self) -> int:
        if self.is_tpu:
            return self.tpu_topology.chip_count
        return self.gpu_count or 1

    @property
    def default_extras(self) -> list[str]:
        return ["datagen-tpu"] if self.is_tpu else ["datagen"]

    @property
    def uses_iris_serve(self) -> bool:
        return self.is_tpu

    @property
    def needs_tpu_runtime_patch(self) -> bool:
        return self.is_tpu

    @property
    def label(self) -> str:
        if self.is_tpu:
            return f"TPU:        {self.tpu}  (vm_count={self.vm_count})"
        return f"GPU:        {self.gpu}"

    def build_resources(self, *, cpu: float, memory: str, disk: str):
        return self.iris_api.build_resources(self.tpu, self.gpu, cpu=cpu, memory=memory, disk=disk)

    def resolve_multinode_defaults(self, replicas: int | None):
        return self.iris_api.resolve_multinode_defaults(self.primary_tpu, self.gpu, replicas)
