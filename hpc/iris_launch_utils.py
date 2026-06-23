"""Compatibility imports for the Iris launcher package.

New code should import from ``hpc.iris`` modules directly. This module remains
for existing launchers and tests that import ``hpc.iris_launch_utils``.
"""

from hpc.iris.launcher import IrisLauncher
from hpc.iris.outputs import DEFAULT_GCS_OUTPUT_ROOT
from hpc.iris.regions import (
    assert_yaml_regions_match_pin,
    discover_region_for_tpu,
    parse_tpu_vm_count,
)
from hpc.iris.settings import DEFAULT_CLUSTER_CONFIG, DEFAULT_PRIORITY, DEFAULT_TASK_IMAGE

__all__ = [
    "DEFAULT_CLUSTER_CONFIG",
    "DEFAULT_GCS_OUTPUT_ROOT",
    "DEFAULT_PRIORITY",
    "DEFAULT_TASK_IMAGE",
    "IrisLauncher",
    "assert_yaml_regions_match_pin",
    "discover_region_for_tpu",
    "parse_tpu_vm_count",
]
