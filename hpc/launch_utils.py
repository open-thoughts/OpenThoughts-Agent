"""
Utility helpers shared across HPC launch entry points.
"""

from __future__ import annotations

import os
import re
import shlex
import socket
import shutil
import subprocess
from collections import defaultdict
from typing import Any, Mapping, Optional

from hpc.hpc import detect_hpc

from .job_name_ignore_list import JOB_NAME_IGNORE_KEYS
from .arguments import JobType
from .sft_launch_utils import build_accelerate_config_block


def sanitize_repo_for_job(repo_id: str) -> str:
    """Return a filesystem-safe representation of a repo identifier."""

    safe = re.sub(r"[^A-Za-z0-9._\-]+", "-", repo_id.strip())
    safe = safe.strip("-_")
    return safe or "consolidate"


def sanitize_repo_component(value: Optional[str]) -> Optional[str]:
    """Extract the meaningful suffix from trace repositories (traces-<slug>)."""

    if not value:
        return None
    match = re.search(r"traces-([A-Za-z0-9._\-]+)", value)
    return match.group(1) if match else None


def derive_datagen_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct a fallback job name for datagen/trace launches."""

    def _sanitize_component(value: str) -> str:
        value = value.strip().rstrip("/")
        if "/" in value:
            value = value.split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "repo"

    parts: list[str] = ["datagen"]
    engine = cli_args.get("datagen_engine") or cli_args.get("trace_engine") or "engine"
    parts.append(str(engine or "engine"))

    repo_candidate = cli_args.get("datagen_target_repo") or cli_args.get("trace_target_repo")
    model_candidate = cli_args.get("datagen_model") or cli_args.get("trace_model")
    if model_candidate:
        parts.append(_sanitize_component(str(model_candidate)))
    elif repo_candidate:
        parts.append(_sanitize_component(str(repo_candidate)))

    job_name = "_".join(filter(None, parts))
    return job_name or "datagen_job"


def derive_consolidate_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct a consolidate-specific job name with a fixed suffix."""

    identifier_raw = (
        cli_args.get("consolidate_input")
        or cli_args.get("consolidate_output_repo")
        or cli_args.get("consolidate_base_repo")
        or "consolidate"
    )
    identifier = sanitize_repo_for_job(str(identifier_raw))
    suffix = "_consolidate"
    max_prefix_len = max(1, 96 - len(suffix))
    if len(identifier) > max_prefix_len:
        identifier = identifier[:max_prefix_len]
    return f"{identifier}{suffix}"


def derive_default_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct job names for non-datagen, non-consolidate workloads."""

    job_name_components: list[str] = []
    job_name_suffix: Optional[str] = None

    for key, value in cli_args.items():
        if not isinstance(value, (str, int, float)):
            continue
        if value == "None" or key in JOB_NAME_IGNORE_KEYS:
            continue

        if key == "seed":
            try:
                if float(value) == 42:
                    continue
            except (TypeError, ValueError):
                pass

        if key not in {"dataset", "model_name_or_path"}:
            job_name_components.append(str(key).replace("_", "-"))

        value_str = str(value)
        if value_str == "Qwen/Qwen2.5-32B-Instruct":
            job_name_suffix = "_32B"
        elif value_str == "Qwen/Qwen2.5-14B-Instruct":
            job_name_suffix = "_14B"
        elif value_str == "Qwen/Qwen2.5-3B-Instruct":
            job_name_suffix = "_3B"
        elif value_str == "Qwen/Qwen2.5-1.5B-Instruct":
            job_name_suffix = "_1.5B"
        else:
            job_name_components.append(value_str.split("/")[-1])

    job_name = "_".join(job_name_components)
    job_name = (
        job_name.replace("/", "_")
        .replace("?", "")
        .replace("*", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace('"', "")
        .replace(" ", "_")
    )
    if job_name_suffix:
        job_name += job_name_suffix

    if len(job_name) > 96:
        print("Truncating job name to less than HF limit of 96 characters...")
        job_name = "_".join(
            "-".join(segment[:4] for segment in chunk.split("-"))
            for chunk in job_name.split("_")
        )
        if len(job_name) > 96:
            raise ValueError(
                f"Job name {job_name} is still too long (96 characters) after truncation. "
                "Try renaming the dataset or providing a shorter YAML config."
            )

    return job_name or "ot_agent_job"


def get_job_name(cli_args: Mapping[str, Any]) -> str:
    """Derive a stable job name from user-provided CLI arguments."""

    job_type = str(cli_args.get("job_type", JobType.default_value()) or JobType.default_value()).lower()
    if job_type == JobType.CONSOLIDATE.value:
        return derive_consolidate_job_name(cli_args)
    if job_type == JobType.DATAGEN.value:
        return derive_datagen_job_name(cli_args)
    return derive_default_job_name(cli_args)

def _parse_optional_int(value: Any, label: str) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def _inject_env_block(text: str, env_map: dict) -> str:
    exports = []
    for k, v in env_map.items():
        if v in (None, ""):
            continue
        quoted = shlex.quote(str(v))
        exports.append(f"export {k}={quoted}")
    if not exports:
        return text
    lines = text.splitlines(True)
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    while idx < len(lines) and (
        lines[idx].startswith("#SBATCH")
        or lines[idx].strip() == ""
        or lines[idx].startswith("#")
    ):
        idx += 1
    return "".join(lines[:idx] + ["\n".join(exports) + "\n"] + lines[idx:])


def _ensure_dependency_directive(text: str, dependency: Optional[str]) -> str:
    if not dependency:
        return text

    directive_prefix = "#SBATCH --dependency"
    lines = text.splitlines()
    for line in lines:
        if directive_prefix in line:
            return text

    insert_idx = 0
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("#!"):
            insert_idx = 1
            continue
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            insert_idx = idx + 1
            continue
        if not stripped:
            insert_idx = idx + 1
            continue
        break

    dependency_line = f"#SBATCH --dependency={dependency}"
    lines.insert(insert_idx, dependency_line)
    new_text = "\n".join(lines)
    if text.endswith("\n"):
        new_text += "\n"
    return new_text


def _merge_dependencies(*deps: Optional[str]) -> Optional[str]:
    merged: list[str] = []
    for dep in deps:
        if not dep:
            continue
        dep_str = str(dep).strip()
        if not dep_str:
            continue
        merged.append(dep_str)
    if not merged:
        return None
    return ",".join(merged)


def launch_sbatch(sbatch_script_path, dependency=None, array: str | None = None) -> str:
    extra_args: list[str] = []
    if dependency is not None:
        extra_args.append(f"--dependency={dependency}")
    if array:
        extra_args.append(f"--array={array}")
    extra_flags = " ".join(extra_args)
    sbatch_cmd = f"sbatch {extra_flags} {sbatch_script_path}".strip()

    result = subprocess.run(
        sbatch_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stdout.strip()
        err = result.stderr.strip()
        combined = "\n".join(filter(None, [msg, err]))
        raise RuntimeError(
            f"sbatch command failed (code {result.returncode}): {sbatch_cmd}\n{combined}"
        )

    raw_output = (result.stdout or "").strip()
    job_id = raw_output.split()[::-1][0]
    print(
        f"Job {job_id} submitted"
        f"{f' with dependency {dependency}' if dependency else ''}"
        f"{f' and array {array}' if array else ''}."
    )
    return job_id


def update_exp_args(exp_args, args, *, explicit_keys: Optional[set[str]] = None):
    explicit_keys = set(explicit_keys or [])
    for key, value in args.items():
        if key.startswith("_"):
            continue

        has_existing = key in exp_args
        existing_value = exp_args.get(key)
        is_explicit = not explicit_keys or key in explicit_keys

        if value is None:
            if has_existing and is_explicit:
                del exp_args[key]
                print(f"Removed {key} from experiment arguments")
            continue

        if has_existing:
            if not is_explicit and value != existing_value:
                continue
            if value != existing_value:
                print(f"Overwrote {key} from {existing_value} to {value}")
        exp_args[key] = value
    return exp_args


def check_exists(local_path: str | os.PathLike[str]) -> bool:
    """Return True when ``local_path`` exists."""

    return os.path.exists(local_path)


def extract_template_keys(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        file = f.read()
    return re.findall(r"(?<!\$)\{([^{}]*)\}", file)


def fill_template(file_path: str, exp_args: dict, new_file_path: str) -> None:
    with open(file_path, "r") as f:
        file = f.read()

    file = re.sub(r"(?<!\$)\{([^{}]*)\}", lambda m: exp_args[m.group(1)], file)

    with open(new_file_path, "w") as f:
        f.write(file)


def _escape_bash_variables(text: str) -> str:
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] == "$" and i + 1 < length and text[i + 1] == "{":
            start = i
            depth = 1
            j = i + 2
            while j < length and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            inner = text[i + 2 : j - 1]
            escaped_inner = _escape_bash_variables(inner)
            result.append("${{" + escaped_inner + "}}")
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def construct_sbatch_script(exp_args: dict) -> str:
    base_script_path = exp_args["train_sbatch_path"]
    with open(base_script_path, "r") as f:
        base_script = f.read()

    kwargs = defaultdict(str, **exp_args)
    kwargs["accelerate_config_block"] = build_accelerate_config_block(exp_args)

    json_files_cat = re.findall(r"cat.*?<<EOT >.*?EOT", base_script, re.DOTALL)
    json_filenames = []
    for json_file in json_files_cat:
        json_file_name = re.match(
            r"cat.*?<<EOT >.*?(\S+).*?EOT", json_file, re.DOTALL
        ).group(1)
        json_filenames.append(json_file_name)

        base_script = re.sub(
            r"cat.*?<<EOT >.*?" + json_file_name.replace("$", "\\$") + r".*?EOT",
            f"cat {json_file_name}",
            base_script,
            count=1,
            flags=re.DOTALL,
        )

    base_script = _escape_bash_variables(base_script)

    time_limit = kwargs.get("time_limit")
    if time_limit is None:
        time_limit = "01:00:00"
        kwargs["time_limit"] = time_limit

    hpc = detect_hpc()
    hpc_name = hpc.name
    if hpc_name == "jureca" or hpc_name == "juwels":
        login_node = socket.gethostname().split(".")[0] + "i"
        if "{login_node}" in base_script:
            if kwargs.get("internet_node", False):
                if not shutil.which("proxychains4"):
                    raise RuntimeError("proxychains4 not found, please install it to use internet_node")
            base_script = base_script.replace("{login_node}", login_node)

    sbatch_script = base_script.format(**kwargs)
    sbatch_script = _ensure_dependency_directive(sbatch_script, exp_args.get("dependency"))

    env_block = {
        "DISABLE_VERSION_CHECK": "1",
    }
    stage_value = str(exp_args.get("stage") or "").lower()
    if exp_args.get("use_mca") and stage_value == "sft":
        env_block["USE_MCA"] = "1"
        os.environ.setdefault("USE_MCA", "1")

    sbatch_script = _inject_env_block(sbatch_script, env_block)

    for json_file, json_file_name in zip(json_files_cat, json_filenames):
        sbatch_script = sbatch_script.replace(f"cat {json_file_name}", json_file)

    sbatch_dir = os.path.join(kwargs["experiments_dir"], "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_script_path = os.path.join(sbatch_dir, f"{kwargs['job_name']}.sbatch")
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)
        print(f"Wrote sbatch script to {sbatch_script_path}")

    return sbatch_script_path


__all__ = [
    "derive_datagen_job_name",
    "_parse_optional_int",
    "_inject_env_block",
    "_ensure_dependency_directive",
    "_merge_dependencies",
    "launch_sbatch",
    "update_exp_args",
    "check_exists",
    "construct_sbatch_script",
    "extract_template_keys",
    "fill_template",
    "get_job_name",
    "sanitize_repo_for_job",
    "sanitize_repo_component",
]
