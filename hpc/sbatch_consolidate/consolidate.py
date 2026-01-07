#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def _reset_dir(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _copy_repo_files(
    src: Path,
    dest: Path,
    *,
    allowed_suffixes: Optional[set[str]] = None,
    skip_existing: bool = False,
) -> None:
    if not src.exists():
        return

    skip_suffixes = {".bin", ".pt", ".ckpt"}
    skip_prefixes = ("pytorch_model", "global_step", "checkpoint", "mp_rank", "zero_pp_rank")
    skip_dirs = {"logs", "wandb"}

    for item in src.rglob("*"):
        if item.is_dir():
            relative = item.relative_to(src)
            if relative.name in skip_dirs or relative.name.startswith(skip_prefixes) or relative.name.startswith("checkpoint"):
                continue
            continue

        name = item.name
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            continue
        if name.endswith(".safetensors"):
            continue
        suffix = item.suffix.lower()
        if suffix in skip_suffixes:
            continue
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue

        relative = item.relative_to(src)
        target = dest / relative
        if skip_existing and target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target, follow_symlinks=False)


def _infer_model_type(candidates: Iterable[Path]) -> Optional[str]:
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return None

    for root in candidates:
        try:
            cfg = AutoConfig.from_pretrained(
                str(root),
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            print(f"Warning: failed to load config from {root}: {exc}", file=sys.stderr)
            continue

        model_type = getattr(cfg, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return model_type.lower()

        architectures = getattr(cfg, "architectures", None)
        if architectures:
            for arch in architectures:
                if isinstance(arch, str) and arch:
                    return arch.lower()
    return None


def _zero_to_fp32_script(project_root: Path) -> Path:
    script_path = project_root / "scripts" / "consolidate" / "zero_to_fp32.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Bundled zero_to_fp32.py not found at {script_path}")
    return script_path


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value for --push-to-hub: {value}")


def _load_run_summary(paths: Iterable[Path]) -> dict:
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to parse run summary at {path}: {exc}", file=sys.stderr)
    return {}


def _extract_wandb_run(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    entity = project = run_id = None
    if len(parts) >= 4 and parts[2] == "runs":
        entity, project, _, run_id = parts[:4]
    elif len(parts) >= 3:
        entity, project, run_id = parts[:3]
    env_entity = os.environ.get("WANDB_ENTITY")
    env_project = os.environ.get("WANDB_PROJECT")
    if env_entity:
        entity = env_entity
    if env_project:
        project = env_project
    if entity and project and run_id:
        return f"{entity}/{project}/{run_id}"
    return None


def _maybe_register_with_db(project_root: Path, hf_repo: str, run_summary: dict, base_repo_hint: Optional[str]) -> None:
    wandb_run = _extract_wandb_run(run_summary.get("wandb_link"))
    dataset_name = run_summary.get("dataset_name")
    base_model = run_summary.get("base_model_name") or base_repo_hint
    training_type = run_summary.get("training_type") or "SFT"
    if not wandb_run or not dataset_name or not base_model:
        print("Warning: Missing wandb/dataset/base-model info; skipping Supabase registration.", file=sys.stderr)
        return

    script_path = project_root / "scripts" / "database" / "manual_db_push.py"
    if not script_path.exists():
        print(f"Warning: manual_db_push not found at {script_path}; skipping registration.", file=sys.stderr)
        return

    cmd = [
        sys.executable,
        str(script_path),
        "--hf-model-id",
        hf_repo,
        "--wandb-run",
        wandb_run,
        "--dataset-name",
        dataset_name,
        "--base-model",
        base_model,
        "--training-type",
        training_type,
    ]
    agent_name = run_summary.get("agent_name")
    if agent_name:
        cmd.extend(["--agent-name", agent_name])
    print("Registering model in Supabase via manual_db_push...")
    subprocess.check_call(cmd, cwd=str(project_root))
    print("✓ Supabase registration complete.")


def consolidate(
    input_spec: str,
    input_kind: str,
    base_repo: Optional[str],
    output_repo: str,
    workdir: Path,
    commit_message: str,
    project_root: Path,
    push_to_hub: bool,
) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required for consolidation.")

    if not os.environ.get("DCFT"):
        os.environ["DCFT"] = str(project_root)

    staged_input_dir = workdir / "input_stage"
    base_dir = workdir / "base_repo"
    merged_dir = workdir / "merged_weights"
    final_dir = workdir / "final_repo"

    print(f"Working directory: {workdir}")
    _reset_dir(workdir)
    _reset_dir(staged_input_dir)
    _reset_dir(base_dir)
    _reset_dir(merged_dir)
    _reset_dir(final_dir)

    normalized_kind = (input_kind or "").strip().lower()
    if normalized_kind not in {"repo", "local"}:
        guessed_path = Path(input_spec).expanduser()
        normalized_kind = "local" if guessed_path.exists() else "repo"

    if normalized_kind == "repo":
        print(f"Downloading input repo: {input_spec}")
        snapshot_download(
            repo_id=input_spec,
            repo_type="model",
            local_dir=staged_input_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
    else:
        source_path = Path(input_spec).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Local consolidation input path not found: {source_path}"
            )
        print(f"Staging local input from {source_path}")
        shutil.copytree(source_path, staged_input_dir, dirs_exist_ok=True)

    if base_repo:
        print(f"Downloading base repo: {base_repo}")
        snapshot_download(
            repo_id=base_repo,
            repo_type="model",
            local_dir=base_dir,
            local_dir_use_symlinks=False,
            token=token,
        )

    zero_to_fp32 = _zero_to_fp32_script(project_root)
    print("Running zero_to_fp32 consolidation...")
    subprocess.check_call(
        [
            sys.executable,
            str(zero_to_fp32),
            str(staged_input_dir),
            str(merged_dir),
            "--safe_serialization",
        ]
    )

    print("Collecting ancillary files...")
    _copy_repo_files(staged_input_dir, final_dir)
    if base_repo:
        _copy_repo_files(
            base_dir,
            final_dir,
            allowed_suffixes={".json", ".jsonl", ".jinja"},
            skip_existing=True,
        )

    for item in list(final_dir.rglob("*")):
        if item.is_dir() and (item.name in {"logs", "wandb"} or item.name.startswith("checkpoint")):
            shutil.rmtree(item, ignore_errors=True)

    print("Copying merged weights...")
    for item in merged_dir.rglob("*"):
        if item.is_dir():
            continue
        relative = item.relative_to(merged_dir)
        destination = final_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination, follow_symlinks=False)

    detected_model_type = _infer_model_type([final_dir, base_dir, staged_input_dir])
    if detected_model_type:
        print(f"Detected model type: {detected_model_type}")
    else:
        print("Warning: unable to infer model type from configuration.", file=sys.stderr)

    run_summary = _load_run_summary([staged_input_dir / "run_summary.json", final_dir / "run_summary.json"])

    if push_to_hub:
        print(f"Uploading consolidated checkpoint to Hugging Face repo {output_repo}...")
        api = HfApi()
        try:
            existing_paths = api.list_repo_files(repo_id=output_repo, repo_type="model", token=token)
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                print(f"Repo {output_repo} not found; creating it before upload...")
                api.create_repo(repo_id=output_repo, repo_type="model", private=False, token=token, exist_ok=True)
                existing_paths = []
            else:
                raise
        operations = [CommitOperationDelete(path_in_repo=path) for path in existing_paths]

        for item in final_dir.rglob("*"):
            if item.is_file():
                relative = item.relative_to(final_dir).as_posix()
                operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(item)))

        api.create_commit(
            repo_id=output_repo,
            repo_type="model",
            operations=operations,
            commit_message=commit_message,
            token=token,
        )
        print(f"✓ Uploaded consolidated checkpoint to {output_repo}")

        _maybe_register_with_db(project_root, output_repo, run_summary, base_repo)
    else:
        print("push_to_hub disabled; skipping upload and database registration.")

    print("Cleaning up temporary directories...")
    shutil.rmtree(staged_input_dir, ignore_errors=True)
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.rmtree(merged_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate ZeRO sharded checkpoints into safetensors and upload.")
    parser.add_argument("--input", required=True, help="Input source: either a local directory or a Hugging Face repo ID.")
    parser.add_argument("--input-kind", choices=["local", "repo"], default="repo", help="Explicit input type to avoid re-detection confusion.")
    parser.add_argument("--output-repo", required=True, help="Destination Hugging Face repo to upload the consolidated checkpoint.")
    parser.add_argument("--base-repo", default="", help="Base repo to copy ancillary files from.")
    parser.add_argument("--workdir", required=True, help="Working directory for consolidation artifacts.")
    parser.add_argument("--commit-message", required=True, help="Commit message for the upload.")
    parser.add_argument("--project-root", required=True, help="Path to OpenThoughts-Agent project root.")
    parser.add_argument("--push-to-hub", default="true", help="Whether to upload/register the merged repo (default: true).")
    args = parser.parse_args()

    consolidate(
        input_spec=args.input,
        input_kind=args.input_kind,
        base_repo=args.base_repo or None,
        output_repo=args.output_repo,
        workdir=Path(args.workdir).expanduser().resolve(),
        commit_message=args.commit_message,
        project_root=Path(args.project_root).expanduser().resolve(),
        push_to_hub=_str_to_bool(args.push_to_hub),
    )


if __name__ == "__main__":
    main()
