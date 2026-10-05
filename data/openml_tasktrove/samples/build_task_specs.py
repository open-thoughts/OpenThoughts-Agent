"""Represent the checked-in OpenML Harbor samples as TaskCompendium TaskSpecs."""

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "taskcompendium @ git+https://github.com/marin-community/marin.git@9394c740befa71f31bb6e908b6305171d5baf01a#subdirectory=lib/taskcompendium",
# ]
# ///

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    ConversationInput,
    DatasetPath,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)

TASKCOMPENDIUM_REVISION = "9394c740befa71f31bb6e908b6305171d5baf01a"
IMPORTER_REVISION = "8d8f24419d203755f5a40b1db45b24815c97ec6e"
SAMPLES_DIR = Path(__file__).parent / "generated"
OUTPUT_DIR = Path(__file__).parent / "task_specs"

RESOURCE_FILES = {
    "environment/data/train.csv": ("data/train.csv", "train.csv", "worker"),
    "environment/data/test.csv": ("data/test.csv", "test.csv", "worker"),
    "environment/data/ATTRIBUTION.md": (
        "data/ATTRIBUTION.md",
        "ATTRIBUTION.md",
        "worker",
    ),
    "tests/hidden_labels.csv": ("hidden_labels.csv", "hidden_labels.csv", "verifier"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verifier_config(sample_dir: Path) -> dict:
    """Read the literal CONFIG embedded in the generated verifier without executing it."""
    tree = ast.parse((sample_dir / "tests/verifier.py").read_text())
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "CONFIG"
            for target in statement.targets
        ):
            continue
        call = statement.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "json"
            or call.func.attr != "loads"
            or len(call.args) != 1
        ):
            break
        return json.loads(ast.literal_eval(call.args[0]))
    raise ValueError(f"Could not find the literal verifier CONFIG in {sample_dir}")


def semantic_context(provenance: dict, config: dict) -> str:
    task_type = provenance["task_type"]
    target = provenance["target"]
    lines = [
        f"Build a {task_type} model for the target column `{target}`.",
        "",
        "Use the labelled training data in `data/train.csv` to predict the target "
        "for every row in `data/test.csv`. Feature values may be missing. Do not use "
        "`row_id` as a feature.",
        "",
        "Return a CSV prediction artifact with exactly two columns, in order: "
        "`row_id,prediction`. Include every test row ID exactly once, with no extra rows.",
    ]
    if task_type == "classification":
        classes = config["classes"]
        lines.extend(
            [
                "Predictions must be integer class IDs from 0 through "
                f"{len(classes) - 1}.",
                "Class meanings, in ID order: "
                + json.dumps(classes, ensure_ascii=False)
                + ".",
            ]
        )
    else:
        lines.append("Predictions must be finite numeric values.")
    lines.append(
        f"Predictions are evaluated using {provenance['metric']} "
        f"({'higher' if provenance['metric'] == 'accuracy' else 'lower'} is better)."
    )
    return "\n".join(lines)


def verifier_parameters(provenance: dict, config: dict) -> dict:
    parameters = {
        "task_type": provenance["task_type"],
        "metric": provenance["metric"],
        "artifact_format": "csv",
        "id_column": "row_id",
        "prediction_column": "prediction",
        "target": provenance["target"],
        "required_columns": ["row_id", "prediction"],
        "require_column_order": True,
        "require_all_test_rows": True,
        "allow_extra_row_ids": False,
        "allow_duplicate_row_ids": False,
        "require_finite_predictions": True,
        "require_regular_file": True,
        "max_file_bytes": 10_000_000,
    }
    if provenance["task_type"] == "classification":
        parameters["valid_classes"] = list(range(len(config["classes"])))
    return parameters


def build_task_spec(sample_dir: Path, output_root: Path) -> TaskSpec:
    provenance = json.loads((sample_dir / "provenance.json").read_text())
    config = verifier_config(sample_dir)
    task_id = provenance["task_id"]
    if sample_dir.name != task_id:
        raise ValueError(
            f"Sample directory and provenance task IDs differ: {sample_dir}"
        )
    for field in ("task_type", "target", "metric"):
        if provenance[field] != config[field]:
            raise ValueError(
                f"Verifier {field} does not match provenance for {task_id}"
            )

    output_dir = output_root / task_id
    resources_dir = output_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    grouped_resources = {"worker": [], "verifier": []}
    for source_name, (
        workspace_path,
        resource_name,
        visibility,
    ) in RESOURCE_FILES.items():
        source_path = sample_dir / source_name
        resource_path = resources_dir / resource_name
        shutil.copyfile(source_path, resource_path)
        source_hash = sha256(source_path)
        if sha256(resource_path) != source_hash:
            raise ValueError(f"Copied resource hash mismatch: {resource_path}")
        grouped_resources[visibility].append(
            TaskResource(
                path=workspace_path,
                source=DatasetPath(
                    path=f"{task_id}/resources/{resource_name}",
                    sha256=source_hash,
                ),
            )
        )

    specification = TaskSpec(
        id=task_id,
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=semantic_context(provenance, config),
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("filesystem", "shell", "process"),
            working_directory="/app",
        ),
        answer_type=AnswerType.FILE,
        resources=ResourceGroups(
            worker=tuple(grouped_resources["worker"]),
            verifier=tuple(grouped_resources["verifier"]),
        ),
        verifier=VerifierSpec(
            kind="tabular_predictions",
            parameters_json=json.dumps(
                verifier_parameters(provenance, config),
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
        source=Source(
            dataset=f"openml/{provenance['openml_dataset_id']}",
            revision=str(provenance["dataset_version"]),
            row=f"task-{provenance['openml_task_id']}:{provenance['target']}",
            importer_revision=IMPORTER_REVISION,
        ),
        tags=("openml", provenance["task_type"], provenance["metric"]),
    )
    if SCHEMA_VERSION != "0.18":
        raise RuntimeError(
            f"Expected TaskCompendium schema 0.18 at {TASKCOMPENDIUM_REVISION}, "
            f"found {SCHEMA_VERSION}"
        )
    serialized = specification.model_dump_json(indent=2)
    if TaskSpec.model_validate_json(serialized) != specification:
        raise ValueError(f"TaskSpec JSON did not round-trip for {task_id}")
    (output_dir / "specification.json").write_text(serialized + "\n")
    return specification


def run(
    samples_dir: Path = SAMPLES_DIR, output_dir: Path = OUTPUT_DIR, task_ids=()
) -> list[TaskSpec]:
    selected = set(task_ids)
    sample_dirs = [path for path in sorted(samples_dir.iterdir()) if path.is_dir()]
    if selected:
        sample_dirs = [path for path in sample_dirs if path.name in selected]
        missing = selected.difference(path.name for path in sample_dirs)
        if missing:
            raise ValueError(f"Unknown sample task IDs: {sorted(missing)}")
    return [build_task_spec(sample_dir, output_dir) for sample_dir in sample_dirs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--task-id", action="append", default=[])
    args = parser.parse_args()
    specifications = run(args.samples_dir, args.output_dir, args.task_id)
    print(
        f"Wrote {len(specifications)} schema-{SCHEMA_VERSION} TaskSpecs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
