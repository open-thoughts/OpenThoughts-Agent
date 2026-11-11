"""Task-type adapters for Harbor task construction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskHandler:
    key: str
    openml_ids: tuple[int, ...]
    split_type: str
    metric: str
    prediction_kind: str
    estimator: str
    instruction_template: str
    solution_template: str
    verifier_template: str
    encode_classes: bool


HANDLERS = {
    "classification": TaskHandler(
        key="classification",
        openml_ids=(1,),
        split_type="classification",
        metric="accuracy",
        prediction_kind="class_code",
        estimator="logistic_regression",
        instruction_template="task_types/classification.md.j2",
        solution_template="solution.py.j2",
        verifier_template="verifier.py.j2",
        encode_classes=True,
    ),
    "regression": TaskHandler(
        key="regression",
        openml_ids=(2,),
        split_type="regression",
        metric="rmse",
        prediction_kind="numeric",
        estimator="random_forest_regressor",
        instruction_template="task_types/regression.md.j2",
        solution_template="solution.py.j2",
        verifier_template="verifier.py.j2",
        encode_classes=False,
    ),
}
HANDLERS_BY_OPENML_ID = {
    task_type_id: handler
    for handler in HANDLERS.values()
    for task_type_id in handler.openml_ids
}


def handler_for(task_type_id):
    try:
        return HANDLERS_BY_OPENML_ID[int(task_type_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported OpenML task type: {task_type_id}") from exc


def supports(task_type_id):
    try:
        handler_for(task_type_id)
    except ValueError:
        return False
    return True
