import csv
import json
import math
from pathlib import Path

from sklearn.metrics import accuracy_score, root_mean_squared_error

CONFIG = json.loads("{\"target\": \"ConcreteCompressiveStrength\", \"features\": [\"Age\", \"BlastFurnaceSlag\", \"Cement\", \"CoarseAggregate\", \"FineAggregate\", \"FlyAsh\", \"Superplasticizer\", \"Water\"], \"numeric\": [\"Age\", \"BlastFurnaceSlag\", \"Cement\", \"CoarseAggregate\", \"FineAggregate\", \"FlyAsh\", \"Superplasticizer\", \"Water\"], \"classes\": [], \"metric\": \"rmse\", \"task_type\": \"regression\", \"prediction_kind\": \"numeric\", \"estimator\": \"random_forest_regressor\", \"seed\": 42}")


def score(submission, hidden_labels):
    if not submission.is_file() or submission.is_symlink():
        raise ValueError("Missing regular submission file")
    if submission.stat().st_size > 10_000_000:
        raise ValueError("Submission exceeds 10 MB")
    with hidden_labels.open(encoding="utf-8", newline="") as stream:
        expected = {r["row_id"]: float(r["target"]) for r in csv.DictReader(stream)}
    with submission.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_id", "prediction"]:
            raise ValueError("Expected columns row_id,prediction")
        predictions = {}
        for row in reader:
            if set(row) != {"row_id", "prediction"} or row["prediction"] is None:
                raise ValueError("Malformed CSV row")
            row_id = row["row_id"]
            if row_id in predictions:
                raise ValueError("Duplicate row_id")
            value = float(row["prediction"])
            if not math.isfinite(value):
                raise ValueError("Non-finite prediction")
            if CONFIG["classes"] and (not value.is_integer() or not 0 <= value < len(CONFIG["classes"])):
                raise ValueError("Invalid class code")
            predictions[row_id] = value
    if set(predictions) != set(expected):
        raise ValueError("Missing or extra row IDs")
    ids = sorted(expected)
    truth, predicted = [expected[i] for i in ids], [predictions[i] for i in ids]
    metric = CONFIG["metric"]
    value = float(root_mean_squared_error(truth, predicted) if metric == "rmse" else accuracy_score(truth, predicted))
    if not math.isfinite(value):
        raise ValueError("Non-finite score")
    return {"valid": 1.0, metric: value}


def main():
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    rewards = {"valid": 0.0, CONFIG["metric"]: 0.0}
    try:
        rewards = score(Path("/app/submission.csv"), Path("/tests/hidden_labels.csv"))
    except (ValueError, OSError, UnicodeError, csv.Error) as exc:
        (output / "error.txt").write_text(str(exc))
    (output / "reward.json").write_text(json.dumps(rewards, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
