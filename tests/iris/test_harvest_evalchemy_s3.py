from pathlib import Path

from scripts.iris import harvest_evalchemy_s3 as harvester


def test_task_metrics_keeps_scores_and_ignores_stderr():
    metrics = harvester.task_metrics(
        {
            "results": {
                "HumanEvalPlus": {
                    "pass@1": 0.53125,
                    "pass@1_stderr": 0.02,
                    "unrelated": "ignored",
                }
            }
        }
    )

    assert metrics == {"HumanEvalPlus:pass@1": 0.5312}


def test_sync_run_mirrors_every_selected_object(monkeypatch, tmp_path):
    copied = []

    def fake_download(_s3, key):
        copied.append(key)
        return tmp_path / Path(key).name

    monkeypatch.setattr(harvester, "download", fake_download)

    keys = [
        "iris/marinbase-eval/model/run/results_a.json",
        "iris/marinbase-eval/model/run/samples.jsonl",
    ]
    assert harvester.sync_run(object(), keys) == {
        key: tmp_path / Path(key).name for key in keys
    }
    assert copied == keys
