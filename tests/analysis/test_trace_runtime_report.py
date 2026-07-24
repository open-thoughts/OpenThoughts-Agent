from __future__ import annotations

from pathlib import Path

from scripts.analysis.trace_runtime_report import (
    ExperimentRuntime,
    JobRuntime,
    StageSample,
    stage_runtime_analysis,
)


def test_stage_runtime_analysis_reports_quantiles_and_top_exception_paths():
    samples = [
        StageSample(
            Path("task-1/result.json"),
            {"overall": 12.0, "agent_execution": 10.0},
            False,
        ),
        StageSample(
            Path("task-2/result.json"), {"overall": 24.0, "agent_execution": 20.0}, True
        ),
        StageSample(
            Path("task-3/result.json"),
            {"overall": 36.0, "agent_execution": 30.0},
            False,
        ),
        StageSample(
            Path("task-4/result.json"), {"overall": 48.0, "agent_execution": 40.0}, True
        ),
    ]
    analysis, top_exception_paths = stage_runtime_analysis(
        [
            ExperimentRuntime(
                "experiment", [JobRuntime("job", None, stage_samples=samples)]
            )
        ],
        [0.0, 0.5, 1.0],
    )

    overall = analysis["stages"]["overall"]
    assert overall["count"] == 4
    assert overall["mean_seconds"] == 30.0
    assert overall["quantiles"] == [
        {"quantile": 0.0, "seconds": 12.0},
        {"quantile": 0.5, "seconds": 30.0},
        {"quantile": 1.0, "seconds": 48.0},
    ]
    assert analysis["agent_execution_exception_frequency_by_quantile"] == [
        {
            "lower_quantile": 0.0,
            "upper_quantile": 0.5,
            "count": 2,
            "exceptions": 1,
            "exception_rate": 0.5,
        },
        {
            "lower_quantile": 0.5,
            "upper_quantile": 1.0,
            "count": 2,
            "exceptions": 1,
            "exception_rate": 0.5,
        },
    ]
    assert top_exception_paths == [Path("task-4/result.json")]
