from src.eval.evaluate import evaluate_rows
from src.eval.metrics import exact_match, token_f1


def test_text_metrics_normalize_punctuation_and_case() -> None:
    assert exact_match("Grasper.", "grasper") == 1.0
    assert token_f1("calot triangle", "calot triangle dissection") > 0.0


def test_evaluate_phase_and_tool_tasks() -> None:
    rows = [
        {
            "task_type": "phase",
            "prediction": "Preparation",
            "ground_truth": "Preparation",
        },
        {
            "task_type": "tool_type",
            "prediction": "grasper, hook",
            "ground_truth": "hook, grasper",
        },
    ]

    metrics = evaluate_rows(rows)

    assert metrics["num_samples"] == 2
    assert metrics["by_task"]["phase"]["accuracy"] == 1.0
    assert metrics["by_task"]["tool_type"]["exact_match"] == 1.0
