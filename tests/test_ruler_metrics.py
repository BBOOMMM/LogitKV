import pandas as pd

from evaluation.ruler.calculate_metrics import calculate_metrics


def test_ruler_metrics_include_macro_average_without_counting_it_as_a_task():
    df = pd.DataFrame(
        [
            {"task": "cwe", "predicted_answer": "alpha", "answer": ["alpha"]},
            {"task": "fwe", "predicted_answer": "wrong", "answer": ["beta"]},
            {"task": "qa_1", "predicted_answer": "gamma", "answer": ["gamma"]},
        ]
    )

    metrics = calculate_metrics(df)

    assert metrics["cwe"] == {"string_match": 100.0}
    assert metrics["fwe"] == {"string_match": 0.0}
    assert metrics["qa_1"] == {"string_match": 100.0}
    assert metrics["macro_average"] == {"string_match": 66.67}
