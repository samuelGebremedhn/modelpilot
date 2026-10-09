import pandas as pd
from automl_core import infer_task, profile_dataframe, prepare_xy, benchmark_models

def test_profile_and_task():
    df = pd.DataFrame({"x": [1,2,3,4,5,6], "cat": ["a","a","b","b","c","c"], "target": [0,0,1,1,1,0]})
    p = profile_dataframe(df)
    assert p.rows == 6
    assert "x" in p.numeric_columns
    assert infer_task(df["target"]) == "classification"

def test_benchmark_runs():
    df = pd.DataFrame({"x1": range(30), "x2": [i % 3 for i in range(30)], "target": [i % 2 for i in range(30)]})
    X, y, _ = prepare_xy(df, "target")
    results, fitted = benchmark_models(X, y, "classification", cv_folds=3)
    assert len(results) >= 2
    assert len(fitted) >= 2
    assert "F1_WEIGHTED" in results.columns
