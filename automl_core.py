from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import io
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

@dataclass
class DatasetProfile:
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    high_cardinality_columns: list[str]
    constant_columns: list[str]


def load_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(raw))
    raise ValueError("Unsupported file type. Use CSV, XLSX/XLS, or JSON.")


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    high_card = [c for c in categorical if df[c].nunique(dropna=True) > min(100, max(20, int(len(df) * 0.2)))]
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return DatasetProfile(
        rows=len(df), columns=len(df.columns),
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric, categorical_columns=categorical,
        high_cardinality_columns=high_card, constant_columns=constant,
    )


def infer_task(y: pd.Series) -> str:
    clean = y.dropna()
    if not pd.api.types.is_numeric_dtype(clean):
        return "classification"
    # Numeric targets with a small number of distinct values are commonly labels.
    unique = clean.nunique()
    if unique <= 20 and (pd.api.types.is_integer_dtype(clean) or unique / max(len(clean), 1) < 0.05):
        return "classification"
    return "regression"


def prepare_xy(df: pd.DataFrame, target: str):
    work = df.dropna(subset=[target]).copy()
    X = work.drop(columns=[target])
    y = work[target]
    # Drop constant columns and obvious ID-like columns only when requested upstream.
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)
    return X, y, constant


def make_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def candidate_models(task: str, random_state: int = 42):
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1500),
            "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=250, random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        }
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }


def benchmark_models(X: pd.DataFrame, y: pd.Series, task: str, cv_folds: int = 5, random_state: int = 42):
    models = candidate_models(task, random_state)
    pre = make_preprocessor(X)
    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted", "precision_weighted": "precision_weighted", "recall_weighted": "recall_weighted"}
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        try:
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            row = {"Model": name}
            for metric in scoring:
                val = float(np.mean(scores[f"test_{metric}"]))
                if metric.startswith("mae") or metric.startswith("rmse"):
                    val = -val
                row[metric.upper()] = val
            row["CV Std"] = float(np.std(scores[list(scores.keys())[0]]))
            rows.append(row)
            fitted[name] = pipe
        except Exception as exc:
            rows.append({"Model": name, "Error": str(exc)})
    result = pd.DataFrame(rows)
    if task == "classification" and "F1_WEIGHTED" in result:
        result = result.sort_values("F1_WEIGHTED", ascending=False, na_position="last").reset_index(drop=True)
    elif task == "regression" and "R2" in result:
        result = result.sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    return result, fitted


def fit_final_model(X, y, task, model_name, random_state=42):
    models = candidate_models(task, random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    pipe = Pipeline([("preprocessor", make_preprocessor(X)), ("model", models[model_name])])
    pipe.fit(X, y)
    return pipe


def evaluate_holdout(pipe, X, y, task, test_size=0.2, random_state=42):
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    pipe2 = clone(pipe)
    pipe2.fit(Xtr, ytr)
    pred = pipe2.predict(Xte)
    if task == "classification":
        return {
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, average="weighted", zero_division=0),
            "Recall": recall_score(yte, pred, average="weighted", zero_division=0),
            "F1": f1_score(yte, pred, average="weighted", zero_division=0),
        }
    return {
        "R²": r2_score(yte, pred),
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": mean_squared_error(yte, pred) ** 0.5,
    }


def feature_importance_dataframe(pipe: Pipeline) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        vals = np.mean(np.abs(coef), axis=0) if np.ndim(coef) > 1 else np.abs(coef)
    else:
        return None
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(len(vals))])
    if len(names) != len(vals):
        return None
    return pd.DataFrame({"Feature": names, "Importance": vals}).sort_values("Importance", ascending=False).reset_index(drop=True)


def serialize_model(pipe) -> bytes:
    return pickle.dumps(pipe)
