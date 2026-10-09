import json
import numpy as np
import pandas as pd
import streamlit as st
from automl_core import load_dataframe, profile_dataframe, infer_task, prepare_xy, benchmark_models, fit_final_model, evaluate_holdout, feature_importance_dataframe, serialize_model

st.set_page_config(page_title="ModelPilot", page_icon="🧭", layout="wide")

st.title("🧭 ModelPilot")
st.caption("Automated machine-learning workflow for local dataset profiling, model benchmarking, evaluation, and prediction.")

with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "json"])
    st.divider()
    st.header("2. Training")
    folds = st.slider("Cross-validation folds", 3, 10, 5)
    test_size = st.slider("Holdout test size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random seed", 0, 9999, 42)

if not uploaded:
    st.info("Upload a CSV, Excel, or JSON dataset to begin.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profile", "Automatic")
    c2.metric("Model search", "4 models")
    c3.metric("Validation", "Cross-validation")
    c4.metric("Deployment", "Local")
    st.markdown("### Workflow")
    st.markdown("**Upload → Profile → Choose target → Infer task → Benchmark → Evaluate → Explain → Predict → Export**")
    st.stop()

try:
    df = load_dataframe(uploaded)
except Exception as exc:
    st.error(str(exc)); st.stop()

if df.empty:
    st.error("The dataset is empty."); st.stop()

profile = profile_dataframe(df)

st.subheader("Dataset overview")
cols = st.columns(5)
cols[0].metric("Rows", f"{profile.rows:,}")
cols[1].metric("Columns", profile.columns)
cols[2].metric("Missing cells", f"{profile.missing_cells:,}")
cols[3].metric("Duplicate rows", f"{profile.duplicate_rows:,}")
cols[4].metric("Numeric features", len(profile.numeric_columns))

with st.expander("Data preview", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Automatic data-quality profile"):
    quality = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True)
    if profile.constant_columns: st.warning("Constant columns: " + ", ".join(profile.constant_columns))
    if profile.high_cardinality_columns: st.warning("High-cardinality categorical columns may reduce model quality: " + ", ".join(profile.high_cardinality_columns))

st.subheader("Model setup")
target = st.selectbox("Select the target column", df.columns)
X, y, dropped = prepare_xy(df, target)
if len(X.columns) == 0:
    st.error("No usable feature columns remain after removing constant columns."); st.stop()

inferred = infer_task(y)
task = st.radio("Problem type", ["classification", "regression"], index=0 if inferred == "classification" else 1, horizontal=True)
st.caption(f"ModelPilot inference: **{inferred}**. You can override it when the automatic guess is not appropriate.")

if task == "classification" and y.nunique() < 2:
    st.error("Classification requires at least two target classes."); st.stop()
if task == "classification" and y.value_counts().min() < folds:
    st.warning(f"The least frequent class has fewer than {folds} samples. Reduce the CV folds or add more data.")

if st.button("🚀 Run ModelPilot", type="primary", use_container_width=True):
    with st.spinner("Benchmarking candidate models..."):
        results, fitted = benchmark_models(X, y, task, folds, int(random_state))
    st.session_state.results = results
    st.session_state.fitted = fitted
    st.session_state.task = task
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.target = target
    st.session_state.test_size = test_size
    st.session_state.random_state = int(random_state)

if "results" not in st.session_state:
    st.stop()

results = st.session_state.results
st.subheader("Benchmark results")
st.dataframe(results, use_container_width=True, hide_index=True)

valid = results[results["Model"].isin(st.session_state.fitted.keys())]
if valid.empty:
    st.error("No candidate model completed successfully. Inspect the Error column above."); st.stop()

best_default = valid.iloc[0]["Model"]
selected_model = st.selectbox("Model to train for final use", valid["Model"].tolist(), index=valid["Model"].tolist().index(best_default))

if st.button("Train final model", use_container_width=True):
    with st.spinner("Training final model on the complete usable dataset..."):
        final_model = fit_final_model(st.session_state.X, st.session_state.y, st.session_state.task, selected_model, st.session_state.random_state)
        holdout = evaluate_holdout(final_model, st.session_state.X, st.session_state.y, st.session_state.task, st.session_state.test_size, st.session_state.random_state)
    st.session_state.final_model = final_model
    st.session_state.final_model_name = selected_model
    st.session_state.holdout = holdout

if "final_model" not in st.session_state:
    st.stop()

st.success(f"Final model ready: {st.session_state.final_model_name}")
metric_cols = st.columns(len(st.session_state.holdout))
for col, (name, value) in zip(metric_cols, st.session_state.holdout.items()):
    col.metric(name, f"{value:.4f}")

imp = feature_importance_dataframe(st.session_state.final_model)
if imp is not None:
    st.subheader("Feature importance")
    top = imp.head(20).sort_values("Importance")
    st.bar_chart(top.set_index("Feature"), horizontal=True)
    with st.expander("Full feature importance table"):
        st.dataframe(imp, use_container_width=True, hide_index=True)
else:
    st.info("This model does not expose native feature importance coefficients.")

st.subheader("Predict on new data")
pred_file = st.file_uploader("Upload a new dataset with the same feature columns", type=["csv", "xlsx", "xls", "json"], key="prediction")
if pred_file:
    try:
        pred_df = load_dataframe(pred_file)
        expected = list(st.session_state.X.columns)
        missing = [c for c in expected if c not in pred_df.columns]
        if missing:
            st.error("Missing feature columns: " + ", ".join(missing))
        else:
            pred_input = pred_df[expected]
            preds = st.session_state.final_model.predict(pred_input)
            output = pred_df.copy()
            output["ModelPilot_Prediction"] = preds
            st.dataframe(output, use_container_width=True)
            st.download_button("Download predictions CSV", output.to_csv(index=False), "modelpilot_predictions.csv", "text/csv")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.subheader("Export")
model_bytes = serialize_model(st.session_state.final_model)
metadata = {
    "project": "ModelPilot",
    "target": st.session_state.target,
    "task": st.session_state.task,
    "model": st.session_state.final_model_name,
    "holdout_metrics": st.session_state.holdout,
    "features": list(st.session_state.X.columns),
}
st.download_button("⬇️ Download trained model (.pkl)", model_bytes, "modelpilot_model.pkl", "application/octet-stream")
st.download_button("⬇️ Download model metadata (.json)", json.dumps(metadata, indent=2), "modelpilot_metadata.json", "application/json")
