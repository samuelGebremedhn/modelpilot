# ModelPilot 🧭

**Automated Machine Learning Toolkit for Local Data Science**

ModelPilot turns a tabular dataset into a repeatable machine-learning workflow: profile the data, choose a target, infer the problem type, benchmark multiple models with cross-validation, train a final model, inspect feature importance, make predictions, and export the trained pipeline.

## Features

- CSV, Excel, and JSON ingestion
- Automatic dataset profiling
- Missing-value and duplicate-row inspection
- Automatic classification/regression task inference
- User override of inferred task
- Numeric imputation + standardization
- Categorical imputation + one-hot encoding
- Four candidate models for each task
- Cross-validation benchmarking
- Holdout evaluation
- Feature importance / coefficient inspection
- Prediction on a second dataset
- Downloadable trained scikit-learn pipeline
- Downloadable model metadata
- Fully local execution; no API key required

## Models

### Classification
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

### Regression
- Ridge Regression
- Random Forest
- Extra Trees
- Gradient Boosting

## Architecture

```text
Dataset
   │
   ▼
Data Loader
   │
   ▼
Profiler ──────────────► Quality Report
   │
   ▼
Target Selection
   │
   ▼
Task Inference
   │
   ▼
Preprocessing Pipeline
   │
   ▼
Cross-Validation Benchmark
   │
   ▼
Best Candidate
   │
   ▼
Final Training
   ├────────► Metrics
   ├────────► Feature Importance
   ├────────► New Predictions
   └────────► Exported Pipeline
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How to use

1. Upload a tabular dataset.
2. Inspect the automatic data-quality profile.
3. Select the target column.
4. Confirm or override classification/regression.
5. Choose the number of cross-validation folds.
6. Run ModelPilot.
7. Compare candidate models.
8. Train the final model.
9. Review holdout metrics and feature importance.
10. Upload new data to generate predictions.
11. Export the trained pipeline and metadata.

## Example datasets

The application works well with common educational datasets such as Iris, Titanic, California Housing, or your own business dataset. ModelPilot does not bundle third-party datasets, keeping the repository lightweight.

## Design principles

**Local first:** data stays on the user's machine during the application workflow.

**Transparent:** the app exposes candidate models, validation metrics, preprocessing, and feature importance instead of hiding the entire modeling process behind a single score.

**Reproducible:** a configurable random seed and cross-validation are used throughout the workflow.

**Practical:** the exported `.pkl` file contains the preprocessing and model pipeline together, reducing train/inference preprocessing mismatches.

## Limitations

ModelPilot is an educational and practical AutoML toolkit, not a replacement for expert model validation. It does not automatically solve every problem involving leakage, temporal data, grouped observations, causal inference, extreme class imbalance, or domain-specific validation. Users should inspect the dataset and choose an appropriate evaluation strategy for consequential applications.

## Tests

Install pytest if needed:

```bash
pip install pytest
```

Run:

```bash
pytest -q
```

## License

MIT License.
