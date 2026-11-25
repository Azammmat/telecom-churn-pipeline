import argparse
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from churn_data import FEATURE_COLUMNS, load_data


LOGGER = logging.getLogger(__name__)


def preprocess(table, config):
    test_size = config["test_fraction"]
    validation_size = config["validation_fraction"]
    holdout_size = test_size + validation_size
    if test_size <= 0 or validation_size <= 0 or holdout_size >= 1:
        raise ValueError("Test and validation fractions must be positive and sum to less than 1")
    seed = config["random_state"]
    features = table[list(FEATURE_COLUMNS)]
    target = table["churned"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=holdout_size, stratify=target, random_state=seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=test_size / holdout_size,
        stratify=y_holdout, random_state=seed,
    )
    splits = {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}
    for name, (_, labels) in splits.items():
        if labels.nunique() != 2:
            raise ValueError(f"The {name} split must contain both churn classes")
        LOGGER.info("%s split: %d subscribers (%d with churn)", name, len(labels), labels.sum())
    return splits


def train_model(name, parameters, X_train, y_train, seed):
    if name == "logistic_regression":
        model = make_pipeline(
            StandardScaler(), LogisticRegression(random_state=seed, **parameters)
        )
    elif name == "random_forest":
        model = RandomForestClassifier(random_state=seed, **parameters)
    else:
        raise ValueError(f"Unsupported model: {name}")
    LOGGER.info("Training %s", name)
    return model.fit(X_train, y_train)


def evaluate(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_artifacts(output, table, model, metrics, quality):
    output.parent.mkdir(parents=True, exist_ok=True)
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes, compress=3)
    members = {
        "features.parquet": table.to_parquet(index=False, engine="pyarrow"),
        "model.joblib": model_bytes.getvalue(),
        "metrics.json": json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        "data_quality.json": json.dumps(quality, indent=2, allow_nan=False) + "\n",
    }
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w") as archive:
            for filename, content in members.items():
                archive.writestr(ZipInfo(filename), content, compress_type=ZIP_DEFLATED)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run_pipeline(config_path, archive_override=None, output_override=None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    logging.basicConfig(**config["logging"])
    archive = archive_override or config_path.parent / config["data"]["archive"]
    output = output_override or config_path.parent / config["artifacts"]["output"]
    if output.resolve() in {archive.resolve(), config_path.resolve()}:
        raise ValueError("Output must differ from the input archive and configuration")
    if output.suffix.lower() != ".zip":
        raise ValueError("Output path must end in .zip")

    threshold = config["evaluation"]["threshold"]
    if not 0 < threshold < 1:
        raise ValueError("Evaluation threshold must be between 0 and 1")
    LOGGER.info("Loading CDR from %s (cutoff %s)", archive, config["data"]["cutoff"])
    table, quality = load_data(archive, config["data"])
    splits = preprocess(table, config["preprocessing"])
    X_train, y_train = splits["train"]
    seed = config["preprocessing"]["random_state"]
    candidates = {}
    validation = {}
    for name, parameters in config["models"].items():
        model = train_model(name, parameters, X_train, y_train, seed)
        candidates[name] = model
        validation[name] = evaluate(model, *splits["validation"], threshold)
        LOGGER.info("%s validation AP: %.4f", name, validation[name]["average_precision"])

    winner = max(validation, key=lambda name: validation[name]["average_precision"])
    final_model = candidates[winner]
    test = evaluate(final_model, *splits["test"], threshold)
    metrics = {
        "selected_model": winner,
        "selection_metric": "validation_average_precision",
        "threshold": threshold,
        "random_state": seed,
        "features": list(FEATURE_COLUMNS),
        "split_sizes": {name: len(labels) for name, (_, labels) in splits.items()},
        "validation": validation,
        "test": test,
    }
    LOGGER.info("Selected %s. Test AP: %.4f, ROC-AUC: %.4f", winner, test["average_precision"], test["roc_auc"])
    save_artifacts(output, table, final_model, metrics, quality)
    LOGGER.info("Published results to %s", output)


def main():
    parser = argparse.ArgumentParser(description="Train a churn model from the TDG archive")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--archive", type=Path, help="Override the input out.zip path")
    parser.add_argument("--output", type=Path, help="Override the result ZIP path")
    args = parser.parse_args()
    try:
        run_pipeline(args.config, args.archive, args.output)
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
