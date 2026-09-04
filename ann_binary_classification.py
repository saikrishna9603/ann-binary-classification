"""ANN binary classification for the car data dataset."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "car data.csv"
ARTIFACTS = ROOT / "ann_artifacts"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["Car_Name"])

    target_column = "Transmission"
    features = data.drop(columns=[target_column])
    target = (data[target_column] == "Automatic").astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    numeric_features = ["Year", "Present_Price", "Kms_Driven", "Owner"]
    categorical_features = ["Fuel_Type", "Seller_Type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "ann",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=42,
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    metrics = {
        "dataset": DATA_PATH.name,
        "rows": len(data),
        "features": list(features.columns),
        "positive_label": "Automatic transmission",
        "class_distribution": target.value_counts().sort_index().to_dict(),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "majority_baseline_accuracy": float(baseline_accuracy),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Manual", "Automatic"], zero_division=0
        ),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(features.columns)},
        ARTIFACTS / "ann_transmission_model.joblib",
    )
    (ARTIFACTS / "metrics.txt").write_text(
        "ANN binary classification\n"
        + "=" * 28
        + "\n"
        + "\n".join(f"{key}: {value}" for key, value in metrics.items()),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["standard", "high"], ax=axes[0], cmap="Blues"
    )
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    axes[1].plot(false_positive_rate, true_positive_rate, label=f"ANN (AUC={metrics['test_roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="majority reference")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "evaluation.png", dpi=160)
    plt.close(figure)

    print("Target: Automatic transmission")
    print(f"ANN accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Majority baseline: {metrics['majority_baseline_accuracy']:.3f}")
    print(f"ANN ROC-AUC: {metrics['test_roc_auc']:.3f}")
    print(f"Artifacts written to: {ARTIFACTS}")


if __name__ == "__main__":
    main()