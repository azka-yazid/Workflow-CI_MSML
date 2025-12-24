import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn


def main():
    data_path = os.path.join(os.path.dirname(__file__), "healthcare-dataset-stroke-data_preprocessing.csv")
    df = pd.read_csv(data_path)

    if "stroke" not in df.columns:
        raise ValueError("Kolom target 'stroke' tidak ditemukan di dataset.")

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # autolog
    mlflow.sklearn.autolog(log_models=True)
    

    mlflow.set_tag("mlflow.runName", "logreg-ci")

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metrics({
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    })

    print("CI run done.")


if __name__ == "__main__":
    main()
