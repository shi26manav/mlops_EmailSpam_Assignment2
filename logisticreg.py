# logisticreg.py

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
from mlflow.models import infer_signature

import mlflow.sklearn
import os


def perform_random_search(X_train, y_train):
    """
    Performs randomized hyperparameter tuning for Logistic Regression.
    Returns best model and best parameters.
    """
    # count_vect=CountVectorizer()

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('clf', LogisticRegression(solver="liblinear", max_iter=1000))
    ])

    param_dist = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__max_df': [0.5, 0.75, 1.0],
        'clf__C': np.logspace(-3, 2, 6),
        'clf__penalty': ['l1', 'l2']
    }

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    save_path = "saved_models/roc_curve.png"

    # save_model(best_model, path="saved_models/best_model.pkl")

    search.fit(X_train, y_train)
       # Save the best CountVectorizer 
    best_vectorizer = search.best_estimator_.named_steps['vectorizer']
    with open("saved_models/count_vectorizer.pkl", "wb") as vec_file:
        joblib.dump(best_vectorizer, vec_file)

    return search.best_estimator_, search.best_params_

def evaluate_classification(model, X_test, y_test):
    """
    Evaluates a binary classification model and returns key metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, X_test, y_test, save_path):
    """Plot ROC curve and return AUC score."""
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    # Save plot
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] ROC curve saved to {save_path}")
    return roc_auc

def save_model(model, path):
    """
    Saves model using joblib.
    """
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


  




# def log_to_mlflow(model, params, metrics, roc_curve_path, input_example):
#     """
#     Logs model, metrics, and ROC curve to MLflow.
#     """
#     mlflow.set_experiment("SpamClassifier_LogisticRegression_Tuning")
#     with mlflow.start_run():
#         mlflow.log_params(params)
#         mlflow.log_metrics({
#             "accuracy": metrics["accuracy"],
#             "precision": metrics["precision"],
#             "recall": metrics["recall"],
#             "f1_score": metrics["f1_score"],
#             "auc_roc": metrics["auc_roc"] if metrics["auc_roc"] else 0
#         })
#         if roc_curve_path:
#             if roc_curve_path and os.path.exists(roc_curve_path):
#                 mlflow.log_artifact(roc_curve_path)
#             else:
#                 print(f"[WARNING] ROC curve file not found: {roc_curve_path}")
#             # mlflow.log_artifact(roc_curve_path)
#          # Prepare input example for signature
#         input_example = pd.DataFrame({"text": [sample_input_text]})
#         prediction_example = model.predict(input_example["text"])
#         signature = infer_signature(input_example, prediction_example)

#          # Log the model with input example and signature
#         mlflow.sklearn.log_model(
#             sk_model=model,
#             artifact_path="logistic_regression_model",
#             input_example=input_example,
#             signature=signature
#         )
#         mlflow.sklearn.log_model(model, "logistic_regression_model")

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import os

def log_to_mlflow(model, params, metrics, roc_curve_path, input_example):
    """
    Logs model, metrics, and ROC curve to MLflow, using the provided input example.
    """
    mlflow.set_experiment("SpamClassifier_LogisticRegression_Tuning")

    with mlflow.start_run():
        # Log model hyperparameters
        mlflow.log_params(params)

        # Log evaluation metrics safely
        mlflow.log_metrics({
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
            "auc_roc": metrics.get("auc_roc", 0)
        })

        # Log ROC curve if it exists
        if roc_curve_path and os.path.exists(roc_curve_path):
            mlflow.log_artifact(roc_curve_path)
        else:
            print(f"[WARNING] ROC curve file not found: {roc_curve_path}")

        # Use passed input_example to infer signature
        try:
            prediction_example = model.predict(input_example["text"])
            signature = infer_signature(input_example, prediction_example)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="logistic_regression_model",
                input_example=input_example,
                signature=signature
            )
            print("[MLflow] Model logged with input example and signature.")
        except Exception as e:
            print(f"[WARNING] Could not infer signature or log model with input example: {e}")
            mlflow.sklearn.log_model(model, "logistic_regression_model")
