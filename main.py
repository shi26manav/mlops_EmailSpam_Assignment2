# main.py

from config import CSV_PATH, TEXT_COL, LABEL_COL
import pandas as pd
from sklearn.model_selection import train_test_split
from logisticreg import (
    perform_random_search,
    evaluate_classification,
    plot_roc_curve,
    save_model,
    log_to_mlflow
)
import os

def run_experiment():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(CSV_PATH, usecols=[TEXT_COL, LABEL_COL])
    df['label'] = df[LABEL_COL]

    X = df[TEXT_COL]
    y = df['label']

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

    print("[INFO] Performing Randomized Search...")
    best_model, best_params = perform_random_search(X_train, y_train)

    print("[INFO] Evaluating model...")
    metrics = evaluate_classification(best_model, X_test, y_test)

    print("[INFO] Plotting ROC curve...")
    os.makedirs("saved_models", exist_ok=True)
    roc_path = "saved_models/roc_curve.png"
    plot_roc_curve(best_model, X_test, y_test, save_path=roc_path)

    print("[INFO] Logging to MLflow...")
    log_to_mlflow(best_model, best_params, metrics, roc_path)

    print("[INFO] Saving best model...")
    save_model(best_model, path="saved_models/best_model.pkl")

if __name__ == "__main__":
    run_experiment()
