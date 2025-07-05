# main.py

from config import CSV_PATH, TEXT_COL, LABEL_COL
import pandas as pd
import joblib

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
    print("\n Evaluation Results:")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1 Score        : {metrics['f1_score']:.4f}")
    print(f"AUC-ROC         : {metrics['auc_roc']:.4f}" if metrics['auc_roc'] is not None else "AUC-ROC         : Not available")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    # print(metrics)

    print("[INFO] Plotting ROC curve...")
    os.makedirs("saved_models", exist_ok=True)
    roc_path = "saved_models/roc_curve.png"
    plot_roc_curve(best_model, X_test, y_test, save_path=roc_path)
    input_text = "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!"
    input_example = pd.DataFrame({"text": [input_text]})

    print("[INFO] Logging to MLflow...")
    log_to_mlflow(best_model, best_params, metrics, roc_path, input_example)

    print("[INFO] Saving best model...")
    save_model(best_model, path="saved_models/best_model.pkl")

    # // Model Unpickling

    # Load the saved model
    model_path = "saved_models/best_model.pkl"
    loaded_model = joblib.load(model_path)

    # Define an example email
    text_example = "Congratulations! You've been selected to win a free trip to Dubai!"

    

    def predict_text_example(loaded_model, text_example):
    
    # Predicts and prints whether the input text is spam or not.
    
        print("\n🔍 Testing on example input:")
        print(f"Input Text: {text_example}")

        # Prepare input as expected by the model (usually DataFrame for pipelines)
        input_df = pd.DataFrame({"text": [text_example]})

        # Predict
        prediction = loaded_model.predict(input_df["text"])[0]
        proba = loaded_model.predict_proba(input_df["text"])[0][1] if hasattr(loaded_model, "predict_proba") else None

        # Show result
        label = "Spam" if prediction == 1 else "Ham"
        print(f"\nPrediction: {label}")
        if proba is not None:
            print(f" Spam Probability: {proba:.4f}")

    # Make prediction
    predict_text_example(loaded_model, text_example)



    # print("[INFO] Saving best model...")
    # save_model(best_model, path="saved_models/best_model.pkl")


if __name__ == "__main__":
    run_experiment()
