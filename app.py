from flask import Flask, request, jsonify
import pandas as pd
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/best_model_parameter', methods=['GET'])
def get_best_model_params():
    model = joblib.load("saved_models/new_best_model.pkl")
    all_params = model.get_params()
    clf_params = {k.replace("logreg__", ""): v for k, v in all_params.items() if k.startswith("logreg__")}
    return jsonify(clf_params)


@app.route('/prediction', methods=['POST'])
def predict():
    model = joblib.load("saved_models/new_best_model.pkl")
    data = request.get_json()
    text = data['text']
    prediction = model.predict([text])[0]
    label = "spam" if prediction == 1 else "ham"
    return jsonify({"prediction": label})


@app.route('/training', methods=['POST'])
def train():
    dataset_path = 'Dataset/emails.csv'
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={'text': 'texts', 'spam': 'labels'}, inplace=True)

    X, y = df['texts'], df['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyperparams = request.get_json() or {}
    valid_params = ['C', 'max_iter', 'solver', 'penalty', 'random_state', 'tol']
    logreg_params = {param: hyperparams[param] for param in valid_params if param in hyperparams}

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(lowercase=True)),
        ('logreg', LogisticRegression(**logreg_params))
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(pipeline, "saved_models/new_best_model.pkl")

    return jsonify({
        "message": "Model retrained using Logistic Regression."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001)
