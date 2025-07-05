from flask import Flask, request, jsonify
import pandas as pd
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")
@app.route('/best_model_hyperparameters', methods=['GET'])
def get_best_model_params():
        all_params = model.get_params()
        clf_params = {k.replace("clf__", ""): v for k, v in all_params.items() if k.startswith("clf__")}
        return jsonify(clf_params)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = model.predict([text])[0]
    label = "spam" if prediction == 1 else "ham"
    return jsonify({"prediction": label})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
