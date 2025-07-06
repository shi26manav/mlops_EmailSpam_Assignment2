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
print("Model type:", type(model))
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



# Load the model and vectorizer using Logistic Regression and Count Vectorizer 
# and store new_best_model and new_count_vectorizer.pkl file in saved_models folder
# curl -X POST http://127.0.0.1:5001/train


@app.route('/train', methods=['POST'])

def train():
    # Path to dataset
    dataset_path = 'Dataset/emails.csv'

    # Check file exists
    if not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset not found at {dataset_path}"}), 404

    try:
        # Read CSV
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip().str.lower()

        # Rename to expected names
        df.rename(columns={'text': 'texts', 'spam': 'labels'}, inplace=True)

        # Validate required columns
        if not {'texts', 'labels'}.issubset(df.columns):
            return jsonify({
                "error": "CSV must contain 'texts' and 'labels' columns",
                "found": df.columns.tolist()
            }), 400

    except FileNotFoundError:
        return jsonify({"error": f"File not found at {dataset_path}"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read or process CSV: {str(e)}"}), 400

    # Extracting features and labels
    X, y = df['texts'], df['labels']

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # New pipeline 
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(lowercase=True)),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    # Fitting model
    pipeline.fit(X_train, y_train)

    # Saving pipeline and vectorizer
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(pipeline, "saved_models/new_best_model.pkl")
    joblib.dump(pipeline.named_steps['vectorizer'], "saved_models/new_count_vectorizer.pkl")

    return jsonify({"output": "Model retrained with Logistic Regression and CountVectorizer."})



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
