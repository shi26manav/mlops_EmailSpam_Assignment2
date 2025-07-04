from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
with open("best_mnnaivebayesmodel.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)
BEST_ALPHA = model.alpha if hasattr(model, 'alpha') else None

@app.route('/best_model_hyperparameters', methods=['GET'])
def get_best_model_param():
    return jsonify({"best_alpha": BEST_ALPHA})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    label = "spam" if prediction == 1 else "ham"
    return jsonify({"prediction": label})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
