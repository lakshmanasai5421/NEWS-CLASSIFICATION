from flask import Flask, request, jsonify, render_template
import pickle
import os

# Define paths for model and vectorizer
MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

# Load trained model and vectorizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = [data["text"]]
    text_tfidf = vectorizer.transform(text)  # Convert text to numerical features
    prediction = model.predict(text_tfidf)[0]

    return jsonify({"prediction": "Fake" if prediction == 0 else "Real"})

if __name__ == "__main__":
    app.run(debug=True)
