from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

model = joblib.load("model.pkl")

def clean_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/")
def home():
    return "Flask app running successfully"

@app.route("/predict", methods=["POST"])
def predict():
    tweet = request.form.get("tweet")

    if not tweet:
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_text(tweet)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0].max()

    label = "Disaster" if pred == 1 else "Not Disaster"

    return jsonify({
        "prediction": label,
        "confidence": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)
