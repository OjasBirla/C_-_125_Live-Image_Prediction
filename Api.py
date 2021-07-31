from flask import Flask, jsonify, request
from Model import getPrediction

App = Flask(__name__)

@App.route("/predict_digits", methods=["POST"])

def predict_Data():
    image = request.files.get("Digits.png")
    prediction = getPrediction(image)

    return jsonify({
        "Prediction": prediction
    }), 200

if __name__ == "__main__":
    App.run()
