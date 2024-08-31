from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

model = tf.keras.models.load_model("../models/model_v0.1.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
