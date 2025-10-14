import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.extend([BASE_DIR, PARENT_DIR])

from flask import Flask, request, jsonify, render_template
import pandas as pd
from linear_regression import LinearRegression as CustomLinearRegression
from main import run_training_pipeline

app = Flask(__name__)

# Inisialisasi model
model = CustomLinearRegression()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        # Jalankan pipeline yang sudah kamu buat
        run_training_pipeline()
        return jsonify({"message": "✅ Model berhasil dilatih dari pipeline main.py"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
