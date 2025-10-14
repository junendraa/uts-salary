from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model/salary_model.pkl')

@app.route('/')
def home():
    # Serve HTML frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare features (sesuai urutan saat training)
        features = np.array([[
            data['age'],
            data['gender'],
            data['education'],
            data['years_exp']
        ]])
        
        # Predict
        prediction = model.predict(features)
        
        return jsonify({
            'salary_prediction': float(prediction[0]),
            'input': data
        })
    
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'required_fields': ['age', 'gender', 'education', 'years_exp']
        }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
