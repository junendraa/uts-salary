from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils.csv_processor import CSVAnalyzer

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'csv'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load salary prediction model
model = joblib.load('model/salary_model.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================
# ROUTE 1: Salary Predictor (UPDATED - TANPA AGE)
# ============================================

@app.route('/')
def home():
    """Salary prediction page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Salary prediction API - FIXED VERSION"""
    try:
        data = request.get_json()
        
        # Prepare features (TANPA AGE - sesuai model baru)
        features = np.array([[
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
            'required_fields': ['gender', 'education', 'years_exp']
        }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================================
# ROUTE 2: CSV Analysis (NEW)
# ============================================

@app.route('/analysis')
def analysis_page():
    """CSV analysis page"""
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze_csv():
    """Analyze uploaded CSV file"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze CSV
        analyzer = CSVAnalyzer(filepath)
        
        # Step 1: Load and validate
        success, message = analyzer.load_and_validate()
        if not success:
            os.remove(filepath)  # Clean up
            return jsonify({'error': message}), 400
        
        # Step 2: Prepare data
        success, message = analyzer.prepare_data()
        if not success:
            os.remove(filepath)  # Clean up
            return jsonify({'error': message}), 400
        
        # Step 3: Train model
        success, message = analyzer.train_model()
        if not success:
            os.remove(filepath)  # Clean up
            return jsonify({'error': message}), 400
        
        # Get results
        results = analyzer.get_summary()
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# ============================================
# Run Application
# ============================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
