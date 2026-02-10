from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'payments.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
else:
    print("Warning: Model file not found. Please train the model first.")


@app.route('/')
def home():
    """Serve the home page"""
    return render_template('home.html')


@app.route('/predict', methods=['GET'])
def predict_page():
    """Serve the prediction page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        transaction_amount = float(request.form.get('transaction_amount', 0))
        transaction_type = request.form.get('transaction_type', '')
        account_balance = float(request.form.get('account_balance', 0))
        location = request.form.get('location', '')
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Prepare input data for prediction
        # Note: You may need to adjust this based on your actual model's input features
        # This is a simplified version - you might need label encoding for categorical variables
        features = np.array([[transaction_amount, account_balance]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Interpret the result
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        
        return jsonify({
            'prediction': result,
            'transaction_amount': transaction_amount,
            'transaction_type': transaction_type,
            'account_balance': account_balance,
            'location': location
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
