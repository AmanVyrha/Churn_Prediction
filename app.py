from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model, scaler, and numerical column names
model = joblib.load('churn_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
numerical_cols = joblib.load('numerical_cols.pkl')

# Load the label encoders for each categorical feature
le_gender = joblib.load('label_encoder_gender.pkl')
le_PaperlessBilling = joblib.load('label_encoder_PaperlessBilling.pkl')
le_InternetService = joblib.load('label_encoder_InternetService.pkl')
le_Contract = joblib.load('label_encoder_Contract.pkl')

# Serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # --- Preprocessing ---
        # 1. Label Encoding for categorical features
        df['gender'] = le_gender.transform(df['gender'])
        df['PaperlessBilling'] = le_PaperlessBilling.transform(df['PaperlessBilling'])
        df['InternetService'] = le_InternetService.transform(df['InternetService'])
        df['Contract'] = le_Contract.transform(df['Contract'])

        # 2. Scale numerical features
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # --- Make prediction ---
        prediction = model.predict(df)[0]
        churn_probability = model.predict_proba(df)[0][1]

        # Decode prediction (using "Yes" and "No" directly)
        prediction_label = "Yes" if prediction == 1 else "No"

        # Return the prediction as JSON
        return jsonify({'prediction': prediction_label, 'churn_probability': float(churn_probability)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
