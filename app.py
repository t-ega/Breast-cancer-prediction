from flask import Flask, request, jsonify, render_template
import joblib
import json
import pandas as pd
import sqlite3
from datetime import datetime
from create_sqlite_db import create_db
import os

app = Flask(__name__)

# Load the model and top features
model = joblib.load('random_forest_top.pkl')

with open('top_features.json', 'r') as f:
    top_features = json.load(f)

label_mapping = {
    0: 'Benign',
    1: 'Malignant' 
}

# Function to save predictions to the database
def save_prediction(name, features, prediction):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    # Insert data into the table
    c.execute('''
        INSERT INTO predictions (name, features, prediction)
        VALUES (?, ?, ?)
    ''', (name, json.dumps(features), prediction))
    
    conn.commit()
    conn.close()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure all top features are present
        missing = [f for f in top_features if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing feature(s): {", ".join(missing)}'
            }), 400
        
        name = data.get('patient_name')  # Get the name field

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        # Convert input data into DataFrame
        input_df = pd.DataFrame([data])
        input_df = input_df[top_features]

        # Predict
        prediction = model.predict(input_df)[0]
        result = label_mapping.get(prediction, "Unknown")

        # Save prediction to database
        save_prediction(name, data, result)

        return jsonify({
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    create_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
