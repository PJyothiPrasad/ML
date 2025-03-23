# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained logistic regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Optionally, load or instantiate the scaler if it was used during training
# For example, if you saved the scaler, you can load it similarly:
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    # Render the HTML template (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the request
    try:
        data = request.get_json()
        age = float(data['age'])
        Sex_male = float(data['Sex_male'])
        cigsPerDay = float(data['cigsPerDay'])
        totChol = float(data['totChol'])
        sysBP = float(data['sysBP'])
        glucose = float(data['glucose'])
    except Exception as e:
        return jsonify({'error': 'Invalid input format', 'message': str(e)}), 400

    # Create a numpy array for the features
    features = np.array([[age, Sex_male, cigsPerDay, totChol, sysBP, glucose]])
    
    # Scale the input features using the same scaler used in training
    features_scaled = scaler.transform(features)
    
    # Predict the probability (or the class)
    prediction = model.predict(features_scaled)[0]
    
    # Return the prediction result as JSON
    return jsonify({'TenYearCHD': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

