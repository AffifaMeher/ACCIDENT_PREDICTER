from flask import Flask, render_template, request
import numpy as np
import joblib
from pymongo import MongoClient

app = Flask(__name__)

# Load your model and preprocessing objects (make sure these files exist)
model = joblib.load('model.pkl')
gender_le = joblib.load('gender_le.pkl')
helmet_le = joblib.load('helmet_le.pkl')
seatbelt_le = joblib.load('seatbelt_le.pkl')
scaler = joblib.load('scaler.pkl')

# MongoDB client setup (make sure MongoDB is running locally)
client = MongoClient('mongodb://localhost:27017/')
db = client['accident_prediction_db']
collection = db['accident_data']

def save_data_to_db(age, gender, speed_of_impact, helmet_used, seatbelt_used, prediction):
    record = {
        'age': int(age),
        'gender': gender,
        'speed_of_impact': float(speed_of_impact),
        'helmet_used': helmet_used,
        'seatbelt_used': seatbelt_used,
        'prediction': float(prediction)
    }
    collection.insert_one(record)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug print to check incoming form data
        print(request.form)

        age = int(request.form['age'])
        gender = request.form['gender']
        speed_of_impact = float(request.form['speed_of_impact'])
        helmet_used = request.form['helmet_used']
        seatbelt_used = request.form['seatbelt_used']

        # Encode categorical variables
        gender_enc = gender_le.transform([gender])[0]
        helmet_enc = helmet_le.transform([helmet_used])[0]
        seatbelt_enc = seatbelt_le.transform([seatbelt_used])[0]

        # Scale numerical features
        num_features = np.array([[age, speed_of_impact]])
        num_features_scaled = scaler.transform(num_features)

        # Combine features into a single input for prediction
        data = np.array([[num_features_scaled[0][0], gender_enc, num_features_scaled[0][1], helmet_enc, seatbelt_enc]])

        prediction = model.predict(data)[0]

        save_data_to_db(age, gender, speed_of_impact, helmet_used, seatbelt_used, prediction)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
