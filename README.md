# Accident Severity Prediction 🚑

This is a web application built using **Flask** that predicts the **severity of a road accident** based on user inputs like age, gender, speed of impact, helmet usage, and seatbelt usage.

---

## 💡 Features

- Input form to gather accident-related details.
- Machine Learning model to predict accident severity.
- Styled frontend using custom CSS.
- Flask backend handles input processing and prediction.
- Integrated preprocessing using Label Encoding.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS
- **Backend:** Flask (Python)
- **Machine Learning:** scikit-learn
- **Others:** Pandas, NumPy, joblib

---

## 📂 Project Structure

Accident_prediction/
│
├── app.py # Flask application
├── model.pkl # Trained ML model (saved using joblib)
├── accident.csv # Dataset used for training
├── static/
│ └── styles.css # Styling for the HTML page
├── templates/
│ └── index.html # Main HTML template
├── model.ipynb # Jupyter Notebook for training
└── README.md # Project documentation

## Install dependencies
   
pip install -r requirements.txt

pip install flask pandas numpy scikit-learn

## Run the application
   
python app.py

Navigate to http://127.0.0.1:5000 in your browser.

## 📊 Model Training

The model is trained in model.ipynb using:

Label Encoding for categorical features

Logistic Regression (or your chosen algorithm)

Dataset: accident.csv

Saved with: joblib.dump(model, 'model.pkl')

## ✅ Inputs Expected

Age: Integer (1-120)

Gender: Male/Female

Speed of Impact (km/h): Float

Helmet Used: Yes/No

Seatbelt Used: Yes/No


 ## To Do
 Add database to store predictions
 
 Add model accuracy feedback
 
 Host app on cloud (Heroku/Render)


