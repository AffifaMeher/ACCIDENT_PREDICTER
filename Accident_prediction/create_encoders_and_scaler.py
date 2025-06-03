from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

# 1. Create and save LabelEncoder for Gender
gender_le = LabelEncoder()
gender_le.fit(['Male', 'Female'])  # Use all possible values in your dataset
joblib.dump(gender_le, 'gender_le.pkl')

# 2. Create and save LabelEncoder for Helmet_Used
helmet_le = LabelEncoder()
helmet_le.fit(['Yes', 'No'])
joblib.dump(helmet_le, 'helmet_le.pkl')

# 3. Create and save LabelEncoder for Seatbelt_Used
seatbelt_le = LabelEncoder()
seatbelt_le.fit(['Yes', 'No'])
joblib.dump(seatbelt_le, 'seatbelt_le.pkl')

# 4. Create and save a StandardScaler for numerical features
scaler = StandardScaler()
# Fit scaler on example numerical data for Age and Speed_of_Impact (use your real training data)
# Here just an example of fitting on dummy data:
example_numerical_data = np.array([
    [25, 30.0],
    [40, 50.0],
    [30, 20.0],
    [50, 70.0]
])
scaler.fit(example_numerical_data)
joblib.dump(scaler, 'scaler.pkl')

print("All encoder and scaler files created successfully!")
