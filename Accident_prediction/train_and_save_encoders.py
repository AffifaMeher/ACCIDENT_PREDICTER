import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (change path if needed)
df = pd.read_csv('accident.csv')

# Fill missing values
df['Gender'] = df['Gender'].fillna('Unknown')
df['Speed_of_Impact'] = df['Speed_of_Impact'].fillna(df['Speed_of_Impact'].mean())
df['Helmet_Used'] = df['Helmet_Used'].fillna('No')
df['Seatbelt_Used'] = df['Seatbelt_Used'].fillna('No')

# Encode categorical variables
gender_le = LabelEncoder()
helmet_le = LabelEncoder()
seatbelt_le = LabelEncoder()

df['Gender_enc'] = gender_le.fit_transform(df['Gender'])
df['Helmet_enc'] = helmet_le.fit_transform(df['Helmet_Used'])
df['Seatbelt_enc'] = seatbelt_le.fit_transform(df['Seatbelt_Used'])

# Save encoders
joblib.dump(gender_le, 'gender_le.pkl')
joblib.dump(helmet_le, 'helmet_le.pkl')
joblib.dump(seatbelt_le, 'seatbelt_le.pkl')

# Scale numerical features
scaler = StandardScaler()
df[['Age_scaled', 'Speed_scaled']] = scaler.fit_transform(df[['Age', 'Speed_of_Impact']])

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Prepare features and target
X = df[['Age_scaled', 'Gender_enc', 'Speed_scaled', 'Helmet_enc', 'Seatbelt_enc']]
y = df['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
acc = model.score(X_test, y_test)
print(f"Model Accuracy: {acc:.2f}")

# Save model
joblib.dump(model, 'model.pkl')
print("Model and preprocessors saved successfully.")
