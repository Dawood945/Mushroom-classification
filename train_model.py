# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('mushroom_cleaned.csv')

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define feature columns (excluding 'season' and 'class')
feature_columns = [
    'cap-diameter', 'cap-shape',
    'gill-attachment', 'gill-color',
    'stem-height', 'stem-width', 'stem-color'
]

# Split data into features and target variable
X = df[feature_columns]
y = df['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoders
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)
