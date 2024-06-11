# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and label encoders
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Define the feature columns (excluding 'season' and 'class')
feature_columns = [
    'cap-diameter', 'cap-shape',
    'gill-attachment', 'gill-color',
    'stem-height', 'stem-width', 'stem-color'
]

# Streamlit app title
st.title("Mushroom Classification: Poisonous or Not")

# Streamlit form for user input
st.header("Enter Mushroom Characteristics:")
user_input = {}

# Capture user input
for column in feature_columns:
    unique_values = list(label_encoders[column].classes_)
    user_input[column] = st.selectbox(column, unique_values)

# Predict button
if st.button('Predict'):
    try:
        # Encode user input
        input_data = []
        for column in feature_columns:
            input_data.append(label_encoders[column].transform([user_input[column]])[0])
        input_data = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display the result
        if prediction == 1:
            st.error("The mushroom is Poisonous!")
        else:
            st.success("The mushroom is Edible!")
    except Exception as e:
        st.error(f"An error occurred: {e}. Please ensure all inputs are valid and try again.")
