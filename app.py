import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load('/workspaces/DAPR-Model/dataset/model/gbm_model.pkl')

# Function to make predictions
def predict_maintenance(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Main function to define Streamlit interface
def main():
    st.title('Predictive Maintenance Interface')

    # Sidebar with input parameters
    st.sidebar.header('Input Parameters')
    engine_rpm = st.sidebar.slider('Engine RPM', min_value=0, max_value=5000, value=2500, step=100)
    lub_oil_pressure = st.sidebar.slider('Lub Oil Pressure', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    fuel_pressure = st.sidebar.slider('Fuel Pressure', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    coolant_pressure = st.sidebar.slider('Coolant Pressure', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    lub_oil_temp = st.sidebar.slider('Lub Oil Temperature', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    coolant_temp = st.sidebar.slider('Coolant Temperature', min_value=0.0, max_value=100.0, value=50.0, step=1.0)

    # Make prediction
    prediction = predict_maintenance(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp)

    # Display prediction
    if prediction == 1:
        st.write('Predicted Engine Condition: Good')
    else:
        st.write('Predicted Engine Condition: Needs Maintenance')

if __name__ == '__main__':
    main()