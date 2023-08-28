import streamlit as st
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Load the model
with open('Naive_bayes.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess and predict downtime
def predict_downtime(data):
    imputer = SimpleImputer(strategy='mean')  # Use mean imputation, you can change this strategy as needed
    data_imputed = imputer.fit_transform(data)

    predictions = model.predict(data_imputed)  # Use your model to make predictions
    return predictions

# Streamlit app
def main():
    st.title(':red[Downtime Prediction App]')

    st.write("Enter the values for each feature:")

    # Get user inputs for each feature column
    features = [
        ":blue[Load_cells]", ":blue[Hydraulic_Pressure(bar)]", ":blue[Coolant_Pressure(bar)]",
        ":blue[Air_System_Pressure(bar)]", ":blue[Coolant_Temperature(°C)]",
        ":blue[Hydraulic_Oil_Temperature(°C)]", ":blue[Proximity_sensors]",
        ":blue[Spindle_Vibration(µm)]", ":blue[Tool_Vibration(µm)]",
        ":blue[Spindle_Speed(RPM)]", ":blue[Voltage(volts)]", ":blue[Torque]",
        ":blue[Cutting_Force(kN)]"
    ]

    user_inputs = []
    for feature in features:
        user_input = st.text_input(f":blue[Enter] {feature}:", value="")
        user_inputs.append(user_input)

    if st.button(':red[Predict Downtime]'):
        # Create a dataframe from user inputs
        input_data = pd.DataFrame([user_inputs], columns=features, dtype=np.float64)

        predictions = predict_downtime(input_data)  # Predict downtime using the model

        # Map 1 to "FAILURE" and 0 to "NON_FAILURE"
        predicted_downtime = 'FAILURE' if predictions[0] == 1 else 'NON_FAILURE'

        st.write(f"Predicted Downtime: {predicted_downtime}")

if __name__ == '__main__':
    main()
