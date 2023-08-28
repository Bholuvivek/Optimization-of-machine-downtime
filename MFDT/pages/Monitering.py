import numpy as np
import pickle
import pandas as pd
import streamlit as st
import evidently

# Loading the saved model
loaded_model = pickle.load(open('Naive_bayes.pkl', 'rb'))

# Creating a function for Prediction
def downtime_prediction(input_data):
    # Convert the input_data dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
   
    # Remove the "Date" and "Machine_ID" columns
    columns_to_remove = ["Date", "Machine_ID"]
    input_df_imputed = input_df.drop(columns=columns_to_remove)

    input_df_imputed = input_df_imputed.fillna(0)

    # Changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_df_imputed.iloc[0])

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 'NON_FAILURE':
        return 'The machine is non failure'
    else:
        return 'The Machine is failure'
   

def main():
    # ...

    # Adding navigation to different pages
    pages = {
        "Main Page": main_page,
        "Model Drift Analysis": drift_analysis_page
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page
    pages[page]()

def main_page():
    st.title('Machine Downtime Prediction Web App')
    # getting the input data from the user
   
   
   
    Load_cells = st.number_input('Number of cells',format="%f")
    Hydraulic_Pressure = st.number_input('Give the value in bar for Hydraulic_Pressure',format="%f")
    Coolant_Pressure = st.number_input('Give the value in bar for Coolant_Pressure',format="%f")
    Air_System_Pressure = st.number_input('Give the value in bar for Air_System_Pressure',format="%f")
    Coolant_Temperature = st.number_input('Give the value in (°C) for Coolant_Temperature',format="%f")
    Hydraulic_Oil_Temperature = st.number_input('Give the value in (°C) for Hydraulic_Oil_Temperature',format="%f")
    Proximity_sensors = st.number_input('value for Proximity_sensors',format="%f")
    Spindle_Vibration = st.number_input('Give the value in (µm) for Spindle_Vibration',format="%f")
    Tool_Vibration = st.number_input('Give the value in (µm) for Tool_Vibration',format="%f")
    Spindle_Speed = st.number_input('Give the value in RPM for Spindle_Speed',format="%f")
    Voltage = st.number_input('Give the value in VOLTS',format="%f")
    Torque = st.number_input('Give the value in Torque',format="%f")
    Cutting_Force = st.number_input('Give the value in KN',format="%f")
   
   
    # code for Prediction
    state = ''
   
    # creating a button for Prediction
   
    if st.button('Downtime Test Result'):
        state = downtime_prediction([Load_cells, Hydraulic_Pressure, Coolant_Pressure, Air_System_Pressure, Coolant_Temperature, Hydraulic_Oil_Temperature, Proximity_sensors,
                                     Spindle_Vibration, Tool_Vibration, Spindle_Speed, Voltage, Torque, Cutting_Force])
       
       
    st.success(state)
   
   
    # Adding a file upload option
    st.write("Upload a CSV file with input data:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Create a new DataFrame for predictions
        predictions_df = pd.DataFrame(columns=['Prediction'])  # Create an empty DataFrame with only "Prediction" column

        # Perform predictions
        for index, row in df.iterrows():
            prediction = downtime_prediction(row)
            predictions_df.loc[index, 'Prediction'] = prediction  # Add prediction to the new DataFrame

        # Combine the original data and predictions
        result_df = pd.concat([predictions_df, df], axis=1)  # Combine the two DataFrames side by side

        # Display the results
        st.write("Predictions:")
        st.dataframe(result_df)

        # Download the results CSV
        st.write("Download Predictions as CSV:")
        csv = result_df.to_csv(index=False)
        st.download_button(label="Download Results", data=csv.encode('utf-8'), file_name="results.csv", mime="text/csv")


def drift_analysis_page():
    st.title('Model Drift Analysis')
   
    # Load historical data for drift analysis
    historical_data = pd.read_csv('G:/360digitmg/model deployemt/prima_13.csv')  # Replace with the actual path
    new_data = pd.read_csv('G:/360digitmg/model deployemt/test_file.csv')  # Replace with the actual path
   

    column_mapping = {
        'feature': [
            'Load_cells',
            'Hydraulic_Pressure(bar)',
            'Coolant_Pressure(bar)',
            'Air_System_Pressure(bar)',
            'Coolant_Temperature(°C)',
            'Hydraulic_Oil_Temperature(°C)',
            'Proximity_sensors',
            'Spindle_Vibration(µm)',
            'Tool_Vibration(µm)',
            'Spindle_Speed(RPM)',
            'Voltage(volts)',
            'Torque',
            'Cutting_Force(kN)'
        ],
        'target': 'Downtime'
    }  
   

    # Perform Evidently drift analysis
    drift_dashboard = evidently.DriftDashboard(
        reference_data=historical_data,
        current_data=new_data,
        column_mapping=column_mapping
    )

    # Generate the drift report
    report = drift_dashboard.report()

    # Display Evidently report
    st.write(report)

    # ... (you can add more components to this page if needed)

if __name__ == '__main__':
    main()
