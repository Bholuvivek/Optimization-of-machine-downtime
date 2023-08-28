import streamlit as st
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer

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

    st.write(":blue[Upload a CSV file:]")
    uploaded_file = st.file_uploader(":red[Choose a file]", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset:")
        st.write(df)

        if st.button('Predict Downtime'):
            data_to_predict = df.drop(['Date', 'Machine_ID'], axis=1)

            predictions = predict_downtime(data_to_predict)  # Predict downtime using the model

            # Map 1 to "FAILURE" and 0 to "NON_FAILURE"
            predictions_mapped = ['FAILURE' if pred == 1 else 'NON_FAILURE' for pred in predictions]

            df['Predicted_Downtime'] = predictions_mapped

            # Move the Predicted_Downtime column to the beginning
            df = df[['Predicted_Downtime'] + [col for col in df.columns if col != 'Predicted_Downtime']]

            st.write("Predicted dataset:")
            st.write(df)

            # Download predicted dataset
            st.markdown('### Download Predicted Dataset')
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predicted Dataset",
                data=csv,
                file_name='predicted_dataset.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
