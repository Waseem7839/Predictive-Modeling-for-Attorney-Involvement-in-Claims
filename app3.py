import pickle
import streamlit as st
import pandas as pd

def load_model():
    # Load the pre-trained model
    model = pickle.load(open('xgb_model.pkl', 'rb'))
    return model


def preprocess_input(data):
    # Add missing features with default values (replace with actual logic if needed)
    data['CLMINSUR'] = 0  # Default value, update based on actual data
    data['Claim_Settlement_Total'] = 0  # Default value, update if needed

    # Selecting required columns for the model
    model_columns = ['CLMSEX', 'CLMINSUR', 'CLMAGE', 'LOSS', 
                     'Accident_Severity', 'Claim_Approval_Status', 
                     'Driving_Record', 'Claim_Settlement_Total']
    data = data[model_columns]

    # Converting all features to integers
    data = data.astype(int)
    return data

def main():
    st.title("Attorney Prediction App")
    
    # Collect user input
    clmsex = st.selectbox("Claimant Sex (0: Female, 1: Male)", [0, 1])
    clmage = st.number_input("Claimant Age", min_value=0, max_value=100, step=1)
    loss = st.number_input("Loss Amount", min_value=0, step=100)
    accident_severity = st.selectbox("Accident Severity (1-3)", [1, 2, 3])
    claim_approval_status = st.selectbox("Claim Approval Status (0: No, 1: Yes)", [0, 1])
    driving_record = st.selectbox("Driving Record (0: Clean, 1: Minor Offenses, 2: Major Offenses)", [0, 1, 2])
    
    # Additional inputs for missing features
    clminsurance = st.selectbox("Claim Insurance (0: No, 1: Yes)", [0, 1])  # For CLMINSUR
    claim_settlement_total = st.number_input("Claim Settlement Total", min_value=0, step=100)  # For Claim_Settlement_Total

    # Create input dataframe
    input_data = pd.DataFrame([[clmsex, clminsurance, clmage, loss, accident_severity, 
                                claim_approval_status, driving_record, claim_settlement_total]],
                              columns=['CLMSEX', 'CLMINSUR', 'CLMAGE', 'LOSS', 
                                       'Accident_Severity', 'Claim_Approval_Status', 
                                       'Driving_Record', 'Claim_Settlement_Total'])

    # Load model and preprocess input
    model = load_model()
    processed_data = preprocess_input(input_data)

    # Make prediction when button is clicked
    if st.button("Predict Attorney Hiring"):
        prediction = model.predict(processed_data)
        result = "Client is likely to hire an attorney." if prediction[0] == 1 else "Client is unlikely to hire an attorney."
        st.write(result)

# Corrected the entry point check
if __name__ == "__main__":
    main()
