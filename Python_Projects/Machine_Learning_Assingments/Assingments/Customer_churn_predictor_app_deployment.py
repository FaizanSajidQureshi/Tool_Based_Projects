import streamlit as st
import pandas as pd
from joblib import load
import time

# Set the page configuration
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained Random Forest model
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    model = load('random_forest_model.joblib')
    return model

model = load_model()

# --- UI LAYOUT ---

# Main title of the app
st.title(" 📈 Customer Churn Prediction App")

# Add some introductory text using markdown
st.markdown("""
This app predicts whether a customer is likely to churn (leave the service) or stay.
Please enter the customer's details in the sidebar to get a prediction.
""")

# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("Enter Customer Information")

# Use a form to group inputs and a single button to trigger the prediction
with st.sidebar.form(key='prediction_form'):
    # A slider is often more intuitive for a range of numbers
    tenure = st.slider("Tenure (in months)", min_value=0, max_value=100, value=12, step=1)
    
    # Using radio buttons for fewer options can look cleaner
    internet_service = st.radio("Internet Service", ('DSL', 'Fiber optic', 'No'))
    
    contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))

    # Use columns to place related inputs side-by-side
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    with col2:
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0, step=1.0)
    
    # The submit button for the form
    submit_button = st.form_submit_button(label='Predict Churn')


# --- PREDICTION LOGIC AND DISPLAY ---
# Only run the prediction if the button has been pressed
if submit_button:
    # Map input values to numeric using the label mapping
    label_mapping = {
        'DSL': 0, 'Fiber optic': 1, 'No': 2,
        'Month-to-month': 0, 'One year': 1, 'Two year': 2,
    }
    
    # Convert categorical features to their numeric representation
    internet_service_encoded = label_mapping[internet_service]
    contract_encoded = label_mapping[contract]
    
    # Create the feature list for the model
    features = [[tenure, internet_service_encoded, contract_encoded, monthly_charges, total_charges]]

    # Add a progress bar for a better user experience
    with st.spinner('Analyzing data...'):
        time.sleep(2) # Simulate a delay for demonstration
        
        # Make a prediction using the model
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

    # NEW: Add a toast message to notify completion
    st.toast('Prediction Complete!', icon='🎉')

    st.header("Prediction Result")

    # Display the result using a more visual component like st.metric
    if prediction[0] == 0:
        st.success("This customer is likely to **STAY**.")
        # Add a fun element for a positive outcome
        st.balloons()
        
        # Display probability in a metric
        st.metric(label="Probability of Staying", value=f"{prediction_proba[0][0]:.2%}")
        
    else:
        st.error("This customer is likely to **CHURN**.")
        # NEW: Add a different fun element for a negative outcome
        st.snow()
        
        # Display probability in a metric
        st.metric(label="Probability of Churning", value=f"{prediction_proba[0][1]:.2%}")

    # Use an expander to show more details about the prediction
    with st.expander("Show more details"):
        st.write("Prediction based on the following input:")
        details = {
            "Tenure": f"{tenure} months",
            "Internet Service": internet_service,
            "Contract Type": contract,
            "Monthly Charges": f"${monthly_charges:.2f}",
            "Total Charges": f"${total_charges:.2f}"
        }
        st.json(details)
        st.write("The model used for this prediction is a Random Forest classifier.")

# Add a footer
st.markdown("---")
st.caption("App built with Streamlit for customer churn prediction.")

