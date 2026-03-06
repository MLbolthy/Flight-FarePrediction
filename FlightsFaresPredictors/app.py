import streamlit as st
import pandas as pd
import joblib

# Load model and details
model = joblib.load('flight_model.pkl')
details = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Flight Price Predictor", layout="centered")
st.title("✈️ Smart Flight Fare Estimator")

# 1. & 2. Handle unseen cities & Autocomplete
# We provide a massive list of global cities. 
# If a user picks one NOT in 'details['cities']', we map it to 'Other' for the model.
global_cities = sorted(details['cities'] + ["New York", "London", "Tokyo", "Dubai", "Paris", "Singapore"])
global_airlines = sorted(details['airlines'] + ["Emirates", "Qatar Airways", "Lufthansa"])

# 3. Rename Source to "Board at"
col1, col2 = st.columns(2)

with col1:
    # selectbox in Streamlit allows typing to search/autocomplete
    source_input = st.selectbox("Board at (Source City)", options=global_cities)
    departure = st.selectbox("Departure Time", ['Morning', 'Afternoon', 'Evening', 'Night'])

with col2:
    dest_input = st.selectbox("Destination City", options=global_cities)
    arrival = st.selectbox("Arrival Time", ['Morning', 'Afternoon', 'Evening', 'Night'])

airline = st.selectbox("Airline", options=global_airlines)
flight_class = st.radio("Class", ['Economy', 'Business'], horizontal=True)
stops = st.select_slider("Stops", options=["Non-stop", "1-stop", "2+-stop"])

# Logic to handle "Unseen" data
def handle_unseen(value, trained_list):
    return value if value in trained_list else "Other"

# Prepare Input Data
input_dict = {
    "Total_Stops": {"Non-stop": 0, "1-stop": 1, "2+-stop": 2}[stops],
    "airline": handle_unseen(airline, details['airlines']),
    "source_city": handle_unseen(source_input, details['cities']),
    "destination_city": handle_unseen(dest_input, details['cities']),
    "departure_time": departure,
    "class": flight_class,
    "arrival_time": arrival
}

# 4. Automatic Prediction (No Button Needed)
# In Streamlit, any widget change triggers a script rerun automatically.
try:
    input_df = pd.get_dummies(pd.DataFrame([input_dict]))
    # Match the model's expected columns
    final_features = input_df.reindex(columns=details['columns'], fill_value=0)
    
    prediction = model.predict(final_features)[0]
    
    st.markdown("---")
    st.metric(label="Predicted Fare", value=f"₹{round(prediction, 2)}")
    st.info(f"Note: Predicting for route {source_input} to {dest_input} using {airline}.")

except Exception as e:
    st.error("Please ensure all fields are selected correctly.")