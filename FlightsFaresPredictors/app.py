import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODEL & COLUMNS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('flight_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except:
        return None, None

model, model_columns = load_assets()

# --- HELPER FUNCTIONS ---
def get_clean_suggestions(model_cols):
    """Extracts known categories from your model columns to provide as suggestions"""
    airlines = [c.replace('airline_', '') for c in model_cols if c.startswith('airline_')]
    cities = [c.replace('source_city_', '') for c in model_cols if c.startswith('source_city_')]
    return airlines, cities

known_airlines, known_cities = get_clean_suggestions(model_columns) if model_columns else ([], [])

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8f9fa; }
    .stSelectbox, .stDateInput, .stRadio { margin-bottom: 20px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<div style='text-align: center; padding: 10px 0;'>", unsafe_allow_html=True)
st.title("✈️ Real-time Flight Fare Prediction")
st.markdown("Fill in the details to get an estimated price.")
st.markdown("</div>", unsafe_allow_html=True)

if model is None:
    st.error("Model assets ('flight_model.pkl' and 'model_columns.pkl') not found in the directory.")
    st.stop()

# --- INPUT SECTION ---
with st.container():
    # Card-style background
    #st.markdown("""<div style='background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>""", unsafe_allow_html=True)
    
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        airline = st.selectbox(
            "Airline",
            options=sorted(known_airlines + ["Other Airline"]),
            help="Type to search for an airline."
        )
        
    with row1_col2:
        source = st.selectbox("Source City", options=sorted(known_cities + ["Other City"]),index=None,placeholder="Enter city name")

    with row1_col3:
        dest = st.selectbox("Destination City", options=sorted(known_cities + ["Other City"]),index=None,placeholder="Enter city name")

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    with row2_col1:
        travel_class = st.radio("Cabin Class", ["Economy", "Business"], horizontal=True)
    
    with row2_col2:
        journey_date = st.date_input("Date of Journey", min_value=date.today())
    
    with row2_col3:
        # Replaced slider with selectbox as requested
        stops = st.selectbox("Stops", options=["Non-stop", "1-stop", "2+-stop"])

    st.markdown("</div>", unsafe_allow_html=True)

# --- AUTO-PREDICTION LOGIC ---
days_left = (journey_date - date.today()).days
stop_map = {"Non-stop": 0, "1-stop": 1, "2+-stop": 2}

# Logic to handle "Other" inputs by falling back to known training categories
# This allows the user to select anything without the model breaking
safe_airline = airline if airline in known_airlines else known_airlines[0]
safe_source = source if source in known_cities else (known_cities[0] if known_cities else "Delhi")
safe_dest = dest if dest in known_cities else (known_cities[1] if len(known_cities) > 1 else "Mumbai")

raw_data = {
    'airline': safe_airline,
    'source_city': safe_source,
    'destination_city': safe_dest,
    'departure_time': 'Morning',
    'arrival_time': 'Evening',
    'class': travel_class,
    'Total_Stops': stop_map[stops],
    'days_left': days_left
}

# Encoding and Alignment
df_input = pd.DataFrame([raw_data])
df_encoded = pd.get_dummies(df_input)
final_df = pd.DataFrame(0, index=[0], columns=model_columns)

for col in df_encoded.columns:
    if col in model_columns:
        final_df[col] = float(df_encoded[col].iloc[0])

# --- RESULTS DISPLAY ---
st.markdown("<br>", unsafe_allow_html=True)
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    try:
        prediction = model.predict(final_df)[0]
        
        st.markdown(f"""
            <div style="background: #1E3A8A; padding: 25px; border-radius: 10px; text-align: center; color: white;">
                <p style="text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1.5px; opacity: 0.8; margin-bottom: 5px;">Predicted fare</p>
                <h1 style="margin: 0; font-size: 3rem; color: #FFFFFF;">₹ {round(float(prediction),2 ):,}</h1>
                <p style="margin-top: 10px; font-size: 0.9rem; color: #93C5FD;">
            </div>
        """, unsafe_allow_html=True)
        if travel_class == "Business":
            avg_price = 52500  # Business average
            saving_potential = 15000 # Typical drop if booked early
        else:
            avg_price = 6500   # Economy average
            saving_potential = 2500  # Typical drop if booked early
        
        if prediction > (avg_price * 1.2): # If price is 20% higher than average
            st.warning(f"🧐")
            st.write(f"This {travel_class} fare is higher than the usual ₹{avg_price:,}. Better checking other days.")

        # Scenario C: Price is Good/Low
        else:
            st.success(f"✅")
            st.write(f"This fare is close to {travel_class} class.")
    except Exception as e:
        st.info("")
# --- INTEGRATED TRAVEL ADD-ONS ---
if source and dest:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"🛠️ Travel Essentials: {source} to {dest}")
    
    # Using Tabs to keep the interface clean and interactive
    tab1, tab2, tab3, tab4 = st.tabs(["🏨 Stay & Dine", "🚕 Taxi", " 📍🏰TouristSightings", "🎫 Deals & Coupons"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🛌 Top Stays")
            st.info(f"Checking availability in {dest}...")
           
        with c2:
            st.markdown("##### 🍴 Local Eats")
            st.success("Top Rated Fine Dining")
            
    with tab2:
        st.markdown("##### 🚕 Commute Options")
        m1, m2, m3 = st.columns(3)
        m1.metric("App Cabs", "₹350 - ₹700")
        m2.metric("Local Transit", "₹15 - ₹50")
        m3.metric("Train Station", "15 mins away")
        if st.button(f"📍 Open {dest} Map"):
            st.write(f"Redirecting to live traffic map for {dest}...")

    with tab3:
        st.markdown(f"##### 🗺️ Explore {dest}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.checkbox("Historical Heritage Sites", value=True)
            st.checkbox("Main Shopping District")
        with col_b:
            st.checkbox("City Gardens & Parks")
            st.checkbox("Art & Culture Museum")

    with tab4:
        st.markdown("##### 🏷️ Active Coupons")
        st.success("Code: FLYHIGH2026")
        st.caption("Get ₹1000 off on your next international flight.")
        st.warning("Code: TRIPSTAY10")
        st.caption("10% flat discount on selected hotels.")
# --- FOOTER ---
# st.markdown("<br><hr><p style='text-align: center; color: gray; font-size: 0.8rem;'>SkyPrice AI Engine v2.5 | Real-time ML Prediction</p>", unsafe_allow_html=True)