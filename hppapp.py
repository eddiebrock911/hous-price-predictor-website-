import streamlit as st
import pickle
import numpy as np

# streamlit run hppapp.py

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏡 House Price Predictor")

st.write("Enter house details below to predict price:")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=2000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
    house_age = st.number_input("House Age (years)", min_value=0, max_value=200, value=10)

with col2:
    location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
    garage = st.selectbox("Garage", ["No", "Yes"])
    condition = st.selectbox("Condition", ["Poor", "Fair", "Good", "Excellent"])

# One-hot encoding for location
location_rural = 1 if location == "Rural" else 0
location_suburban = 1 if location == "Suburban" else 0
location_urban = 1 if location == "Urban" else 0

# Encode garage
garage_encoder = 1 if garage == "Yes" else 0

# Encode condition
condition_mapping = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
condition_encoded = condition_mapping[condition]

# Arrange input features
features = np.array([[area, bedrooms, bathrooms, floors, house_age,
                      location_rural, location_suburban,
                      location_urban, garage_encoder, condition_encoded]])

# Scale features
features_scaled = scaler.transform(features)

# Predict button
if st.button("💰 Predict Price"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted House Price: ₹ {prediction:,.0f}")
