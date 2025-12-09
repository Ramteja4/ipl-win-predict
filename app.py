# app.py
# Streamlit app that loads win_prob_model.pkl and performs prediction.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(page_title="IPL Win Predictor", layout="centered")

st.title("IPL Win Predictor")
st.write("This app loads `win_prob_model.pkl` and predicts win probability based on your inputs.")

MODEL_PATH = "win_prob_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, "Model file `win_prob_model.pkl` not found in repo root."
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

model, load_error = load_model()

if load_error:
    st.error(load_error)
    st.info("Please upload `win_prob_model.pkl` to the root of your GitHub repository and redeploy.")
    st.stop()

st.subheader("Enter Match Details")

with st.form("prediction_form"):
    batting_team = st.text_input("Batting Team", "CSK")
    bowling_team = st.text_input("Bowling Team", "MI")
    city = st.text_input("City", "Mumbai")

    runs_left = st.number_input("Runs Left", min_value=0, value=50)
    balls_left = st.number_input("Balls Left", min_value=0, value=30)
    wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, value=5)
    total_runs_x = st.number_input("Target Runs", min_value=0, value=180)
    crr = st.number_input("Current Run Rate (CRR)", min_value=0.0, value=8.5)
    rrr = st.number_input("Required Run Rate (RRR)", min_value=0.0, value=9.0)

    submit = st.form_submit_button("Predict Win Probability")

if submit:
    try:
        # Create input DataFrame exactly as model expects
        input_data = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [city],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets_left": [wickets_left],
            "total_runs_x": [total_runs_x],
            "crr": [crr],
            "rrr": [rrr],
        })

        st.write("Input Data:")
        st.dataframe(input_data)

        # Try predict_proba first
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]  # probability of win
            st.success(f"Win Probability: **{proba * 100:.2f}%**")
        else:
            pred = model.predict(input_data)[0]
            st.success(f"Predicted Outcome: **{pred}**")

    except Exception as e:
        st.error("Prediction failed!")
        st.code(traceback.format_exc())
