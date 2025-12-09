# app.py
# Minimal Streamlit app wrapper for IPL win prediction.
# Replace the dummy model logic below with your real model loading/prediction code.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="IPL Win Predictor", layout="centered")

st.title("IPL Win Predictor")
st.write("Simple demo: enter some basic match inputs and click Predict.")

# --- Input form ---
with st.form("predict_form"):
    team1 = st.text_input("Team 1 (Home)", value="TeamA")
    team2 = st.text_input("Team 2 (Away)", value="TeamB")
    toss_winner = st.selectbox("Toss winner", options=[team1, team2])
    city = st.text_input("City", value="Mumbai")
    overs = st.slider("Overs completed", 0, 20, 10)
    submit = st.form_submit_button("Predict")

# --- Prediction logic (placeholder) ---
def load_model():
    # Try to load a saved model 'model.joblib' from repo root (if you have it).
    # If not present, we use a dummy fallback.
    model_path = "model.joblib"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"Failed loading model.joblib: {e}")
            return None
    return None

model = load_model()

def make_features(team1, team2, toss_winner, city, overs):
    # Convert inputs to a simple numeric feature vector as example placeholder.
    # Replace this with the exact features your real model expects.
    return np.array([[len(team1), len(team2), 1 if toss_winner==team1 else 0, len(city), overs]])

if submit:
    st.write("Running prediction...")
    X = make_features(team1, team2, toss_winner, city, overs)
    if model is not None:
        try:
            pred_proba = model.predict_proba(X)
            pred_label = model.predict(X)
            st.success(f"Predicted winner: **{pred_label[0]}**")
            st.write(f"Probability: {pred_proba[0].round(3)}")
        except Exception as e:
            st.error("Model present but prediction failed. (Maybe feature mismatch.)")
            st.write(repr(e))
    else:
        # Dummy fallback: predict team with longer name (just as demo)
        demo_winner = team1 if len(team1) >= len(team2) else team2
        st.info("No saved model found. Showing demo prediction.")
        st.write(f"Predicted winner (demo): **{demo_winner}**")

st.markdown("---")
st.write("If you have a trained model file named `model.joblib` in the repo root, the app will try to use it.")
