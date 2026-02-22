# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("heart.csv")

st.title("❤️ Heart Disease Prediction System")

# Dataset preview
st.subheader("📌 Dataset Preview")
st.write(df.head())

# Sidebar input
st.sidebar.header("🔍 Enter Patient Details")

def user_input():
    age = st.sidebar.number_input("Age", 20, 100, 40)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar (1=True, 0=False)", [1, 0])
    restecg = st.sidebar.selectbox("Rest ECG (0-2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
    oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("CA (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thal (1-3)", [1, 2, 3])

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame([data])

user_data = user_input()

st.subheader("📌 Entered Patient Data")
st.write(user_data)

# Prediction
scaled_data = scaler.transform(user_data)
prediction = model.predict(scaled_data)[0]

if prediction == 1:
    result = "❤️ Heart Disease Present"
else:
    result = "🍀 No Heart Disease"

st.header("🔮 Prediction Result")
st.success(result)

# ----------------------------------------
# FIXED: Feature Importance Graph (Always Random Forest)
# ----------------------------------------

st.subheader("🔥 Major Causes of Heart Disease")

from sklearn.ensemble import RandomForestClassifier

# Train a fresh Random Forest just for feature importance
X = df.drop("target", axis=1)
y = df["target"]

rf = RandomForestClassifier()
rf.fit(X, y)

importances = rf.feature_importances_

fig, ax = plt.subplots()
ax.barh(X.columns, importances)
ax.set_xlabel("Importance Score")
ax.set_title("Major Causes of Heart Disease")
st.pyplot(fig)
