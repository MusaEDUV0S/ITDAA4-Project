import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
# Function to load the machine learning model
def load_model():
    try:
        model = joblib.load('C:/Users/HP/Downloads/heart.sav')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_model()


# Function to make predictions
def predict_heart_disease(model, features):
    if model is None:
        return "Model not loaded"
    # Convert features into a DataFrame
    features_df = pd.DataFrame([features], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    prediction = model.predict(features_scaled)
    return prediction[0]

# App title
st.title("Heart Disease Prediction")


age = st.slider("Age", 1, 100, 50)
sex = st.selectbox("Sex (1 = Male, 2 = Female)", [0, 1])
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.slider("Resting Electrocardiographic Results (0 = Normal, 1 = Abnormal, 2 = Ventricular hypertrophy)", 0, 2, 3)
thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0= No)", [0, 1])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.slider("Slope of the Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)", 0, 2, 1)
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (0-4)", 0, 4, 0)
thal = st.slider("Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect, 0 = Unknown)", 0, 3, 2)

# code for Prediction
if st.button("Predict"):
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    result = predict_heart_disease(model, features)
    
    if result == "Model not loaded":
        st.error("Model could not be loaded. Please check the file path and try again.")
    elif result == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")