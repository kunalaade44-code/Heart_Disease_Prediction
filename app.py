import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Provide the following detail")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox('Sex',['M','F'])
chest_pain = st.selectbox('Chest Pain Type',['ATA','NAP','ASY','TA'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl',['0','1'])
resting_ecg = st.selectbox('Resting ECG',['Normal','ST','LVH'])
max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina',['Y','N'])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope',['Up','Flat','Down'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex'+ sex:1,
        'Chest Pain Type'+ chest_pain:1,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_blood_sugar,
        'RestingECG'+ resting_ecg:1,
        'Max Heart Rate ': max_hr,
        'Exercise Angina'+ exercise_angina:1,
        'Oldpeak': oldpeak,
        'ST_Slope'+ st_slope:1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease") 
    else:
        st.success("Low Risk of Heart Disease")