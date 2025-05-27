import streamlit as st
import pickle
import numpy as np

st.title("Prediksi Risiko Diabetes")
st.write("Model ini menggunakan data medis dasar untuk memprediksi kemungkinan diabetes.")

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Input user
glucose = st.number_input("Glukosa", 0, 200)
blood_pressure = st.number_input("Tekanan Darah", 0, 140)
bmi = st.number_input("BMI", 0.0, 60.0)
age = st.number_input("Usia", 1, 100)
insulin = st.number_input("Insulin", 0, 900)
pregnancies = st.number_input("Jumlah Kehamilan", 0, 20)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
skin_thickness = st.number_input("Skin Thickness", 0, 100)

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    st.success("Pasien Berisiko Diabetes" if prediction == 1 else "Pasien Tidak Berisiko Diabetes")
