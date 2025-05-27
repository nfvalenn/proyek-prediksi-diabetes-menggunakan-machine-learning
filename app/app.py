import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Prediksi Risiko Diabetes", layout="centered")

st.title("🩺 Prediksi Risiko Diabetes")
st.write("Model ini menggunakan data medis dasar untuk memprediksi kemungkinan seseorang menderita diabetes.")

# Cek keberadaan model
model_path = "diabetes_model.pkl"
if not os.path.exists(model_path):
    st.error(f"File model `{model_path}` tidak ditemukan. Pastikan file berada di direktori yang sama dengan script ini.")
    st.stop()

# Load model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Form input pengguna
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", 0, 20)
        glucose = st.number_input("Glukosa", 0, 200)
        blood_pressure = st.number_input("Tekanan Darah", 0, 140)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 60.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
        age = st.number_input("Usia", 1, 100)

    submitted = st.form_submit_button("🔍 Prediksi")

# Proses prediksi
if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probabilitas kelas 1 (diabetes)

    st.markdown("### Hasil Prediksi")
    if prediction == 1:
        st.error(f"🚨 Pasien **berisiko** diabetes. Probabilitas: **{probability:.2%}**")
    else:
        st.success(f"✅ Pasien **tidak berisiko** diabetes. Probabilitas: **{probability:.2%}**")
