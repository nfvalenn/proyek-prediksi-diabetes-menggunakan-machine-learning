import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Prediksi Risiko Diabetes", layout="centered")

st.title("🩺 Prediksi Risiko Diabetes")
st.markdown("""
Model ini menggunakan data medis dasar untuk memprediksi kemungkinan seseorang menderita **diabetes**.

Silakan isi data pasien di bawah ini untuk melakukan prediksi.
""")

# Cek keberadaan model
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
if not os.path.exists(model_path):
    st.error("❌ File model `diabetes_model.pkl` tidak ditemukan.\nPastikan file berada di direktori yang sama dengan script ini.")
    st.stop()

# Load model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Form input pengguna
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", 0, 20, step=1)
        glucose = st.number_input("Glukosa (mg/dL)", 0, 200, step=1)
        blood_pressure = st.number_input("Tekanan Darah (mm Hg)", 0, 140, step=1)
        skin_thickness = st.number_input("Ketebalan Lipatan Kulit (mm)", 0, 100, step=1)

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, step=1)
        bmi = st.number_input("BMI (kg/m²)", 0.0, 60.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
        age = st.number_input("Usia (tahun)", 1, 100, step=1)

    submitted = st.form_submit_button("🔍 Prediksi Risiko")

# Proses prediksi
if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probabilitas kelas 1 (diabetes)
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
        st.stop()

    st.markdown("### 🔎 Hasil Prediksi")
    if prediction == 1:
        st.error(f"🚨 Pasien **berisiko** diabetes.\n\nProbabilitas: **{probability:.2%}**")
    else:
        st.success(f"✅ Pasien **tidak berisiko** diabetes.\n\nProbabilitas: **{probability:.2%}**")
