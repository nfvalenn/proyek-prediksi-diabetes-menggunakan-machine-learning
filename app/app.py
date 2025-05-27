import streamlit as st
import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ----- CONFIG -----
st.set_page_config(
    page_title="🩺 Prediksi Risiko Diabetes",
    layout="centered",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

# ----- STYLE CSS -----
st.markdown(
    """
    <style>
    .big-font {
        font-size:24px !important;
        color: #4B8BBE;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-positive {
        color: red;
        font-weight: bold;
    }
    .result-negative {
        color: green;
        font-weight: bold;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True
)

# ----- HEADER -----
st.title("🩺 Prediksi Risiko Diabetes")
st.write("Gunakan data medis dasar berikut untuk memprediksi risiko diabetes.")

# ----- LOAD MODEL -----
model_path = "diabetes_model.pkl"
if not os.path.exists(model_path):
    st.error(f"File model `{model_path}` tidak ditemukan. Upload file `diabetes_model.pkl` ke direktori yang sama.")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ----- SIDEBAR -----
st.sidebar.header("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini memprediksi risiko diabetes berdasarkan data Pima Indians Diabetes Database.
Model machine learning menggunakan Random Forest (atau model yang kamu pilih).
""")
st.sidebar.markdown("### Data Sample (5 baris)")
# Untuk contoh, kita load dataset dari CSV (bisa ganti sesuai kebutuhan)
@st.cache_data
def load_data():
    # Kalau punya dataset lokal, bisa ganti path di bawah
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.sidebar.dataframe(df.head())

# ----- FORM INPUT -----
with st.form("form_prediksi"):
    st.markdown("<div class='big-font'>Masukkan Data Medis:</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", 0, 20, help="Masukkan jumlah kehamilan pasien.")
        glucose = st.number_input("Glukosa (mg/dL)", 0, 200, help="Kadar glukosa darah.")
        blood_pressure = st.number_input("Tekanan Darah (mm Hg)", 0, 140, help="Tekanan darah pasien.")
        skin_thickness = st.number_input("Ketebalan Kulit (mm)", 0, 100, help="Ketebalan kulit pasien.")
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", 0, 900, help="Kadar insulin pasien.")
        bmi = st.number_input("BMI", 0.0, 60.0, help="Body Mass Index pasien.")
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, help="Faktor riwayat diabetes keluarga.")
        age = st.number_input("Usia (tahun)", 1, 100, help="Usia pasien.")
    submitted = st.form_submit_button("🔍 Prediksi")

# ----- PREDIKSI & HASIL -----
if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Hasil Prediksi")
    if prediction == 1:
        st.markdown(f"<p class='result-positive'>🚨 Pasien <b>berisiko</b> diabetes! Probabilitas: <b>{probability:.2%}</b></p>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"<p class='result-negative'>✅ Pasien <b>tidak berisiko</b> diabetes. Probabilitas: <b>{probability:.2%}</b></p>", unsafe_allow_html=True)

    # Visualisasi probabilitas
    st.subheader("Visualisasi Probabilitas")
    prob_df = pd.DataFrame({
        'Kategori': ['Tidak Risiko', 'Risiko Diabetes'],
        'Probabilitas': [1 - probability, probability]
    })

    fig, ax = plt.subplots()
    sns.barplot(x='Kategori', y='Probabilitas', data=prob_df, palette=['green', 'red'], ax=ax)
    ax.set_ylim(0, 1)
    for i, v in enumerate(prob_df['Probabilitas']):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
    st.pyplot(fig)

# ----- FOOTER -----
st.markdown("""
---
Aplikasi dibuat oleh Kamu  
Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
""")
