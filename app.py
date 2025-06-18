import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
logreg = joblib.load("logreg_model.pkl")
rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set halaman
st.set_page_config(page_title="Prediksi Persetujuan Pinjaman", layout="centered")
st.title("🏦 Prediksi Persetujuan Pinjaman Bank")
st.markdown("Isi form berikut untuk memprediksi apakah pinjaman akan disetujui atau tidak.")

# Form input user
with st.form("form_input"):
    loan_amnt = st.number_input("Jumlah Pinjaman (`loan_amnt`)", min_value=1000, step=100)
    loan_int_rate = st.number_input("Suku Bunga Pinjaman (%) (`loan_int_rate`)", min_value=0.0, step=0.1)
    person_income = st.number_input("Pendapatan Pemohon per Tahun (`person_income`)", min_value=0, step=1000)

    home_ownership = st.selectbox("Status Kepemilikan Rumah (`person_home_ownership`)", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_intent = st.selectbox("Tujuan Pinjaman (`loan_intent`)", [
        'EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'
    ])
    previous_default = st.selectbox("Apakah Ada Pinjaman Default Sebelumnya? (`previous_loan_defaults_on_file`)", ['Yes', 'No'])

    model_choice = st.selectbox("Pilih Model", ['Logistic Regression', 'Random Forest', 'XGBoost'])

    submitted = st.form_submit_button("Prediksi")

# Prediksi jika tombol ditekan
if submitted:
    # Encode data kategorikal
    home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    intent_map = {
        'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2,
        'PERSONAL': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5
    }
    default_map = {'Yes': 1, 'No': 0}

    # Susun data input sesuai urutan fitur
    input_data = pd.DataFrame([[
        loan_int_rate,
        loan_amnt,
        person_income,
        home_map[home_ownership],
        intent_map[loan_intent],
        default_map[previous_default]
    ]], columns=[
        'loan_int_rate',
        'loan_amnt',
        'person_income',
        'person_home_ownership',
        'loan_intent',
        'previous_loan_defaults_on_file'
    ])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    if model_choice == 'Logistic Regression':
        prediction = logreg.predict(input_scaled)[0]
    elif model_choice == 'Random Forest':
        prediction = rf.predict(input_scaled)[0]
    else:
        prediction = xgb.predict(input_scaled)[0]

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.success("✅ Pinjaman Disetujui")
    else:
        st.error("❌ Pinjaman Tidak Disetujui")
