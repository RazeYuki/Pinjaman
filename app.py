import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model terbaik dan scaler
model = joblib.load('model_terbaik.pkl')

# Judul
st.title("💳 Prediksi Persetujuan Pinjaman Bank")
st.markdown("Masukkan informasi di bawah ini untuk memprediksi apakah pinjaman akan disetujui atau tidak.")

# Form Input
with st.form("loan_form"):
    loan_amnt = st.number_input("Jumlah Pinjaman (USD)", min_value=500, max_value=50000, value=10000, step=500)
    loan_int_rate = st.slider("Suku Bunga Pinjaman (%)", 0.0, 40.0, 13.5)
    person_income = st.number_input("Pendapatan Pemohon (USD)", min_value=500, max_value=500000, value=50000, step=1000)
    person_home_ownership = st.selectbox("Kepemilikan Rumah", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    previous_loan_defaults_on_file = st.selectbox("Pernah Gagal Bayar Sebelumnya?", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Prediksi")

# Mapping Input ke Bentuk Data Model
if submitted:
    # Konversi input ke bentuk DataFrame
    input_data = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })

    # Label Encoding manual sesuai model
    label_maps = {
        'person_home_ownership': {"MORTGAGE": 2, "RENT": 3, "OWN": 1, "OTHER": 0},
        'loan_intent': {
            "EDUCATION": 0, "MEDICAL": 1, "VENTURE": 5,
            "PERSONAL": 3, "HOMEIMPROVEMENT": 2, "DEBTCONSOLIDATION": 4
        },
        'previous_loan_defaults_on_file': {"Yes": 1, "No": 0}
    }

    for col, mapping in label_maps.items():
        input_data[col] = input_data[col].map(mapping)

    # Prediksi
    prediction = model.predict(input_data)[0]

    # Output
    st.subheader("📌 Hasil Prediksi:")
    if prediction == 1:
        st.success("🎉 Pinjaman kemungkinan besar akan disetujui.")
    else:
        st.error("❌ Pinjaman kemungkinan besar akan ditolak.")

    # Tombol reset
    if st.button("🔁 Coba Lagi"):
        st.experimental_rerun()
