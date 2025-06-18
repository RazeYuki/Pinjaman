import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
logreg = joblib.load("logreg_model.pkl")
rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set halaman
st.set_page_config(page_title="Prediksi Pinjaman Bank", layout="centered")
st.title("💰 Prediksi Persetujuan Pinjaman Bank")
st.markdown("Silakan isi formulir berikut untuk memprediksi apakah pinjaman akan disetujui.")

# Inisialisasi state
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# Tombol reset
if st.button("🔁 Coba Lagi"):
    st.session_state.show_result = False

# Form input pengguna
if not st.session_state.show_result:
    with st.form("form_input"):
        loan_amnt = st.number_input("💵 Jumlah Pinjaman", min_value=1000, step=100)
        loan_int_rate = st.number_input("📈 Suku Bunga Pinjaman (%)", min_value=0.0, step=0.1)
        person_income = st.number_input("👤 Pendapatan Tahunan", min_value=0, step=1000)

        home_ownership = st.selectbox("🏠 Status Kepemilikan Rumah", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        loan_intent = st.selectbox("🎯 Tujuan Pinjaman", [
            'EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'
        ])
        previous_default = st.selectbox("📉 Riwayat Gagal Bayar", ['Pernah Gagal Bayar', 'Tidak Pernah'])

        model_choice = st.selectbox("🧠 Pilih Model Prediksi", ['Logistic Regression', 'Random Forest', 'XGBoost'])

        submitted = st.form_submit_button("🔍 Prediksi")

        if submitted:
        try:
            # Proses input pengguna
            home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
            intent_map = {
                'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2,
                'PERSONAL': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5
            }
            default_map = {'Pernah Gagal Bayar': 1, 'Tidak Pernah': 0}

            input_data = pd.DataFrame([[
                loan_amnt,
                loan_int_rate,
                person_income,
                home_map[home_ownership],
                intent_map[loan_intent],
                default_map[previous_default]
            ]], columns=[
                'loan_amnt',
                'loan_int_rate',
                'person_income',
                'person_home_ownership',
                'loan_intent',
                'previous_loan_defaults_on_file'
            ])

            input_data = input_data[scaler.feature_names_in_]
            input_scaled = scaler.transform(input_data)

            # Prediksi
            if model_choice == 'Logistic Regression':
                prediction = logreg.predict(input_scaled)[0]
            elif model_choice == 'Random Forest':
                prediction = rf.predict(input_scaled)[0]
            else:
                prediction = xgb.predict(input_scaled)[0]

            st.session_state.show_result = True
            st.session_state.prediction = prediction

        except Exception as e:
            st.error(f"Ada kesalahan saat memproses input: {e}")

# Tampilkan hasil
if st.session_state.show_result:
    st.subheader("📊 Hasil Prediksi")
    if st.session_state.prediction == 1:
        st.success("✅ Pinjaman Anda kemungkinan **DISETUJUI**.")
    else:
        st.error("❌ Pinjaman Anda kemungkinan **TIDAK DISETUJUI**.")
