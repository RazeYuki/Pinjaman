import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_terbaik.pkl")

st.set_page_config(
    page_title="Prediksi Persetujuan Pinjaman",
    layout="centered"
)

# ==============================
# HEADER
# ==============================
st.title("🏦 Prediksi Persetujuan Pinjaman")

st.markdown("""
Aplikasi ini membantu memprediksi **apakah pengajuan pinjaman berpotensi disetujui atau ditolak**
berdasarkan informasi pemohon.

📌 *Aplikasi ini merupakan **sistem pendukung keputusan**, bukan keputusan final dari bank.*
""")

st.markdown("---")

# ==============================
# DATA PEMOHON
# ==============================
st.subheader("📄 Data Pemohon")

loan_amnt = st.number_input(
    "Jumlah Pinjaman (USD)",
    min_value=500,
    max_value=50_000,
    value=10_000,
    step=500,
    help="Total dana pinjaman yang diajukan"
)

loan_int_rate = st.slider(
    "Suku Bunga Pinjaman (%)",
    min_value=0.0,
    max_value=40.0,
    value=13.5,
    help="Suku bunga tahunan pinjaman"
)

person_income = st.number_input(
    "Pendapatan Pemohon per Tahun (USD)",
    min_value=500,
    max_value=500_000,
    value=50_000,
    step=1_000,
    help="Pendapatan tahunan pemohon"
)

# ==============================
# INFORMASI TAMBAHAN
# ==============================
st.subheader("🏠 Informasi Tambahan")

person_home_ownership = st.selectbox(
    "Status Kepemilikan Rumah",
    ["MORTGAGE", "RENT", "OWN", "OTHER"],
    help="Status tempat tinggal pemohon"
)

loan_intent = st.selectbox(
    "Tujuan Pinjaman",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ],
    help="Tujuan penggunaan dana pinjaman"
)

previous_loan_defaults_on_file = st.selectbox(
    "Riwayat Gagal Bayar Sebelumnya",
    ["No", "Yes"],
    help="Apakah pemohon pernah gagal bayar sebelumnya"
)

st.markdown("---")

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Persetujuan"):

    # ==============================
    # DATAFRAME INPUT
    # ==============================
    input_data = pd.DataFrame({
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "previous_loan_defaults_on_file": [previous_loan_defaults_on_file]
    })

    # ==============================
    # LABEL ENCODING (SESUAI MODEL)
    # ==============================
    label_maps = {
        "person_home_ownership": {
            "OTHER": 0,
            "OWN": 1,
            "MORTGAGE": 2,
            "RENT": 3
        },
        "loan_intent": {
            "EDUCATION": 0,
            "MEDICAL": 1,
            "HOMEIMPROVEMENT": 2,
            "PERSONAL": 3,
            "DEBTCONSOLIDATION": 4,
            "VENTURE": 5
        },
        "previous_loan_defaults_on_file": {
            "No": 0,
            "Yes": 1
        }
    }

    for col, mapping in label_maps.items():
        input_data[col] = input_data[col].map(mapping)

    # ==============================
    # PREDIKSI
    # ==============================
    prediction = model.predict(input_data)[0]

    # ==============================
    # OUTPUT
    # ==============================
    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.success("✅ **Pinjaman Diprediksi DISETUJUI**")
        st.markdown("""
        Profil pemohon menunjukkan karakteristik yang **relatif baik**
        berdasarkan pola data historis.
        """)
    else:
        st.error("❌ **Pinjaman Diprediksi DITOLAK**")
        st.markdown("""
        Profil pemohon menunjukkan tingkat risiko yang **lebih tinggi**
        berdasarkan pola data historis.
        """)

    st.info("""
    ⚠️ **Catatan Penting:**  
    Hasil prediksi ini bersifat **pendukung keputusan** dan tidak
    merepresentasikan keputusan mutlak dari pihak bank.
    """)
