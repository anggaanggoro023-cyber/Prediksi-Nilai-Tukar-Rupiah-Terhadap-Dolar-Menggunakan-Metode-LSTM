"""Prediksi Nilai Tukar Rupiah terhadap Dolar Menggunakan Metode LSTM.

Jalankan:  streamlit run app.py
"""
import streamlit as st

from kurs import tema
from kurs.ui import DashboardUI, DataUI, HasilPrediksiUI, HyperparameterUI, PelatihanUI

st.set_page_config(page_title="Prediksi Kurs USD/IDR (LSTM)", page_icon=":material/query_stats:", layout="wide")
tema.pasang()


def dashboard():
    DashboardUI().halamanDashboard()


def ambil_data():
    DataUI().halamanAmbilData()


def data_historis():
    DataUI().halamanDataHistoris()


def hyperparameter():
    HyperparameterUI().halamanPengaturan()


def latih_model():
    PelatihanUI().halamanPelatihan()


def hasil_prediksi():
    HasilPrediksiUI().halamanHasil()


halaman = st.navigation({
    "Utama": [st.Page(dashboard, title="Dashboard", icon=":material/dashboard:", default=True)],
    "Data": [st.Page(ambil_data, title="Ambil Data", icon=":material/cloud_download:", url_path="ambil-data"),
             st.Page(data_historis, title="Data Historis", icon=":material/table_chart:", url_path="data-historis")],
    "Model": [st.Page(hyperparameter, title="Hyperparameter", icon=":material/tune:", url_path="hyperparameter"),
              st.Page(latih_model, title="Latih Model", icon=":material/model_training:", url_path="latih-model")],
    "Hasil": [st.Page(hasil_prediksi, title="Hasil Prediksi", icon=":material/query_stats:", url_path="hasil-prediksi")],
})

with st.sidebar:
    st.markdown('<div class="side-foot"><b>Kurs LSTM v1.0.0</b><br>Prediksi Nilai Tukar Rupiah terhadap Dolar</div>',
                unsafe_allow_html=True)

halaman.run()
