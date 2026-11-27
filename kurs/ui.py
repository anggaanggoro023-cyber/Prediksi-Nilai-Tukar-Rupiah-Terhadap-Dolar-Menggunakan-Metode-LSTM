"""Boundary: DataUI, HyperparameterUI, PelatihanUI, HasilPrediksiUI (+ DashboardUI).

Setiap class = satu halaman Streamlit. Nama method mengikuti Class Diagram;
method ``halaman...`` menyusun urutan pemanggilannya seperti di Sequence Diagram.
DashboardUI hanya ringkasan (tidak ada di diagram, tidak mengubah data).
"""
import datetime as dt

import pandas as pd
import streamlit as st

from . import grafik, tema
from .controller import RENTANG_VALID, DataController, ModelController, PrediksiController
from .entity import HYPERPARAMETER_DEFAULT, DataKurs, HasilPrediksi
from .sumber import UnggahFile, YahooFinance

NAMA_HP = {
    "windowSize": "Ukuran window (hari)",
    "rasioData": "Porsi data latih",
    "jumlahUnit": "Neuron LSTM lapisan 1",
    "dropout": "Dropout",
    "epoch": "Epoch",
    "batchSize": "Batch size",
    "learningRate": "Learning rate",
}


def _baris_hp(nilai: dict) -> list:
    """Mengubah dict hyperparameter menjadi baris tabel [nama tampil, nilai]."""
    return [[NAMA_HP.get(k, k), str(v)] for k, v in nilai.items()]


def _perubahan(selisih: float):
    """Lencana 'Naik', 'Turun', atau 'Tetap' menurut tanda selisih."""
    if abs(selisih) < 0.005:
        return tema.pill("Tetap", "abu")
    if selisih > 0:
        return tema.pill("Naik " + tema.angka(selisih, 2), "hijau")
    return tema.pill("Turun " + tema.angka(abs(selisih), 2), "merah")


def _kartu(kunci: str):
    """Wadah Streamlit bergaya kartu dengan kunci CSS unik."""
    return st.container(key=f"kartu_{kunci}")


# ---------------------------------------------------------------------------
class DataUI:
    def __init__(self):
        """Menyiapkan DataUI dengan DataController."""
        self.controller = DataController()

    # Sequence 1 - Mengambil data kurs
    def bukaMenuAmbilData(self):
        """Menampilkan judul halaman Ambil Data."""
        tema.topbar("Data", "Ambil Data")

    def tampilkanPilihanPeriode(self):
        """Menampilkan form sumber dan periode. Mengembalikan (periode, sumber, tombolDiklik)."""
        with _kartu("sumber"):
            tema.judul_kartu("Sumber dan periode data",
                             "Ambil kurs USD/IDR sebagai data latih, dari Yahoo Finance atau dari berkas kurs "
                             "(misalnya unduhan JISDOR dari Bank Indonesia).")
            pilihan = st.radio("Sumber data", ["Yahoo Finance", "Unggah berkas (CSV / Excel)"], horizontal=True)
            hari_ini = dt.date.today()
            kolomKiri, kolomKanan = st.columns(2)
            mulai = kolomKiri.date_input("Tanggal mulai", value=hari_ini - dt.timedelta(days=365))
            akhir = kolomKanan.date_input("Tanggal akhir", value=hari_ini)
            sumber = None
            if pilihan == "Yahoo Finance":
                sumber = YahooFinance(st.text_input("Ticker Yahoo Finance", value="IDR=X"))
            else:
                berkas = st.file_uploader("Berkas kurs", type=["csv", "xlsx", "xls"])
                if berkas is not None:
                    tabelBerkas = UnggahFile.baca(berkas)
                    kolom = list(tabelBerkas.columns)
                    kolomTanggalTebakan, kolomNilaiTebakan = UnggahFile.tebakKolom(tabelBerkas)
                    kolomA, kolomB = st.columns(2)
                    sumber = UnggahFile(berkas,
                                        kolomA.selectbox("Kolom tanggal", kolom, index=kolom.index(kolomTanggalTebakan)),
                                        kolomB.selectbox("Kolom nilai kurs", kolom, index=kolom.index(kolomNilaiTebakan)))
            self.pilihPeriode((mulai, akhir))
            diklik = self.klikAmbilData()
        return (mulai, akhir), sumber, diklik

    def pilihPeriode(self, periode):
        """Menyimpan periode pilihan peneliti."""
        st.session_state["periode"] = periode

    def klikAmbilData(self) -> bool:
        """Menampilkan tombol 'Ambil Data'. True bila diklik."""
        return st.button("Ambil Data", type="primary")

    def tampilkanRingkasan(self, ringkasan: dict):
        """Menampilkan ringkasan data yang berhasil diambil."""
        st.success("Data kurs berhasil diambil dan disimpan.")
        tema.baris_kartu([
            tema.kartu("Jumlah baris", tema.angka(ringkasan["jumlahBaris"]), "data kurs tersimpan", "database", "biru"),
            tema.kartu("Tanggal awal", str(ringkasan["tanggalAwal"]), "data paling lama", "kalender", "hijau"),
            tema.kartu("Tanggal akhir", str(ringkasan["tanggalAkhir"]), "data paling baru", "kalender", "teal"),
            tema.kartu("Rentang kurs", tema.angka(ringkasan["kursMinimum"]) + " - " + tema.angka(ringkasan["kursMaksimum"]),
                       "rupiah per dolar", "tren", "oranye", True),
        ])

    def tampilkanPesanError(self, pesan: str):
        """Menampilkan pesan kesalahan pengambilan data."""
        st.error(f"{pesan}\n\nSilakan coba ambil data kembali.")

    def halamanAmbilData(self):
        """Menyusun halaman Ambil Data mengikuti urutan pesan Sequence 1."""
        self.bukaMenuAmbilData()
        periode, sumber, diklik = self.tampilkanPilihanPeriode()
        if not diklik:
            return
        if sumber is None:
            return self.tampilkanPesanError("Pilih berkas kurs terlebih dahulu.")
        if periode[0] >= periode[1]:
            return self.tampilkanPesanError("Tanggal mulai harus sebelum tanggal akhir.")
        with st.spinner("Mengambil data kurs..."):
            hasil = self.controller.ambilData(periode, sumber)
        if hasil["status"] == "berhasil":
            self.tampilkanRingkasan(hasil["ringkasan"])
        else:
            self.tampilkanPesanError(hasil["pesan"])

    # Sequence 2 - Melihat data historis
    def pilihMenuDataHistoris(self):
        """Menampilkan judul halaman Data Historis."""
        tema.topbar("Data", "Data Historis")

    def tampilkanTabelDanGrafik(self, data: pd.DataFrame):
        """Menampilkan kartu ringkasan, grafik, dan tabel data historis."""
        data = data.sort_values("tanggal").reset_index(drop=True)
        akhir = data.iloc[-1]
        tema.baris_kartu([
            tema.kartu("Jumlah data", tema.angka(len(data)), "hari kerja", "database", "biru"),
            tema.kartu("Kurs terakhir", tema.angka(akhir["nilaiKurs"], 2), str(akhir["tanggal"].date()), "aktivitas", "hijau"),
            tema.kartu("Tertinggi", tema.angka(data["nilaiKurs"].max(), 2), "dalam periode", "tren", "oranye"),
            tema.kartu("Terendah", tema.angka(data["nilaiKurs"].min(), 2), "dalam periode", "tren", "teal"),
        ])
        with _kartu("grafik_historis"):
            tema.judul_kartu("Pergerakan kurs USD/IDR", "Kurs penutupan harian")
            st.pyplot(grafik.historis(data))
        with _kartu("tabel_historis"):
            tema.judul_kartu("Tabel data historis", "Data terbaru di atas")
            selisih = data["nilaiKurs"].diff().fillna(0.0)
            baris = [[str(t.date()), tema.angka(v, 2), _perubahan(d)]
                     for t, v, d in zip(data["tanggal"], data["nilaiKurs"], selisih)][::-1]
            tema.tabel_berhalaman("historis", ["Tanggal", "Kurs (Rp)", "Perubahan"], baris, rata_kanan=(1,))

    def tampilkanPesanDataKosong(self):
        """Menampilkan peringatan bahwa data belum tersedia."""
        st.warning("Data belum tersedia. Ambil data terlebih dahulu di menu Ambil Data.")

    def halamanDataHistoris(self):
        """Menyusun halaman Data Historis mengikuti urutan pesan Sequence 2."""
        self.pilihMenuDataHistoris()
        data = self.controller.periksaDataTersedia()
        if data is None:
            self.tampilkanPesanDataKosong()
        else:
            self.tampilkanTabelDanGrafik(data)


# ---------------------------------------------------------------------------
class HyperparameterUI:
    def __init__(self):
        """Menyiapkan HyperparameterUI dengan ModelController."""
        self.controller = ModelController()

    # Sequence 3 - Mengatur hyperparameter
    def bukaPengaturan(self):
        """Menampilkan judul halaman Hyperparameter."""
        tema.topbar("Model", "Hyperparameter")

    def tampilkanNilaiDefault(self, kunci: str = "hyperparameterSaatIni"):
        """Menampilkan form nilai saat ini. Mengembalikan (nilai, tombolSimpanDiklik)."""
        hyperparameterSaatIni = self.controller.getHyperparameter()
        with st.form(f"form_{kunci}"):
            tema.judul_kartu("Pengaturan hyperparameter",
                             "Opsional. Jika tidak diubah, sistem memakai nilai default atau nilai tersimpan terakhir.")
            kolomA, kolomB = st.columns(2)
            nilai = {
                "windowSize": kolomA.number_input(NAMA_HP["windowSize"], value=int(hyperparameterSaatIni["windowSize"]), step=1),
                "rasioData": kolomA.number_input(NAMA_HP["rasioData"], value=float(hyperparameterSaatIni["rasioData"]), step=0.05, format="%.2f"),
                "jumlahUnit": kolomA.number_input(NAMA_HP["jumlahUnit"] + " (lapisan 2 = setengahnya)", value=int(hyperparameterSaatIni["jumlahUnit"]), step=8),
                "dropout": kolomA.number_input(NAMA_HP["dropout"], value=float(hyperparameterSaatIni["dropout"]), step=0.05, format="%.2f"),
                "epoch": kolomB.number_input(NAMA_HP["epoch"], value=int(hyperparameterSaatIni["epoch"]), step=1),
                "batchSize": kolomB.number_input(NAMA_HP["batchSize"], value=int(hyperparameterSaatIni["batchSize"]), step=1),
                "learningRate": kolomB.number_input(NAMA_HP["learningRate"], value=float(hyperparameterSaatIni["learningRate"]), step=0.0005, format="%.5f"),
            }
            simpan = st.form_submit_button("Simpan hyperparameter", type="primary")
        return nilai, simpan

    def ubahNilai(self, nilai: dict):
        """Meminta ModelController memvalidasi nilai, lalu menampilkan nilai baru atau pesan error."""
        hasil = self.controller.validasiNilai(nilai)
        if hasil["status"] == "valid":
            self.tampilkanNilaiBaru(hasil["nilai"])
        else:
            self.tampilkanPesanError(hasil["pesan"])

    def tampilkanNilaiBaru(self, nilai: dict):
        """Menampilkan nilai hyperparameter yang tersimpan."""
        st.success("Hyperparameter disimpan dan dipakai pada pelatihan berikutnya.")
        with _kartu("hp_baru"):
            tema.judul_kartu("Nilai yang dipakai")
            tema.tabel_sederhana(["Parameter", "Nilai"], _baris_hp(nilai), rata_kanan=(1,))

    def tampilkanPesanError(self, pesan: str):
        """Menampilkan pesan nilai di luar batas."""
        st.error(f"Nilai di luar batas: {pesan}. Nilai sebelumnya tetap dipakai.")

    def halamanPengaturan(self):
        """Menyusun halaman Hyperparameter mengikuti urutan pesan Sequence 3."""
        self.bukaPengaturan()
        nilai, simpan = self.tampilkanNilaiDefault("hyperparameterSaatIni")
        if simpan:
            self.ubahNilai(nilai)
        if st.button("Kembalikan ke nilai default"):
            self.controller.validasiNilai(HYPERPARAMETER_DEFAULT)
            st.rerun()


# ---------------------------------------------------------------------------
class PelatihanUI:
    def __init__(self):
        """Menyiapkan PelatihanUI dengan ModelController dan HyperparameterUI."""
        self.controller = ModelController()
        self.hyperparameterUI = HyperparameterUI()

    # Sequence 4 - Melatih model (+ 5 bagi data, 6 preprocessing, 7 evaluasi)
    def bukaHalamanPelatihan(self):
        """Menampilkan judul halaman Latih Model."""
        tema.topbar("Model", "Latih Model")

    def aturHyperparameter(self, nilai=None):
        """Bagian opsional untuk mengubah hyperparameter di halaman pelatihan."""
        with st.expander("Ubah hyperparameter (opsional)"):
            nilai, simpan = self.hyperparameterUI.tampilkanNilaiDefault("pelatihan")
            if simpan:
                hasil = self.controller.validasiNilai(nilai)
                if hasil["status"] == "valid":
                    self.tampilkanNilaiBaru(hasil["nilai"])
                else:
                    self.hyperparameterUI.tampilkanPesanError(hasil["pesan"])

    def tampilkanNilaiBaru(self, nilai: dict):
        """Menampilkan konfirmasi hyperparameter diperbarui."""
        st.success("Hyperparameter diperbarui. Nilai baru dipakai pada pelatihan ini.")

    def tampilkanHyperparameter(self):
        """Menampilkan tabel hyperparameter dan tombol latih. True bila tombol diklik."""
        with _kartu("hp_dipakai"):
            tema.judul_kartu("Hyperparameter yang dipakai",
                             "Alur: validasi, bagi data latih/uji, preprocessing, pelatihan, evaluasi (RMSE dan MAPE).")
            tema.tabel_sederhana(["Parameter", "Nilai"], _baris_hp(self.controller.getHyperparameter()), rata_kanan=(1,))
            st.write("")
            diklik = self.klikLatihModel()
        return diklik

    def klikLatihModel(self) -> bool:
        """Menampilkan tombol 'Latih model'. True bila diklik."""
        return st.button("Latih model", type="primary")

    def lanjutKePreprocessing(self):
        """Menampilkan tahap preprocessing."""
        st.write("Data latih dan data uji sudah dibagi. Lanjut ke preprocessing...")

    def lanjutKePelatihan(self):
        """Menampilkan tahap pelatihan."""
        st.write("Preprocessing selesai, window terbentuk. Lanjut ke pelatihan...")

    def tampilkanNilaiEvaluasi(self, evaluasi: dict):
        """Menampilkan RMSE dan MAPE pada data latih dan uji."""
        tema.baris_kartu([
            tema.kartu("RMSE data uji", tema.angka(evaluasi["RMSE (Rp)"]["Data uji"], 2), "rupiah", "target", "biru"),
            tema.kartu("MAPE data uji", tema.angka(evaluasi["MAPE (%)"]["Data uji"], 3) + " %", "kesalahan persentase", "aktivitas", "oranye"),
            tema.kartu("RMSE data latih", tema.angka(evaluasi["RMSE (Rp)"]["Data latih"], 2), "rupiah", "target", "teal"),
            tema.kartu("MAPE data latih", tema.angka(evaluasi["MAPE (%)"]["Data latih"], 3) + " %", "kesalahan persentase", "aktivitas", "kuning"),
        ])
        with _kartu("evaluasi"):
            tema.judul_kartu("Nilai evaluasi", "Dibandingkan pada data latih dan data uji")
            baris = [[nama, tema.angka(v["Data latih"], 6), tema.angka(v["Data uji"], 6)] for nama, v in evaluasi.items()]
            tema.tabel_sederhana(["Metrik", "Data latih", "Data uji"], baris, rata_kanan=(1, 2))

    def tampilkanPesanError(self, pesan: str):
        """Menampilkan pesan data atau parameter tidak valid."""
        st.error(f"{pesan}\n\nPerbaiki data atau parameter, lalu coba lagi.")

    def halamanPelatihan(self):
        """Menyusun halaman Latih Model mengikuti urutan pesan Sequence 4."""
        self.bukaHalamanPelatihan()
        if self.controller.dataKurs.getDataKurs() is None:
            return st.warning("Data kurs belum tersedia. Ambil data terlebih dahulu di menu Ambil Data.")
        self.aturHyperparameter()
        diklik = self.tampilkanHyperparameter()
        if not diklik:
            tersimpan = self.controller.hasilPrediksi.getHasilPrediksi()
            if tersimpan is not None:
                st.info("Hasil pelatihan terakhir.")
                self.tampilkanNilaiEvaluasi(tersimpan["evaluasi"]["tabel"])
            return
        with st.status("Melatih model...", expanded=True) as status:
            bar = st.progress(0.0, text="Menyiapkan...")

            def lapor(tahap, info=None):
                if tahap == "bagi":
                    st.write("Memvalidasi dan membagi data latih / uji...")
                elif tahap == "preprocessing":
                    self.lanjutKePreprocessing()
                elif tahap == "pelatihan":
                    self.lanjutKePelatihan()
                elif tahap == "epoch":
                    bar.progress(info["epoch"] / info["total"], text=f"Epoch {info['epoch']}/{info['total']} - loss {info['loss']:.6f}")
                elif tahap == "evaluasi":
                    st.write("Mengevaluasi model dengan data uji...")

            hasil = self.controller.latihModel(self.controller.getHyperparameter(), lapor)
            status.update(label="Pelatihan selesai" if hasil["status"] == "berhasil" else "Pelatihan gagal",
                          state="complete" if hasil["status"] == "berhasil" else "error")
        if hasil["status"] == "berhasil":
            st.success("Model terlatih. Model dan hasil prediksi tersimpan.")
            self.tampilkanNilaiEvaluasi(hasil["evaluasi"])
        else:
            self.tampilkanPesanError(hasil["pesan"])


# ---------------------------------------------------------------------------
class HasilPrediksiUI:
    def __init__(self):
        """Menyiapkan HasilPrediksiUI dengan PrediksiController."""
        self.controller = PrediksiController()

    # Sequence 8 - Melihat hasil prediksi
    def bukaHalamanHasil(self):
        """Menampilkan judul halaman Hasil Prediksi."""
        tema.topbar("Hasil", "Hasil Prediksi")

    def tampilkanPesanModelKosong(self):
        """Menampilkan peringatan model belum dilatih."""
        st.warning("Model belum tersedia, silakan latih model terlebih dahulu di menu Latih Model.")

    def tampilkanPesanDataKosong(self):
        """Menampilkan peringatan data belum tersedia."""
        st.warning("Data belum tersedia.")

    def tampilkanPesanError(self, pesan: str):
        """Menampilkan pesan kesalahan."""
        st.error(pesan)

    def tampilkanTabelDanGrafik(self, hasil: dict):
        """Menampilkan kartu ringkasan, grafik, dan tabel data uji."""
        ev, uji, depan = hasil["evaluasi"], hasil["uji"], hasil["depan"]
        tema.baris_kartu([
            tema.kartu("RMSE data uji", tema.angka(ev["rmse"]["uji"], 2), "rupiah", "target", "biru"),
            tema.kartu("MAPE data uji", tema.angka(ev["mape"]["uji"], 3) + " %", "kesalahan persentase", "aktivitas", "oranye"),
            tema.kartu("Data uji", tema.angka(len(uji)), "hari kerja", "database", "teal"),
            tema.kartu("Ramalan berikutnya", tema.angka(depan["nilaiPrediksi"].iloc[0], 2), str(depan["tanggal"].iloc[0].date()),
                       "tren", "hijau"),
        ])
        self.tampilkanGrafik(self.controller.ambilDataGrafik())
        with st.expander("Tabel data uji (aktual vs prediksi)"):
            galat = uji["nilaiPrediksi"] - uji["nilaiAktual"]
            baris = [[str(t.date()), tema.angka(a, 2), tema.angka(p, 2), _perubahan(g)]
                     for t, a, p, g in zip(uji["tanggal"], uji["nilaiAktual"], uji["nilaiPrediksi"], galat)][::-1]
            tema.tabel_berhalaman("uji", ["Tanggal", "Aktual (Rp)", "Prediksi (Rp)", "Selisih prediksi"], baris, rata_kanan=(1, 2))

    # Sequence 9 - Melihat grafik aktual vs prediksi
    def bukaBagianGrafik(self):
        """Menampilkan judul bagian grafik."""
        tema.judul_kartu("Grafik aktual vs prediksi", "Data uji")

    def tampilkanGrafik(self, fig):
        """Menampilkan grafik aktual vs prediksi, atau pesan bila data kosong."""
        with _kartu("grafik_hasil"):
            self.bukaBagianGrafik()
            if fig is None:
                return self.tampilkanPesanDataKosong()
            st.pyplot(fig)

    # Sequence 10 - Memilih rentang waktu
    def bukaPilihanRentang(self):
        """Menampilkan judul bagian ramalan ke depan."""
        tema.judul_kartu("Ramalan ke depan", "Ramalan berantai dari akhir data uji, dihitung per hari kerja.")

    def tampilkanPilihan(self) -> int:
        """Menampilkan pilihan rentang (7 atau 30 hari kerja). Mengembalikan pilihan."""
        return st.radio("Rentang waktu ramalan", list(RENTANG_VALID), format_func=lambda x: f"{x} hari kerja", horizontal=True)

    def pilihRentangWaktu(self, rentang: int):
        """Meminta ramalan sesuai rentang. Mengembalikan DataFrame, atau None bila tidak valid."""
        hasil = self.controller.ambilPrediksi(rentang)
        if hasil["status"] == "tidakValid":
            self.tampilkanPesanError(hasil["pesan"])
            return None
        return hasil["prediksi"]

    def perbaruiTabelDanGrafik(self, prediksi: pd.DataFrame, terakhir_aktual: float):
        """Menampilkan tabel ramalan sesuai rentang terpilih."""
        sebelum = prediksi["nilaiPrediksi"].shift(1)
        sebelum.iloc[0] = terakhir_aktual
        selisih = prediksi["nilaiPrediksi"] - sebelum
        baris = [[str(t.date()), tema.angka(v, 2), _perubahan(d)]
                 for t, v, d in zip(prediksi["tanggal"], prediksi["nilaiPrediksi"], selisih)]
        tema.tabel_sederhana(["Tanggal", "Ramalan kurs (Rp)", "Perubahan"], baris, rata_kanan=(1,))

    # Sequence 11 - Mengunduh / mencetak
    def klikUnduhCetak(self) -> bool:
        """Menampilkan sakelar unduh/cetak. True bila aktif."""
        return st.toggle("Unduh / cetak hasil prediksi")

    def tampilkanPilihanFormat(self) -> str:
        """Menampilkan pilihan format file (CSV atau PDF)."""
        return st.radio("Format file", ["CSV", "PDF"], horizontal=True)

    def pilihFormat(self, format: str) -> dict:
        """Meminta PrediksiController membuat file dalam format terpilih."""
        return self.controller.buatFile(format)

    def sediakanFile(self, berkas: dict):
        """Menampilkan tombol unduh untuk file yang sudah dibuat."""
        st.download_button("Unduh file", data=berkas["berkas"], file_name=berkas["nama"], mime=berkas["mime"], type="primary")

    def halamanHasil(self):
        """Menyusun halaman Hasil Prediksi mengikuti urutan pesan Sequence 8 sampai 11."""
        self.bukaHalamanHasil()
        hasil = self.controller.periksaModelTersedia()
        if hasil is None:
            return self.tampilkanPesanModelKosong()
        self.tampilkanTabelDanGrafik(hasil)
        with _kartu("ramalan"):
            self.bukaPilihanRentang()
            prediksi = self.pilihRentangWaktu(self.tampilkanPilihan())
            if prediksi is not None:
                self.perbaruiTabelDanGrafik(prediksi, float(hasil["uji"]["nilaiAktual"].iloc[-1]))
        with _kartu("unduh"):
            tema.judul_kartu("Unduh atau cetak", "Berkas berisi data uji dan ramalan sesuai rentang yang dipilih.")
            if self.klikUnduhCetak():
                berkas = self.pilihFormat(self.tampilkanPilihanFormat())
                if berkas["status"] == "berhasil":
                    self.sediakanFile(berkas)
                else:
                    self.tampilkanPesanError(berkas["pesan"])


# ---------------------------------------------------------------------------
class DashboardUI:
    """Ringkasan keadaan sistem. Tidak ada di diagram dan tidak mengubah data."""

    def halamanDashboard(self):
        """Menampilkan ringkasan keadaan sistem (hanya baca)."""
        data = DataKurs().getDataKurs()
        latih, uji = DataKurs().getDataLatihUji()
        hasil = HasilPrediksi().getHasilPrediksi()
        tema.topbar("", "Dashboard")
        kolomKiri, kolomKanan = st.columns([5, 1], vertical_alignment="center")
        kolomKiri.markdown(f'<p class="jdl" style="font-size:20px">{tema.sapaan()}, Peneliti!</p>'
                    f'<p class="sub">{tema.tanggal_id(dt.date.today())}</p>', unsafe_allow_html=True)
        kolomKanan.button("Perbarui")

        kosong = "-"
        dataUrut = None
        if data is not None:
            dataUrut = data.sort_values("tanggal")
            n_data = tema.angka(len(dataUrut))
            rentang = str(dataUrut["tanggal"].iloc[0].date())
            rentang_ket = "s.d. " + str(dataUrut["tanggal"].iloc[-1].date())
            terakhir = tema.angka(dataUrut["nilaiKurs"].iloc[-1], 2)
        else:
            n_data = rentang = terakhir = kosong
            rentang_ket = "Belum ada data"
        pembagian = f"{len(latih)} / {len(uji)}" if latih is not None and len(latih) else kosong
        ev = hasil["evaluasi"] if hasil else None
        tema.baris_kartu([
            tema.kartu("Data kurs", n_data, "baris tersimpan" if data is not None else "Belum ada data", "database", "biru"),
            tema.kartu("Kurs terakhir", terakhir, "rupiah per dolar" if data is not None else "Belum ada data", "aktivitas", "hijau"),
            tema.kartu("Data latih / uji", pembagian, "baris", "lapisan", "teal"),
            tema.kartu("Model LSTM", "Terlatih" if hasil else "Belum", "siap meramal" if hasil else "Belum dilatih", "cek",
                       "hijau" if hasil else "abu"),
        ])
        tema.baris_kartu([
            tema.kartu("RMSE data uji", tema.angka(ev["rmse"]["uji"], 2) if ev else kosong, "rupiah" if ev else "Belum ada hasil", "target", "biru"),
            tema.kartu("MAPE data uji", (tema.angka(ev["mape"]["uji"], 3) + " %") if ev else kosong,
                       "kesalahan persentase" if ev else "Belum ada hasil", "aktivitas", "oranye"),
            tema.kartu("Ramalan berikutnya", tema.angka(hasil["depan"]["nilaiPrediksi"].iloc[0], 2) if hasil else kosong,
                       str(hasil["depan"]["tanggal"].iloc[0].date()) if hasil else "Belum ada hasil", "tren", "teal"),
            tema.kartu("Rentang data", rentang, rentang_ket, "kalender", "kuning"),
        ])
        kiri, kanan = st.columns([3, 2])
        with kiri:
            with _kartu("dash_grafik"):
                tema.judul_kartu("Kurs 90 hari terakhir", "Dari data yang tersimpan")
                if dataUrut is None:
                    st.info("Belum ada data. Mulai dari menu Ambil Data.")
                else:
                    st.pyplot(grafik.historis(dataUrut.tail(90)))
        with kanan:
            with _kartu("dash_status"):
                tema.judul_kartu("Status proses", "Urutan pengerjaan sistem")
                langkah = [
                    ("Ambil data kurs", "Menu Ambil Data", data is not None, "Tersedia", "Belum ada"),
                    ("Bagi data latih dan uji", "Terjadi saat melatih model", latih is not None and len(latih) > 0, "Selesai", "Belum"),
                    ("Latih model LSTM", "Menu Latih Model", hasil is not None, "Terlatih", "Belum"),
                    ("Hasil prediksi", "Menu Hasil Prediksi", hasil is not None, "Siap", "Belum"),
                ]
                st.markdown("".join(
                    '<div class="stat-row"><div><b>' + n + "</b><small>" + ket + "</small></div>"
                    + tema.pill(ya if ok else tidak, "hijau" if ok else "abu") + "</div>"
                    for n, ket, ok, ya, tidak in langkah), unsafe_allow_html=True)
