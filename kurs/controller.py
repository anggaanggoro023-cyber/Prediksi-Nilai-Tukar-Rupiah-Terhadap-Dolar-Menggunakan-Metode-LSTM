"""Control: DataController, ModelController, PrediksiController.

Nama method sama dengan Class Diagram dan Sequence Diagram. Class ini tidak
memakai Streamlit, jadi bisa diuji tanpa tampilan.
"""
import io
import json
import math

import numpy as np
import pandas as pd

from .entity import STORAGE, DataKurs, HasilPrediksi, ModelLSTM

BATAS = {  # nama: (minimum, maksimum, tipe)
    "windowSize": (10, 200, int),
    "rasioData": (0.5, 0.95, float),
    "jumlahUnit": (8, 256, int),
    "dropout": (0.0, 0.5, float),
    "epoch": (1, 500, int),
    "batchSize": (4, 256, int),
    "learningRate": (0.00001, 0.1, float),
}
RENTANG_VALID = (7, 30)
HARI_RAMAL = 30


class DataController:
    def __init__(self):
        """Menyiapkan DataController dengan referensi ke entity DataKurs."""
        self.periode = None
        self.dataKurs = DataKurs()

    def ambilData(self, periode, sumber) -> dict:
        """Sequence 1: sumber.requestDataKurs -> DataKurs.simpanData -> ringkasan."""
        self.periode = periode
        try:
            data = sumber.requestDataKurs(periode)
        except Exception as e:  # koneksi gagal / data kosong
            return {"status": "gagal", "pesan": str(e)}
        if data is None or len(data) == 0:
            return {"status": "gagal", "pesan": "Data kosong."}
        self.dataKurs.simpanData(data)
        return {"status": "berhasil", "ringkasan": {
            "jumlahBaris": int(len(data)),
            "tanggalAwal": pd.to_datetime(data["tanggal"]).min().date(),
            "tanggalAkhir": pd.to_datetime(data["tanggal"]).max().date(),
            "kursMinimum": float(data["nilaiKurs"].min()),
            "kursMaksimum": float(data["nilaiKurs"].max()),
        }}

    def periksaDataTersedia(self):
        """Sequence 2: mengembalikan DataFrame, atau None jika data kosong."""
        return self.dataKurs.getDataKurs()


class ModelController:
    def __init__(self):
        """Menyiapkan ModelController dengan referensi ke DataKurs, ModelLSTM, dan HasilPrediksi."""
        self.dataKurs = DataKurs()
        self.modelLSTM = ModelLSTM()
        self.hasilPrediksi = HasilPrediksi()
        self.rasioData = 0.8
        self._hp = self.modelLSTM.getHyperparameter()
        self._scaler = None
        self._tanggalUji = None
        self._aktualUji = None
        self._prediksiUji = None

    # ---- Sequence 3: hyperparameter -----------------------------------
    def getHyperparameter(self) -> dict:
        """Mengambil hyperparameter dari ModelLSTM."""
        return self.modelLSTM.getHyperparameter()

    def validasiNilai(self, nilai: dict) -> dict:
        """Memeriksa tipe dan batas nilai; bila valid disimpan ke ModelLSTM. Mengembalikan dict status."""
        bersih, galat = {}, []
        for namaParameter, nilaiMasukan in nilai.items():
            if namaParameter not in BATAS:
                continue
            batasBawah, batasAtas, tipe = BATAS[namaParameter]
            try:
                nilaiTerkonversi = tipe(nilaiMasukan)
            except (TypeError, ValueError):
                galat.append(f"{namaParameter}: harus berupa angka")
                continue
            if not (batasBawah <= nilaiTerkonversi <= batasAtas):
                galat.append(f"{namaParameter}: harus di antara {batasBawah} dan {batasAtas} (anda mengisi {nilaiTerkonversi})")
            else:
                bersih[namaParameter] = nilaiTerkonversi
        if galat:
            return {"status": "tidakValid", "pesan": "; ".join(galat), "nilai": self.getHyperparameter()}
        self.modelLSTM.setHyperparameter(bersih)
        return {"status": "valid", "nilai": self.getHyperparameter()}

    # ---- Sequence 4: melatih model ------------------------------------
    def validasiDataDanParameter(self):
        """Memastikan data ada dan cukup untuk window dan rasio. Mengembalikan (valid, pesan)."""
        df = self.dataKurs.getDataKurs()
        if df is None:
            return False, "Data kurs belum tersedia. Ambil data terlebih dahulu."
        hp = self._hp
        cek = self.validasiNilai(hp)
        if cek["status"] != "valid":
            return False, cek["pesan"]
        butuh = max(math.ceil((hp["windowSize"] + 10) / hp["rasioData"]), math.ceil(5 / (1 - hp["rasioData"])))
        if len(df) < butuh:
            return False, (f"Data kurang: ada {len(df)} baris, dibutuhkan minimal {butuh} baris "
                           f"untuk window {hp['windowSize']} dan rasio latih {hp['rasioData']}.")
        return True, ""

    def latihModel(self, hyperparameter: dict = None, progress=None) -> dict:
        """Alur pelatihan penuh (Sequence 4): validasi, bagi data, preprocessing, latih, evaluasi."""
        lapor = progress or (lambda tahap, info=None: None)
        self._hp = {**self.getHyperparameter(), **(hyperparameter or {})}
        self.rasioData = float(self._hp["rasioData"])
        # Pesan 7: validasiDataDanParameter()  (lalu pesan 8-9: getDataKurs() di dalamnya)
        valid, pesan = self.validasiDataDanParameter()
        if not valid:
            return {"status": "gagal", "pesan": pesan}

        # Pesan 10: bagiData(dataKurs)
        lapor("bagi")
        hasilTahap = self.bagiData(self.rasioData)
        if hasilTahap["status"] != "berhasil":
            return hasilTahap
        # Pesan 11: preprocessing(dataLatih, dataUji)
        lapor("preprocessing")
        hasilTahap = self.preprocessing()
        if hasilTahap["status"] != "berhasil":
            return hasilTahap
        # Pesan 12-13: ModelLSTM.latih(dataLatih, hyperparameter) -> modelTerlatih
        lapor("pelatihan")
        self.modelLSTM.latih((self._X_latih, self._y_latih), self._hp,
                             onEpoch=lambda epochKe, totalEpoch, loss: lapor("epoch", {"epoch": epochKe, "total": totalEpoch, "loss": loss}))
        # Pesan 14-16: evaluasiModel(dataUji) lalu simpanHasilPrediksi(prediksi, RMSE, MAPE)
        lapor("evaluasi")
        evaluasi = self.evaluasiModel((self._X_uji, self._y_uji))
        return {"status": "berhasil", "evaluasi": evaluasi}

    # ---- Sequence 5: membagi data -------------------------------------
    def bagiData(self, rasio: float) -> dict:
        """Membagi data berurutan menjadi data latih dan uji, lalu menyimpannya (Sequence 5)."""
        df = self.dataKurs.getDataKurs()
        data = self.urutkanBerdasarkanTanggal(df)
        latih, uji = self.pisahkanData(rasio, data)
        if len(latih) < 30 or len(uji) < 5:
            return {"status": "gagal", "pesan": "Data latih atau data uji tidak cukup."}
        self.dataKurs.simpanDataLatihUji(latih, uji)
        return {"status": "berhasil", "jumlahLatih": len(latih), "jumlahUji": len(uji)}

    def urutkanBerdasarkanTanggal(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Mengurutkan data kurs dari tanggal terlama ke terbaru."""
        data = self.dataKurs.getDataKurs() if data is None else data
        return data.sort_values("tanggal").reset_index(drop=True)

    def pisahkanData(self, rasio: float, data: pd.DataFrame = None):
        """Memotong data urut menjadi (latih, uji) menurut rasio; tidak diacak karena deret waktu."""
        data = self.urutkanBerdasarkanTanggal(data)
        jumlahLatih = int(len(data) * rasio)           # berurutan, bukan acak (data deret waktu)
        return data.iloc[:jumlahLatih].reset_index(drop=True), data.iloc[jumlahLatih:].reset_index(drop=True)

    # ---- Sequence 6: preprocessing ------------------------------------
    def preprocessing(self) -> dict:
        """Membersihkan data, fit scaler pada data latih, transformasi, dan membentuk window (Sequence 6)."""
        latih, uji = self.dataKurs.getDataLatihUji()
        self._latih, self._uji = latih, uji
        self.bersihkanData()
        self.fitScaler(self._latih)
        self.transform(self._latih, self._uji)
        return self.bentukWindow(int(self._hp["windowSize"]))

    def bersihkanData(self):
        """Membuang data kosong dan tanggal ganda pada data latih dan uji."""
        for nama in ("_latih", "_uji"):
            dataBersih = getattr(self, nama).dropna(subset=["tanggal", "nilaiKurs"]).drop_duplicates("tanggal")
            setattr(self, nama, dataBersih.sort_values("tanggal").reset_index(drop=True))

    def fitScaler(self, dataLatih: pd.DataFrame):
        """Scaler hanya dilatih pada data latih (mencegah kebocoran data)."""
        batasBawah, batasAtas = float(dataLatih["nilaiKurs"].min()), float(dataLatih["nilaiKurs"].max())
        if batasAtas - batasBawah < 1e-9:
            batasAtas = batasBawah + 1.0
        self._scaler = {"min": batasBawah, "max": batasAtas}
        (STORAGE / "scaler.json").write_text(json.dumps(self._scaler))

    def transform(self, dataLatih: pd.DataFrame, dataUji: pd.DataFrame):
        """Menormalkan data latih dan uji dengan scaler yang dilatih dari data latih."""
        batasBawah, batasAtas = self._scaler["min"], self._scaler["max"]
        self._sLatih = ((dataLatih["nilaiKurs"].to_numpy(dtype=float) - batasBawah) / (batasAtas - batasBawah))
        self._sUji = ((dataUji["nilaiKurs"].to_numpy(dtype=float) - batasBawah) / (batasAtas - batasBawah))

    def bentukWindow(self, windowSize: int) -> dict:
        """Membentuk pasangan (X, y) sliding window untuk data latih dan data uji."""
        if len(self._sLatih) <= windowSize or len(self._sUji) < 1:
            return {"status": "gagal", "pesan": "Data kurang dari window size."}

        def jendela(seri, awal):
            X = np.array([seri[i:i + windowSize] for i in range(awal, len(seri) - windowSize)], dtype="float32")
            y = np.array([seri[i + windowSize] for i in range(awal, len(seri) - windowSize)], dtype="float32")
            return X.reshape(-1, windowSize, 1), y

        self._X_latih, self._y_latih = jendela(self._sLatih, 0)
        konteks = np.concatenate([self._sLatih[-windowSize:], self._sUji])  # data uji ditebak dengan konteks akhir data latih
        self._X_uji, self._y_uji = jendela(konteks, 0)
        self._konteks = konteks
        self._tanggalUji = self._uji["tanggal"].reset_index(drop=True)
        return {"status": "berhasil", "jumlahWindowLatih": len(self._X_latih), "jumlahWindowUji": len(self._X_uji)}

    # ---- Sequence 7: evaluasi -----------------------------------------
    def denormalisasi(self, hasilPrediksi) -> np.ndarray:
        """Mengembalikan nilai ternormalisasi ke skala rupiah."""
        batasBawah, batasAtas = self._scaler["min"], self._scaler["max"]
        return np.asarray(hasilPrediksi, dtype=float) * (batasAtas - batasBawah) + batasBawah

    @staticmethod
    def _rmse(nilaiAktual, nilaiPrediksi):
        """Root Mean Squared Error antara nilai aktual dan prediksi."""
        return float(np.sqrt(np.mean((np.asarray(nilaiAktual) - np.asarray(nilaiPrediksi)) ** 2)))

    @staticmethod
    def _mape(nilaiAktual, nilaiPrediksi):
        """Mean Absolute Percentage Error (persen); nilai aktual nol diabaikan."""
        nilaiAktual, nilaiPrediksi = np.asarray(nilaiAktual, dtype=float), np.asarray(nilaiPrediksi, dtype=float)
        bukanNol = nilaiAktual != 0                                   # asumsi: nilai aktual tidak nol
        return float(np.mean(np.abs((nilaiAktual[bukanNol] - nilaiPrediksi[bukanNol]) / nilaiAktual[bukanNol])) * 100)

    def hitungRMSE(self) -> float:
        """RMSE data uji (rupiah)."""
        return self._rmse(self._aktualUji, self._prediksiUji)

    def hitungMAPE(self) -> float:
        """MAPE data uji (persen)."""
        return self._mape(self._aktualUji, self._prediksiUji)

    def evaluasiModel(self, dataUji) -> dict:
        """Menguji model, menghitung RMSE dan MAPE, meramal ke depan, dan menyimpan hasil (Sequence 7)."""
        prediksiUjiNormal = self.modelLSTM.prediksi(dataUji)
        prediksiLatihNormal = self.modelLSTM.prediksi((self._X_latih,))
        self._prediksiUji = self.denormalisasi(prediksiUjiNormal)
        self._aktualUji = self._uji["nilaiKurs"].to_numpy(dtype=float)
        rmseUji, mapeUji = self.hitungRMSE(), self.hitungMAPE()

        aktualLatih = self.denormalisasi(self._y_latih)
        prediksiLatih = self.denormalisasi(prediksiLatihNormal)
        rmseLatih, mapeLatih = self._rmse(aktualLatih, prediksiLatih), self._mape(aktualLatih, prediksiLatih)
        skala = self._scaler["max"] - self._scaler["min"]
        mseNormal = {"latih": float(np.mean((self._y_latih - prediksiLatihNormal) ** 2)),
                 "uji": float(np.mean((self._y_uji - prediksiUjiNormal) ** 2))}
        evaluasi = {
            "MSE (skala 0-1)": {"Data latih": mseNormal["latih"], "Data uji": mseNormal["uji"]},
            "RMSE (skala 0-1)": {"Data latih": math.sqrt(mseNormal["latih"]), "Data uji": math.sqrt(mseNormal["uji"])},
            "RMSE (Rp)": {"Data latih": rmseLatih, "Data uji": rmseUji},
            "MAPE (%)": {"Data latih": mapeLatih, "Data uji": mapeUji},
        }
        # ramalan berantai ke depan (30 hari kerja), dipilih 7 atau 30 hari di halaman hasil
        ramalNormal = self.modelLSTM.ramal(self._konteks[-int(self._hp["windowSize"]):], HARI_RAMAL)
        tanggalRamal = pd.bdate_range(self._tanggalUji.iloc[-1] + pd.Timedelta(days=1), periods=HARI_RAMAL)
        depan = pd.DataFrame({"tanggal": tanggalRamal, "nilaiPrediksi": self.denormalisasi(ramalNormal)})
        uji = pd.DataFrame({"tanggal": self._tanggalUji, "nilaiAktual": self._aktualUji,
                            "nilaiPrediksi": self._prediksiUji})
        self.hasilPrediksi.simpanHasilPrediksi(
            {"uji": uji, "depan": depan, "evaluasi": {"tabel": evaluasi, "hyperparameter": self._hp}},
            RMSE={"latih": rmseLatih, "uji": rmseUji}, MAPE={"latih": mapeLatih, "uji": mapeUji})
        return evaluasi


class PrediksiController:
    def __init__(self):
        """Menyiapkan PrediksiController dengan referensi ke entity HasilPrediksi."""
        self.rentangWaktu = 7
        self.formatFile = "CSV"
        self.hasilPrediksi = HasilPrediksi()

    def periksaModelTersedia(self):
        """Mengembalikan hasil prediksi, atau None jika model belum dilatih."""
        return self.hasilPrediksi.getHasilPrediksi()

    def validasiPilihan(self, rentang) -> bool:
        """True bila rentang termasuk pilihan yang diizinkan (RENTANG_VALID)."""
        return rentang in RENTANG_VALID

    def ambilPrediksi(self, rentang: int) -> dict:
        """Mengambil ramalan sesuai rentang; menolak pilihan yang tidak valid (Sequence 10)."""
        if not self.validasiPilihan(rentang):
            return {"status": "tidakValid", "pesan": f"Pilihan rentang harus {RENTANG_VALID[0]} atau {RENTANG_VALID[1]} hari."}
        self.rentangWaktu = rentang
        return {"status": "valid", "prediksi": self.hasilPrediksi.getPrediksi(rentang)}

    def ambilDataGrafik(self):
        """Membuat grafik aktual vs prediksi dari data uji (Sequence 9). None bila belum ada hasil."""
        aktual, prediksi = self.hasilPrediksi.getAktualDanPrediksi()
        if aktual is None:
            return None
        return self.buatGrafik(aktual, prediksi)

    def buatGrafik(self, aktual: pd.DataFrame, prediksi: pd.DataFrame, depan: pd.DataFrame = None):
        """Menggambar grafik aktual, prediksi, dan (opsional) ramalan ke depan."""
        from . import grafik
        return grafik.aktual_vs_prediksi(aktual, prediksi, depan)

    def buatFile(self, format: str) -> dict:
        """Mengambil hasil prediksi lalu membuat file unduhan (Sequence 11)."""
        hasil = self.hasilPrediksi.getHasilPrediksi()
        if hasil is None:
            return {"status": "gagal", "pesan": "Data prediksi tidak tersedia."}
        return self.buatFileHasil(hasil, format)

    def buatFileHasil(self, data: dict, format: str) -> dict:
        """Membuat berkas CSV atau PDF dari data hasil. Mengembalikan dict berisi byte berkas."""
        self.formatFile = format
        uji = data["uji"].assign(jenis="Uji (aktual vs prediksi)")
        depan = data["depan"].head(self.rentangWaktu).assign(nilaiAktual=np.nan, jenis=f"Ramalan {self.rentangWaktu} hari")
        tabelGabungan = pd.concat([uji[["tanggal", "nilaiAktual", "nilaiPrediksi", "jenis"]],
                           depan[["tanggal", "nilaiAktual", "nilaiPrediksi", "jenis"]]], ignore_index=True)
        tabelGabungan["tanggal"] = pd.to_datetime(tabelGabungan["tanggal"]).dt.strftime("%Y-%m-%d")
        if format.upper() == "CSV":
            return {"status": "berhasil", "berkas": tabelGabungan.to_csv(index=False).encode("utf-8"),
                    "nama": "hasil_prediksi_kurs.csv", "mime": "text/csv"}
        if format.upper() == "PDF":
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt

            buf = io.BytesIO()
            with PdfPages(buf) as pdf:
                fig = self.buatGrafik(data["uji"][["tanggal", "nilaiAktual"]], data["uji"][["tanggal", "nilaiPrediksi"]],
                                      data["depan"].head(self.rentangWaktu))
                ev = data["evaluasi"]
                fig.suptitle("Prediksi Nilai Tukar Rupiah terhadap Dolar (LSTM)", fontsize=13, y=0.99)
                fig.text(0.01, 0.01, f"RMSE uji: Rp {ev['rmse']['uji']:.2f}   |   MAPE uji: {ev['mape']['uji']:.3f}%", fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)
                baris = tabelGabungan.tail(45).round(2).fillna("-").astype(str).values.tolist()
                fig2, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                tabelPdf = ax.table(cellText=baris, colLabels=["Tanggal", "Aktual", "Prediksi", "Jenis"], loc="upper center")
                tabelPdf.auto_set_font_size(False)
                tabelPdf.set_fontsize(8)
                tabelPdf.scale(1, 1.25)
                pdf.savefig(fig2)
                plt.close(fig2)
            return {"status": "berhasil", "berkas": buf.getvalue(), "nama": "hasil_prediksi_kurs.pdf", "mime": "application/pdf"}
        return {"status": "gagal", "pesan": "Format tidak dikenal."}
