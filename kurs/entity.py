"""Entity: DataKurs, ModelLSTM, HasilPrediksi (sesuai Class Diagram).

Data disimpan sebagai berkas di folder ``storage/`` supaya tetap ada setelah
halaman Streamlit di-refresh. Nama class dan method sama dengan Class Diagram.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

STORAGE = Path(__file__).resolve().parent.parent / "storage"
STORAGE.mkdir(exist_ok=True)

HYPERPARAMETER_DEFAULT = {
    "windowSize": 80,       # ukuran window (hari)
    "rasioData": 0.8,       # porsi data latih
    "jumlahUnit": 64,       # unit LSTM lapisan 1 (lapisan 2 = setengahnya)
    "dropout": 0.2,
    "epoch": 50,
    "batchSize": 32,
    "learningRate": 0.001,
}


class DataKurs:
    """Entity data kurs. Menyimpan dan membaca data kurs historis serta data latih dan uji. Atribut: idKurs, tanggal, nilaiKurs, jenisData."""

    FILE_KURS = STORAGE / "kurs.csv"
    FILE_LATIH_UJI = STORAGE / "kurs_latih_uji.csv"

    def __init__(self):
        """Atribut diagram: baris data kurs terbaru yang disimpan atau dibaca."""
        self._idKurs = 0
        self._tanggal = None
        self._nilaiKurs = 0.0
        self._jenisData = "Historis"

    def _catatBarisTerbaru(self, df: pd.DataFrame):
        """Mengisi atribut (idKurs, tanggal, nilaiKurs) dari baris terakhir data."""
        baris = df.iloc[-1]
        self._idKurs, self._tanggal, self._nilaiKurs = int(baris["idKurs"]), baris["tanggal"], float(baris["nilaiKurs"])

    def simpanData(self, data: pd.DataFrame) -> bool:
        """Menyimpan data kurs (diurutkan, diberi idKurs) ke berkas dan mengosongkan pembagian latih/uji lama."""
        df = data[["tanggal", "nilaiKurs"]].copy()
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df = df.sort_values("tanggal").reset_index(drop=True)
        df.insert(0, "idKurs", range(1, len(df) + 1))
        df.to_csv(self.FILE_KURS, index=False)
        self._catatBarisTerbaru(df)
        if self.FILE_LATIH_UJI.exists():   # data lama tidak berlaku lagi
            self.FILE_LATIH_UJI.unlink()
        return True

    def getDataKurs(self):
        """Membaca data kurs tersimpan. Mengembalikan DataFrame, atau None bila belum ada data."""
        if not self.FILE_KURS.exists():
            return None
        df = pd.read_csv(self.FILE_KURS, parse_dates=["tanggal"])
        if len(df) == 0:
            return None
        self._catatBarisTerbaru(df)
        return df

    def simpanDataLatihUji(self, dataLatih: pd.DataFrame, dataUji: pd.DataFrame) -> bool:
        """Menyimpan data latih dan data uji (diberi jenisData 'Latih' atau 'Uji')."""
        dataLatihBerlabel = dataLatih.copy()
        dataLatihBerlabel["jenisData"] = "Latih"
        dataUjiBerlabel = dataUji.copy()
        dataUjiBerlabel["jenisData"] = "Uji"
        pd.concat([dataLatihBerlabel, dataUjiBerlabel], ignore_index=True).to_csv(self.FILE_LATIH_UJI, index=False)
        return True

    def getDataLatihUji(self):
        """Mengembalikan tuple (dataLatih, dataUji), atau (None, None) bila belum dibagi."""
        if not self.FILE_LATIH_UJI.exists():
            return None, None
        df = pd.read_csv(self.FILE_LATIH_UJI, parse_dates=["tanggal"])
        latih = df[df["jenisData"] == "Latih"].drop(columns="jenisData").reset_index(drop=True)
        uji = df[df["jenisData"] == "Uji"].drop(columns="jenisData").reset_index(drop=True)
        return latih, uji


class ModelLSTM:
    """Entity model LSTM. Menyimpan hyperparameter, melatih model, membuat prediksi, dan meramal ke depan."""

    FILE_HP = STORAGE / "hyperparameter.json"
    FILE_MODEL = STORAGE / "model_lstm.keras"

    def __init__(self):
        """Membuat model kosong dan memuat hyperparameter (default atau tersimpan terakhir)."""
        self.model = None
        self._idModel = 1
        self._sinkronkanAtribut(self.getHyperparameter())

    def _sinkronkanAtribut(self, hp: dict):
        """Menyalin hyperparameter ke atribut diagram supaya selalu sama dengan yang tersimpan."""
        self._jumlahUnit, self._epoch, self._batchSize = int(hp["jumlahUnit"]), int(hp["epoch"]), int(hp["batchSize"])
        self._learningRate, self._windowSize = float(hp["learningRate"]), int(hp["windowSize"])

    # --- hyperparameter -------------------------------------------------
    def getHyperparameter(self) -> dict:
        """Mengembalikan hyperparameter tersimpan terakhir, atau nilai default."""
        hp = dict(HYPERPARAMETER_DEFAULT)
        if self.FILE_HP.exists():
            try:
                hp.update(json.loads(self.FILE_HP.read_text()))
            except (ValueError, OSError):
                pass
        return hp

    def setHyperparameter(self, nilai: dict) -> bool:
        """Menyimpan hyperparameter baru (nilai sudah divalidasi ModelController)."""
        hp = self.getHyperparameter()
        hp.update(nilai)
        self.FILE_HP.write_text(json.dumps(hp, indent=2))
        self._sinkronkanAtribut(hp)
        return True

    # --- model ----------------------------------------------------------
    def latih(self, dataLatih, hyperparameter: dict, onEpoch=None):
        """dataLatih = (X, y) hasil window. Mengembalikan model terlatih."""
        from tensorflow import keras

        X, y = dataLatih
        hp = hyperparameter
        keras.utils.set_random_seed(42)
        unit1 = int(hp["jumlahUnit"])
        unit2 = max(4, unit1 // 2)
        model = keras.Sequential([
            keras.layers.Input(shape=(int(hp["windowSize"]), 1)),
            keras.layers.LSTM(unit1, return_sequences=True),
            keras.layers.Dropout(float(hp["dropout"])),
            keras.layers.LSTM(unit2),
            keras.layers.Dropout(float(hp["dropout"])),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(float(hp["learningRate"])), loss="mse")

        total = int(hp["epoch"])

        class _Laporan(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if onEpoch:
                    onEpoch(epoch + 1, total, float((logs or {}).get("loss", 0.0)))

        model.fit(X, y, epochs=total, batch_size=int(hp["batchSize"]),
                  shuffle=False, verbose=0, callbacks=[_Laporan()])
        model.save(self.FILE_MODEL)
        self.model = model
        return model

    def prediksi(self, dataUji) -> np.ndarray:
        """Memprediksi data (X, y) atau X yang sudah ternormalisasi. Mengembalikan array 1 dimensi."""
        X = dataUji[0] if isinstance(dataUji, tuple) else dataUji
        return self.model.predict(X, verbose=0).ravel()

    def ramal(self, windowTerakhir: np.ndarray, hari: int) -> np.ndarray:
        """Ramalan berantai ke depan (tambahan di luar diagram)."""
        ukuranWindow = len(windowTerakhir)
        seri = list(map(float, windowTerakhir))
        for _ in range(hari):
            masukanModel = np.array(seri[-ukuranWindow:], dtype="float32").reshape(1, ukuranWindow, 1)
            seri.append(float(self.model(masukanModel, training=False).numpy()[0, 0]))
        return np.array(seri[-hari:])


class HasilPrediksi:
    """Entity hasil prediksi. Menyimpan hasil uji, ramalan ke depan, dan nilai evaluasi (RMSE dan MAPE)."""

    FILE_UJI = STORAGE / "hasil_uji.csv"
    FILE_DEPAN = STORAGE / "hasil_depan.csv"
    FILE_EVAL = STORAGE / "evaluasi.json"

    def __init__(self):
        """Atribut diagram: hasil uji terbaru (baris terakhir) dan nilai evaluasi data uji."""
        self._idHasil = 0
        self._tanggal = None
        self._nilaiAktual = 0.0
        self._nilaiPrediksi = 0.0
        self._rmse = 0.0
        self._mape = 0.0

    def _catatHasilTerbaru(self, hasil: dict):
        """Mengisi atribut dari baris terakhir data uji dan evaluasi."""
        baris = hasil["uji"].iloc[-1]
        self._idHasil, self._tanggal = int(baris["idHasil"]), baris["tanggal"]
        self._nilaiAktual, self._nilaiPrediksi = float(baris["nilaiAktual"]), float(baris["nilaiPrediksi"])
        self._rmse, self._mape = float(hasil["evaluasi"]["rmse"]["uji"]), float(hasil["evaluasi"]["mape"]["uji"])

    def simpanHasilPrediksi(self, prediksi: dict, RMSE: dict, MAPE: dict) -> bool:
        """prediksi = {"uji": DataFrame, "depan": DataFrame, "evaluasi": dict}"""
        uji = prediksi["uji"].copy()
        uji.insert(0, "idHasil", range(1, len(uji) + 1))
        uji.to_csv(self.FILE_UJI, index=False)
        prediksi["depan"].to_csv(self.FILE_DEPAN, index=False)
        ev = dict(prediksi.get("evaluasi", {}))
        ev["rmse"] = RMSE
        ev["mape"] = MAPE
        self.FILE_EVAL.write_text(json.dumps(ev, indent=2))
        return True

    def getHasilPrediksi(self):
        """Mengembalikan dict {uji, depan, evaluasi}, atau None bila model belum dilatih."""
        if not (self.FILE_UJI.exists() and self.FILE_DEPAN.exists() and self.FILE_EVAL.exists()):
            return None
        hasil = {
            "uji": pd.read_csv(self.FILE_UJI, parse_dates=["tanggal"]),
            "depan": pd.read_csv(self.FILE_DEPAN, parse_dates=["tanggal"]),
            "evaluasi": json.loads(self.FILE_EVAL.read_text()),
        }
        self._catatHasilTerbaru(hasil)
        return hasil

    def getAktualDanPrediksi(self):
        """Mengembalikan (dataAktual, dataPrediksi) pada data uji, atau (None, None)."""
        hasil = self.getHasilPrediksi()
        if hasil is None:
            return None, None
        uji = hasil["uji"]
        return uji[["tanggal", "nilaiAktual"]], uji[["tanggal", "nilaiPrediksi"]]

    def getPrediksi(self, rentang: int = None):
        """Mengembalikan ramalan ke depan sebanyak `rentang` hari kerja (semua bila None)."""
        hasil = self.getHasilPrediksi()
        if hasil is None:
            return None
        depan = hasil["depan"]
        return depan if rentang is None else depan.head(int(rentang))
