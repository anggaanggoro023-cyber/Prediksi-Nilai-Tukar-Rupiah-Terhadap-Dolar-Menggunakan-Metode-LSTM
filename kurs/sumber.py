"""Sumber data eksternal (pengganti class BankIndonesia di diagram).

Kedua sumber punya method ``requestDataKurs(periode)`` yang mengembalikan
DataFrame berkolom ``tanggal`` dan ``nilaiKurs``.
"""
import re
from datetime import timedelta

import pandas as pd


def _ke_angka(seri: pd.Series) -> pd.Series:
    """Ubah teks seperti 'Rp 15.416,00' atau '15,416.00' menjadi angka."""
    if pd.api.types.is_numeric_dtype(seri):
        return seri.astype(float)

    def _satu(x):
        if pd.isna(x):
            return float("nan")
        s = re.sub(r"[^0-9,.\-]", "", str(x))
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
        elif s.count(".") > 1 or (s.count(".") == 1 and len(s.split(".")[-1]) == 3 and len(s) > 5):
            s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return seri.map(_satu)


class YahooFinance:
    """Ambil kurs dari Yahoo Finance (default ticker IDR=X = USD/IDR)."""

    def __init__(self, ticker: str = "IDR=X"):
        """Menyimpan ticker (default IDR=X)."""
        self.ticker = ticker.strip() or "IDR=X"

    def requestDataKurs(self, periode) -> pd.DataFrame:
        """Mengunduh kurs penutupan pada periode (mulai, akhir). Dilempar ValueError bila kosong."""
        import yfinance as yf

        mulai, akhir = periode
        df = yf.download(self.ticker, start=mulai, end=akhir + timedelta(days=1), progress=False)
        if df is None or df.empty:
            raise ValueError("Data kosong. Periksa ticker, periode, atau koneksi internet.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out = df[["Close"]].reset_index()
        out.columns = ["tanggal", "nilaiKurs"]
        out["tanggal"] = pd.to_datetime(out["tanggal"]).dt.tz_localize(None).dt.normalize()
        return out.dropna()


class UnggahFile:
    """Pakai berkas CSV/Excel (misalnya unduhan kurs JISDOR dari Bank Indonesia)."""

    def __init__(self, berkas, kolomTanggal: str, kolomNilai: str):
        """Menyimpan berkas dan nama kolom tanggal serta nilai kurs."""
        self.berkas, self.kolomTanggal, self.kolomNilai = berkas, kolomTanggal, kolomNilai

    @staticmethod
    def baca(berkas) -> pd.DataFrame:
        """Membaca berkas CSV atau Excel menjadi DataFrame."""
        nama = getattr(berkas, "name", "").lower()
        berkas.seek(0)
        if nama.endswith((".xlsx", ".xls")):
            return pd.read_excel(berkas)
        try:
            return pd.read_csv(berkas, sep=None, engine="python")
        except Exception:
            berkas.seek(0)
            return pd.read_csv(berkas)

    @staticmethod
    def tebakKolom(df: pd.DataFrame):
        """Menebak kolom tanggal dan kolom nilai kurs dari nama kolom."""
        tgl = next((c for c in df.columns if "tanggal" in str(c).lower() or "date" in str(c).lower()), df.columns[0])
        nilai = next((c for c in df.columns if c != tgl and ("kurs" in str(c).lower() or "close" in str(c).lower()
                                                            or "jisdor" in str(c).lower() or "nilai" in str(c).lower())),
                     next((c for c in df.columns if c != tgl), df.columns[-1]))
        return tgl, nilai

    def requestDataKurs(self, periode) -> pd.DataFrame:
        """Membaca berkas, menyaring periode, dan mengembalikan kolom tanggal dan nilaiKurs."""
        df = self.baca(self.berkas)
        out = pd.DataFrame({
            "tanggal": pd.to_datetime(df[self.kolomTanggal], errors="coerce", dayfirst=True),
            "nilaiKurs": _ke_angka(df[self.kolomNilai]),
        }).dropna()
        mulai, akhir = pd.Timestamp(periode[0]), pd.Timestamp(periode[1])
        out = out[(out["tanggal"] >= mulai) & (out["tanggal"] <= akhir)]
        if out.empty:
            raise ValueError("Tidak ada data kurs pada periode yang dipilih di berkas ini.")
        return out
