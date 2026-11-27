"""Grafik matplotlib dengan gaya yang sama dengan tema aplikasi (tanpa Streamlit)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HIJAU, BIRU, ABU, TEKS = "#16A34A", "#2563EB", "#CBD5E1", "#334155"


def _gaya():
    """Menetapkan gaya dasar matplotlib."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "Liberation Sans", "Arial", "DejaVu Sans"],
        "axes.edgecolor": ABU, "axes.labelcolor": TEKS, "xtick.color": TEKS, "ytick.color": TEKS,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11.5, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11.5,
    })


def _rapikan(ax, judul=None):
    """Merapikan sumbu: garis bantu, format ribuan, dan judul."""
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.margins(x=0.01)
    if judul:
        ax.set_title(judul, loc="left", fontsize=12, fontweight="bold", color="#1E3A8A", pad=12)


def aktual_vs_prediksi(aktual, prediksi, depan=None):
    """Grafik garis aktual vs prediksi (dan ramalan ke depan bila ada)."""
    _gaya()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(aktual["tanggal"], aktual["nilaiAktual"], color=HIJAU, lw=2, label="Aktual")
    ax.plot(prediksi["tanggal"], prediksi["nilaiPrediksi"], color=BIRU, lw=2, label="Prediksi LSTM")
    if depan is not None and len(depan):
        ax.plot(depan["tanggal"], depan["nilaiPrediksi"], color=BIRU, lw=2, ls="--", label=f"Ramalan {len(depan)} hari")
    ax.set_ylabel("Kurs USD/IDR (Rp)")
    _rapikan(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=3, borderaxespad=0.3)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


def historis(data, judul=None):
    """Grafik garis data kurs historis."""
    _gaya()
    fig, ax = plt.subplots(figsize=(9.5, 3.7))
    ax.plot(data["tanggal"], data["nilaiKurs"], color=BIRU, lw=2)
    ax.fill_between(data["tanggal"], data["nilaiKurs"], data["nilaiKurs"].min(), color=BIRU, alpha=0.07)
    ax.set_ylabel("Kurs USD/IDR (Rp)")
    ax.set_ylim(data["nilaiKurs"].min() * 0.998, data["nilaiKurs"].max() * 1.002)
    _rapikan(ax, judul)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig
