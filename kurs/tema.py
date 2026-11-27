"""Tema tampilan: CSS dan komponen kecil (topbar, kartu angka, badge, tabel berhalaman)."""
import datetime as dt
import html

import pandas as pd
import streamlit as st

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]

IKON = {  # ikon garis (gaya lucide), 24x24
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "kalender": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "tren": '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "jam": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "cek": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "aktivitas": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "lapisan": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "unduh": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp textarea, .stApp [data-baseweb] { font-family: 'Roboto', system-ui, 'Segoe UI', Arial, sans-serif; }
.stApp { background: #F4F7FE; color: #0F172A; }

/* ---------- kerangka ---------- */
header[data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1240px; }
h1, h2, h3 { color: #1E3A8A; letter-spacing: -0.01em; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #1E3A8A; border-right: 0; }
[data-testid="stSidebar"] * { color: #E6EDFF; }
[data-testid="stSidebarHeader"] { display: none; }
[data-testid="stSidebarContent"] { padding-top: 0; }
[data-testid="stSidebarNav"] { padding-top: 0; }
[data-testid="stSidebarNav"]::before {
  content: ""; display: block; height: 76px; margin: 0 0 6px 0;
  background: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='76'%3E%3Ccircle cx='40' cy='38' r='25' fill='%2316A34A'/%3E%3Ctext x='40' y='45' font-family='Arial' font-size='18' font-weight='700' fill='white' text-anchor='middle'%3EKL%3C/text%3E%3Ctext x='76' y='34' font-family='Arial' font-size='15' font-weight='700' fill='white'%3EPrediksi Kurs%3C/text%3E%3Ctext x='76' y='52' font-family='Arial' font-size='13' fill='%23B6C6F2'%3EMetode LSTM%3C/text%3E%3C/svg%3E") no-repeat left center;
}
[data-testid="stSidebarNavSeparator"] { display: none; }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] * {
  color: #8FA6E6 !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stNavSectionHeader"] { padding: 10px 10px 2px 14px; }
[data-testid="stSidebarNavLink"] { border-radius: 6px; margin: 1px 10px; padding: 8px 12px; }
[data-testid="stSidebarNavLink"] span { color: #E6EDFF; font-size: 14.5px; font-weight: 400; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: #2563EB; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #fff; font-weight: 500; }
.side-foot { color: #B6C6F2; font-size: 12px; line-height: 1.5; padding: 18px 20px; border-top: 1px solid rgba(255,255,255,.12); margin-top: 24px; }
.side-foot b { color: #fff; font-size: 13px; }

/* ---------- topbar ---------- */
.topbar { display: flex; justify-content: space-between; align-items: center; background: #fff;
  margin: 0 -2rem 1.4rem -2rem; padding: 12px 2rem; border-bottom: 1px solid #E2E8F0; }
.crumb { font-size: 12px; color: #64748B; }
.tb-title { font-size: 19px; font-weight: 700; color: #1E3A8A; line-height: 1.25; }
.tb-right { display: flex; align-items: center; gap: 18px; }
.tb-date { font-size: 12.5px; color: #64748B; }
.tb-user { display: flex; align-items: center; gap: 10px; }
.tb-user .av { width: 34px; height: 34px; border-radius: 50%; background: #16A34A; color: #fff; display: grid; place-items: center; font-size: 13px; font-weight: 700; }
.tb-user b { display: block; font-size: 13px; color: #0F172A; line-height: 1.2; }
.tb-user small { color: #64748B; font-size: 11.5px; }

/* ---------- kartu angka ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px; }
.kpi { background: #fff; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 18px; }
.kpi-h { display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 500; color: #1E293B; }
.kpi .ik { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; }
.kpi .ik svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.kpi-v { font-size: 26px; font-weight: 700; color: #0F172A; margin-top: 10px; line-height: 1.15; }
.kpi-v.kecil { font-size: 17px; padding-top: 6px; line-height: 1.3; }
.kpi-c { font-size: 12.5px; color: #64748B; margin-top: 2px; }
.hijau { background: #DCFCE7; color: #15803D; } .oranye { background: #FFEDD5; color: #C2410C; }
.biru { background: #DBEAFE; color: #1D4ED8; } .teal { background: #CCFBF1; color: #0F766E; }
.merah { background: #FEE2E2; color: #B91C1C; } .kuning { background: #FEF9C3; color: #A16207; }
.abu { background: #F1F5F9; color: #475569; }

/* ---------- kartu isi ---------- */
div[class*="st-key-kartu"], [data-testid="stForm"] { background: #fff; border: 1px solid #E2E8F0 !important; border-radius: 10px; padding: 16px 18px; box-shadow: none; }
[data-testid="stMarkdownContainer"] p.jdl { font-size: 15.5px; font-weight: 700; color: #1E3A8A; margin: 0; line-height: 1.3; }
[data-testid="stMarkdownContainer"] p.sub { font-size: 12.5px; color: #64748B; margin: 2px 0 12px 0; line-height: 1.4; }

/* ---------- badge ---------- */
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }

/* ---------- tabel ---------- */
.tbl-wrap { border: 1px solid #E2E8F0; border-radius: 8px; overflow: hidden; background: #fff; }
.tbl-scroll { max-height: 440px; overflow-y: auto; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.tbl th { position: sticky; top: 0; background: #E8F0FE; color: #1E3A8A; font-weight: 500; text-align: left; padding: 10px 12px; border-bottom: 1px solid #C7D7F5; }
table.tbl th, table.tbl td { border-left: 0 !important; border-right: 0 !important; }
table.tbl td { padding: 9px 12px; border-bottom: 1px solid #EEF2F7; color: #0F172A; }
table.tbl tr:nth-child(even) td { background: #F7F9FC; }
table.tbl .r { text-align: right; font-variant-numeric: tabular-nums; }
.tbl-info { font-size: 12.5px; color: #64748B; padding-top: 8px; }

/* ---------- daftar status ---------- */
.stat-row { display: flex; align-items: center; justify-content: space-between; padding: 11px 0; border-bottom: 1px solid #EEF2F7; }
.stat-row:last-child { border-bottom: 0; }
.stat-row b { display: block; font-size: 14px; color: #0F172A; }
.stat-row small { color: #64748B; font-size: 12.5px; }

/* ---------- widget ---------- */
[data-testid="stBaseButton-primary"] { border-radius: 6px; font-weight: 700; border: 0; }
[data-testid="stBaseButton-secondary"] { border-radius: 6px; background: #fff; border: 1px solid #CBD5E1; color: #1E3A8A; }
[data-testid="stBaseButton-secondary"]:hover { border-color: #2563EB; color: #2563EB; }
[data-testid="stBaseButton-secondary"]:disabled { background: #F1F5F9; color: #94A3B8; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"], [data-testid="stDateInputField"] {
  border-radius: 6px !important; background: #fff !important; border: 1px solid #CBD5E1 !important; }
[data-baseweb="input"] input, [data-testid="stNumberInputContainer"] input { border: 0 !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stDateInputField"]:focus-within { border-color: #2563EB !important; }
[data-baseweb="base-input"] { background: #fff !important; border: 0 !important; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background: #fff; }
[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stDownloadButton"] button { border-radius: 6px; }
label[data-testid="stWidgetLabel"] p { font-size: 13px; color: #334155; font-weight: 500; }
"""


def pasang():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- format
def tanggal_id(tanggal: dt.date) -> str:
    return f"{HARI[tanggal.weekday()]}, {tanggal.day} {BULAN[tanggal.month - 1]} {tanggal.year}"


def angka(nilai, desimal: int = 0) -> str:
    """Format Indonesia: 17.802,50"""
    return f"{nilai:,.{desimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def sapaan() -> str:
    jam = dt.datetime.now().hour
    return "Selamat pagi" if jam < 11 else "Selamat siang" if jam < 15 else "Selamat sore" if jam < 18 else "Selamat malam"


class Aman(str):
    """Teks HTML yang sudah aman (tidak di-escape lagi)."""


def pill(teks: str, warna: str) -> Aman:
    return Aman(f'<span class="pill {warna}">{html.escape(teks)}</span>')


# ---------------------------------------------------------------- komponen
def topbar(bagian: str, judul: str):
    bc = f"{html.escape(bagian)} &rsaquo;" if bagian else "&nbsp;"
    st.markdown(
        f'<div class="topbar"><div><div class="crumb">{bc}</div><div class="tb-title">{html.escape(judul)}</div></div>'
        f'<div class="tb-right"><div class="tb-date">{tanggal_id(dt.date.today())}</div>'
        f'<div class="tb-user"><span class="av">PN</span><div><b>Peneliti</b><small>Pengguna</small></div></div></div></div>',
        unsafe_allow_html=True)


def judul_kartu(judul: str, sub: str = ""):
    st.markdown(f'<p class="jdl">{html.escape(judul)}</p>' + (f'<p class="sub">{html.escape(sub)}</p>' if sub else ""),
                unsafe_allow_html=True)


def kartu(label: str, nilai: str, ket: str, ikon: str, warna: str, kecil: bool = False) -> str:
    kls = "kpi-v kecil" if kecil else "kpi-v"
    return (f'<div class="kpi"><div class="kpi-h"><span class="ik {warna}"><svg viewBox="0 0 24 24">{IKON[ikon]}</svg></span>'
            f'<span>{html.escape(label)}</span></div><div class="{kls}">{nilai}</div><div class="kpi-c">{html.escape(ket)}</div></div>')


def baris_kartu(daftar: list):
    st.markdown('<div class="kpi-grid">' + "".join(daftar) + "</div>", unsafe_allow_html=True)


def _sel(v) -> str:
    return v if isinstance(v, Aman) else html.escape(str(v))


def tabel_html(kolom: list, baris: list, rata_kanan: tuple = ()) -> str:
    def kls(i):
        return ' class="r"' if i in rata_kanan else ""

    kepala = "".join("<th" + kls(i) + ">" + html.escape(k) + "</th>" for i, k in enumerate(kolom))
    isi = "".join("<tr>" + "".join("<td" + kls(i) + ">" + _sel(v) + "</td>" for i, v in enumerate(b)) + "</tr>"
                  for b in baris)
    return f'<div class="tbl-wrap"><div class="tbl-scroll"><table class="tbl"><thead><tr>{kepala}</tr></thead><tbody>{isi}</tbody></table></div></div>'


def tabel_sederhana(kolom: list, baris: list, rata_kanan: tuple = ()):
    st.markdown(tabel_html(kolom, baris, rata_kanan), unsafe_allow_html=True)


def _ubah_hal(kunci: str, arah: int):
    st.session_state[f"{kunci}_hal"] = max(1, st.session_state.get(f"{kunci}_hal", 1) + arah)


def _reset_hal(kunci: str):
    st.session_state[f"{kunci}_hal"] = 1


def tabel_berhalaman(kunci: str, kolom: list, baris: list, rata_kanan: tuple = ()):
    """Tabel dengan paginasi seperti di aplikasi referensi."""
    total = len(baris)
    per = int(st.session_state.get(f"{kunci}_baris", 15))
    maks = max(1, -(-total // per))
    hal = min(max(1, st.session_state.get(f"{kunci}_hal", 1)), maks)
    st.session_state[f"{kunci}_hal"] = hal
    awal, akhir = (hal - 1) * per, min(hal * per, total)
    st.markdown(tabel_html(kolom, baris[awal:akhir], rata_kanan), unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns([3.2, 1.0, 1.3, 1.3, 1.5], vertical_alignment="center")
    k1.markdown(f'<div class="tbl-info">Menampilkan {awal + 1 if total else 0}&ndash;{akhir} dari {total} data</div>', unsafe_allow_html=True)
    k2.selectbox("Baris", [10, 15, 25, 50], index=1, key=f"{kunci}_baris", label_visibility="collapsed",
                 on_change=_reset_hal, args=(kunci,))
    k3.button("‹ Sebelumnya", key=f"{kunci}_prev", disabled=hal <= 1, on_click=_ubah_hal, args=(kunci, -1))
    k4.markdown(f'<div class="tbl-info" style="text-align:center">Halaman {hal} / {maks}</div>', unsafe_allow_html=True)
    k5.button("Berikutnya ›", key=f"{kunci}_next", disabled=hal >= maks, on_click=_ubah_hal, args=(kunci, 1))
