# Prediksi Nilai Tukar Rupiah terhadap Dolar Menggunakan Metode LSTM

Website Streamlit yang mengikuti diagram UML Kelompok 10 (use case, activity, sequence, state machine, class, object).

## Anggota Tim

Kelompok 10, Rekayasa Perangkat Lunak:

- Fikry Maulana Achmad — feature/entity dan feature/antarmuka-ui
- Tri Anggoro — feature/controller
- Wildan Fadhil Nadhif — feature/tampilan dan feature/sumber-data

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501

## Cara pakai (urutan menu)

0. **Dashboard**: ringkasan keadaan sistem (hanya tampilan, tidak mengubah data, tidak ada di diagram).

1. **Ambil Data**: pilih Yahoo Finance (ticker `IDR=X`) atau unggah berkas CSV/Excel (misalnya JISDOR dari Bank Indonesia), pilih periode, klik *Ambil Data*.
2. **Data Historis**: lihat tabel dan grafik kurs.
3. **Hyperparameter** (opsional): ubah window, porsi data latih, neuron, dropout, epoch, batch size, learning rate. Nilai di luar batas ditolak dan nilai lama tetap dipakai.
4. **Latih Model**: klik *Latih model*. Urutannya: validasi, bagi data latih/uji (berurutan), preprocessing (scaler hanya dilatih pada data latih), pelatihan, evaluasi.
5. **Hasil Prediksi**: grafik aktual vs prediksi, ramalan 7 atau 30 hari kerja, unduh CSV atau PDF.

## Peta diagram ke kode

| Diagram | File | Isi |
|---|---|---|
| Class `<<boundary>>` | `kurs/ui.py` | `DataUI`, `HyperparameterUI`, `PelatihanUI`, `HasilPrediksiUI` (+ `DashboardUI`) |
| Tampilan | `kurs/tema.py`, `kurs/grafik.py`, `.streamlit/config.toml` | tema, komponen, gaya grafik |
| Class `<<control>>` | `kurs/controller.py` | `DataController`, `ModelController`, `PrediksiController` |
| Class `<<entity>>` | `kurs/entity.py` | `DataKurs`, `ModelLSTM`, `HasilPrediksi` |
| Class `<<external system>>` | `kurs/sumber.py` | `YahooFinance`, `UnggahFile` (pengganti `BankIndonesia`, method `requestDataKurs`) |

Nama method sama dengan diagram. Method `halaman...` di class UI menyusun urutan pemanggilan seperti di sequence diagram.

## Catatan

- Data, model, dan hasil disimpan di folder `storage/`. Di Streamlit Community Cloud folder ini tidak permanen.
- Data kurs hanya ada pada hari kerja, jadi "7 hari" dan "30 hari" dihitung sebagai hari kerja.
- Hasil pelatihan bisa sedikit berbeda antar komputer meskipun seed sudah dikunci.
- Yang ditambahkan di luar Class Diagram: hyperparameter `dropout` dan `rasioData` (di `ModelLSTM`), method `ramal` (di `ModelLSTM`), serta `UnggahFile` sebagai sumber data alternatif.

## Penerapan aturan Konstruksi PL (Pertemuan 10)

| Aturan pada materi | Penerapan di kode |
|---|---|
| Nama kelas pada diagram menjadi nama class | 12 class sama dengan Class Diagram (`DataUI`, `DataController`, `DataKurs`, dan seterusnya), PascalCase |
| Atribut menjadi field | Atribut tiap class dibuat sebagai field private (`_idKurs`, `_jumlahUnit`, `_rmse`, `_rasioData`, `rentangWaktu`, dan lainnya) |
| Metode menjadi method | Nama dan urutan method sama dengan diagram (camelCase) |
| Relasi menjadi referensi objek | `DataUI` memegang `DataController`, controller memegang entity (`self.dataKurs`, `self.modelLSTM`, `self.hasilPrediksi`) |
| Urutan pesan Sequence Diagram menjadi urutan pemanggilan | Method `halaman...` di class UI dan `latihModel` di `ModelController` (diberi komentar nomor pesan) |
| Wireframe/prototipe menjadi tata letak | `tema.py` dan `.streamlit/config.toml` mengikuti desain UI/UX |
| Penamaan deskriptif | Nama dari diagram memakai camelCase; fungsi bantu Python (`tema.py`, `grafik.py`) memakai snake_case |
| Komentar kode | Setiap class dan method punya docstring |
| Satu fungsi satu tugas | Pemrosesan dipecah: `bagiData`, `bersihkanData`, `fitScaler`, `transform`, `bentukWindow`, `hitungRMSE`, `hitungMAPE` |
| DRY | `_rmse` dan `_mape` dipakai untuk data latih dan uji; `tema.kartu`, `tema.tabel_sederhana`, dan `tema.tabel_berhalaman` dipakai ulang di semua halaman |
| KISS | Sumber data dibuat dua kelas sederhana (`YahooFinance`, `UnggahFile`) dengan satu method yang sama, `requestDataKurs` |

