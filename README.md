<div align="center">

# 🌽 CornLeaf Vision

**Perbandingan arsitektur VGG16 vs MobileNetV2 untuk klasifikasi penyakit daun jagung**

Capstone project mata kuliah Kecerdasan Buatan (fokus Computer Vision) yang membandingkan performa
dua arsitektur *Convolutional Neural Network* — **VGG16** dan **MobileNetV2** — dalam mengklasifikasikan
penyakit pada daun tanaman jagung, dibungkus dalam dashboard web Flask yang interaktif.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-VGG16%20%7C%20MobileNetV2-4CAF50)](#-arsitektur-model)
[![Accuracy](https://img.shields.io/badge/Best%20Val%20Accuracy-93.55%25-brightgreen)](#-performa-model)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#-lisensi--sumber-dataset)

</div>

---

## 📖 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Rumusan Masalah](#-rumusan-masalah)
- [Fitur Utama](#-fitur-utama)
- [Performa Model](#-performa-model)
- [Arsitektur Model](#-arsitektur-model)
- [Tumpukan Teknologi](#-tumpukan-teknologi)
- [Struktur Folder](#-struktur-folder)
- [Instalasi & Menjalankan Aplikasi](#-instalasi--menjalankan-aplikasi)
- [Alur Pipeline: Data → Training → Evaluasi](#-alur-pipeline-data--training--evaluasi)
- [Riwayat Commit](#-riwayat-commit)
- [Lisensi & Sumber Dataset](#-lisensi--sumber-dataset)
- [Referensi](#-referensi)

---

## 🎯 Tentang Proyek

**CornLeaf Vision** adalah proyek yang membandingkan dua arsitektur CNN populer untuk mendeteksi
penyakit pada daun jagung dari sebuah foto: **VGG16** dan **MobileNetV2**, keduanya dilatih dengan
pendekatan *transfer learning* dari bobot ImageNet.

VGG16 dikenal sebagai arsitektur yang dalam dengan jumlah parameter besar sehingga cenderung
menghasilkan akurasi tinggi namun berat secara komputasi. MobileNetV2 dirancang dengan pendekatan
*depthwise separable convolution* sehingga lebih ringan dan efisien, cocok untuk perangkat dengan
sumber daya terbatas seperti smartphone petani.

Selain kode training dan evaluasi, proyek ini juga dilengkapi aplikasi web berbasis **Flask** untuk
mendemonstrasikan dan membandingkan hasil kedua model secara langsung, lengkap dengan visualisasi
kurva pelatihan, confusion matrix, dan classification report.

---

## ❓ Rumusan Masalah

1. Bagaimana perbandingan akurasi klasifikasi penyakit daun jagung antara arsitektur VGG16 dan MobileNetV2?
2. Bagaimana perbandingan efisiensi komputasi antara kedua arsitektur tersebut?
3. Arsitektur mana yang lebih sesuai untuk implementasi pada perangkat dengan sumber daya terbatas?

---

## ✨ Fitur Utama

| Halaman | Deskripsi |
|---|---|
| 🏠 **Beranda** | Ringkasan statistik kedua model: akurasi, jumlah kelas, status model aktif |
| 🔍 **Prediksi Gambar** | Unggah foto daun jagung → klasifikasi jenis penyakit menggunakan model pilihan |
| 📊 **Evaluasi Model** | Grafik akurasi/loss per-epoch, confusion matrix, dan classification report untuk VGG16 & MobileNetV2 |
| 🗂️ **Dataset & Kelas** | Informasi dataset dan daftar 4 kelas penyakit daun jagung |
| ℹ️ **Tentang** | Penjelasan cara kerja sistem dan perbandingan kedua arsitektur |

### Detail fitur halaman Evaluasi

- 📈 Kurva akurasi & loss per epoch untuk masing-masing model
- 🧩 Confusion matrix dan classification report (precision, recall, F1-score) per kelas
- ⚖️ Perbandingan langsung performa VGG16 vs MobileNetV2 melalui endpoint `/api/comparison`

---

## 📈 Performa Model

Hasil evaluasi pada data validasi (bukan data simulasi):

| Model | Akurasi Validasi | Jumlah Kelas | Jumlah Gambar Validasi |
|---|---|---|---|
| **VGG16** | 92.59% | 4 | 837 |
| **MobileNetV2** | 93.55% | 4 | 837 |

> Detail lengkap kurva training, confusion matrix, dan classification report per kelas dapat dilihat
> langsung di halaman **Evaluasi Model** pada aplikasi web.

---

## 🧠 Arsitektur Model

Kedua model menggunakan pendekatan *transfer learning* (bobot awal dari ImageNet):

- **VGG16** - arsitektur CNN dalam (16 layer berbobot) dengan jumlah parameter besar, dikenal
  akurat namun berat secara komputasi.
- **MobileNetV2** - arsitektur ringan berbasis *inverted residuals* dan *depthwise separable
  convolution*, dirancang efisien untuk perangkat dengan sumber daya terbatas.

**Spesifikasi input/output:**
- Ukuran input gambar: `224 x 224` piksel (RGB)
- Pembagian data: 80% training / 20% validasi (otomatis, karena dataset tidak memiliki folder
  train/valid terpisah)
- Output: 4 kelas (Blight, Common Rust, Gray Leaf Spot, Healthy)

---

## 🛠️ Tumpukan Teknologi

| Kategori | Teknologi |
|---|---|
| Backend | Flask 3.0 |
| Model & Training | TensorFlow / Keras ≥ 2.15, VGG16, MobileNetV2 |
| Pengolahan Gambar | Pillow, OpenCV, NumPy |
| Evaluasi & Visualisasi | Pandas, Matplotlib, Seaborn, Scikit-learn |
| Frontend | Bootstrap, Bootstrap Icons |
| Eksperimen | Jupyter |

---

## 📁 Struktur Folder

```
app.py                        backend Flask, semua route halaman & API
data_loader.py                pemuatan dataset ke format tf.data (split 80:20)
train_vgg16.py                script training model VGG16
train_mobilenetv2.py          script training model MobileNetV2
evaluate.py                   script evaluasi (kurva, confusion matrix, classification report)
templates/                    halaman HTML (Jinja2)
├── base.html
├── partials/
├── index.html
├── prediksi.html
├── result.html
├── evaluasi.html
├── dataset.html
└── tentang.html
static/                       CSS, JS, dan aset tema
dataset/                      lokasi ekstraksi dataset Kaggle (diabaikan git, lihat .gitignore)
models/                       model .keras hasil training (diabaikan git)
outputs/                      kurva, confusion matrix, dan JSON hasil evaluasi kedua model
requirements.txt              daftar dependensi library
README.md
```

---

## 🚀 Instalasi & Menjalankan Aplikasi

### Prasyarat
- Python 3.10 atau lebih baru
- `pip`

### Langkah-langkah

1. **Clone / unduh repositori ini**, lalu masuk ke direktori proyek:
   ```bash
   git clone https://github.com/Alnazh/CornLeaf-vgg16-vs-mobilenetv2.git
   cd CornLeaf-vgg16-vs-mobilenetv2
   ```

2. **Buat virtual environment** (opsional tapi disarankan) dan aktifkan:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Instal seluruh dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan server**:
   ```bash
   python app.py
   ```

5. **Buka di browser**:
   ```
   http://127.0.0.1:5000
   ```

---

## 🔁 Alur Pipeline: Data → Training → Evaluasi

Proyek ini mencerminkan alur kerja *machine learning* lengkap, dari persiapan data hingga evaluasi:

1. **Persiapan data** - unduh dataset dari Kaggle:
   [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset),
   lalu letakkan folder kelas langsung di dalam `dataset/`. Pemuatan dan pembagian data (80% train /
   20% validasi) ditangani otomatis oleh `data_loader.py`.

2. **Training** - latih kedua arsitektur secara terpisah:
   ```bash
   python train_vgg16.py
   python train_mobilenetv2.py
   ```
   Model tersimpan otomatis di `models/`.

3. **Evaluasi** - hasilkan kurva akurasi/loss, confusion matrix, dan classification report untuk
   kedua model sekaligus:
   ```bash
   python evaluate.py
   ```
   Seluruh hasil (JSON dan gambar) tersimpan di `outputs/`, dan otomatis ditampilkan pada halaman
   **Evaluasi Model** di aplikasi web.

---

## 📜 Lisensi & Sumber Dataset

Dataset yang digunakan: **Corn or Maize Leaf Disease Dataset** oleh Smaranjit Ghose (Kaggle),
dikurasi dari PlantVillage dan PlantDoc, terdiri dari 4 kelas (Blight, Common Rust, Gray Leaf Spot,
Healthy) dengan total 4.188 gambar.

🔗 Sumber: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset
