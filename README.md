# Analisis Perbandingan Kinerja Arsitektur VGG16 dan MobileNetV2 untuk Klasifikasi Penyakit Daun Jagung

## Deskripsi Proyek

Proyek ini merupakan capstone project mata kuliah Kecerdasan Buatan dengan fokus studi Computer Vision. Penelitian ini membandingkan performa dua arsitektur Convolutional Neural Network, yaitu VGG16 dan MobileNetV2, dalam tugas klasifikasi penyakit pada daun tanaman jagung.

VGG16 dikenal sebagai arsitektur yang dalam dengan jumlah parameter besar sehingga cenderung menghasilkan akurasi tinggi namun berat secara komputasi. MobileNetV2 dirancang dengan pendekatan depthwise separable convolution sehingga lebih ringan dan efisien, cocok untuk perangkat dengan sumber daya terbatas seperti smartphone petani.

Selain kode training dan evaluasi, proyek ini juga dilengkapi aplikasi web berbasis Flask untuk mendemonstrasikan hasil kedua model secara langsung.

## Rumusan Masalah

1. Bagaimana perbandingan akurasi klasifikasi penyakit daun jagung antara arsitektur VGG16 dan MobileNetV2?
2. Bagaimana perbandingan efisiensi komputasi antara kedua arsitektur tersebut?
3. Arsitektur mana yang lebih sesuai untuk implementasi pada perangkat dengan sumber daya terbatas?

## Dataset

Corn or Maize Leaf Disease Dataset, sumber: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

Terdiri dari 4 kelas (Blight, Common Rust, Gray Leaf Spot, Healthy) dengan total 4.188 gambar. Letakkan folder kelas langsung di dalam `dataset/`.

## Struktur Repository

```
.
├── app.py                      # backend Flask, aplikasi web
├── data_loader.py              # pemuatan dataset ke format tf.data
├── train_vgg16.py              # training model VGG16
├── train_mobilenetv2.py        # training model MobileNetV2
├── evaluate.py                 # generate kurva, confusion matrix, dan JSON evaluasi
├── templates/                  # halaman HTML aplikasi web
│   ├── base.html
│   ├── partials/
│   ├── index.html
│   ├── prediksi.html
│   ├── evaluasi.html
│   ├── dataset.html
│   └── tentang.html
├── static/
│   ├── css/style.css
│   └── js/
├── dataset/                     # dataset (diabaikan git, lihat .gitignore)
├── models/                      # model .keras dan history training
├── outputs/                     # kurva, confusion matrix, dan JSON hasil evaluasi
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```
2. Unduh dataset dari tautan Kaggle, letakkan di `dataset/`
3. Training kedua model:
   ```
   python train_vgg16.py
   python train_mobilenetv2.py
   ```
4. Generate evaluasi (kurva, confusion matrix, JSON):
   ```
   python evaluate.py
   ```
5. Jalankan aplikasi web:
   ```
   python app.py
   ```
   Buka `http://127.0.0.1:5000`

## Metrik Evaluasi

Akurasi, precision, recall, dan F1-score per kelas, ditampilkan dalam bentuk confusion matrix dan classification report pada halaman Evaluasi di aplikasi web.

## Referensi

1. Corn or Maize Leaf Disease Dataset, Kaggle, smaranjitghose, dikurasi dari PlantVillage dan PlantDoc.
2. Simonyan, K. and Zisserman, A., Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16), 2014.
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L., MobileNetV2, Inverted Residuals and Linear Bottlenecks, 2018.
