# 😃😠😢 Deteksi Emosi Wajah Real-Time dengan RetinaFace & ONNX 🚀

Sistem deteksi emosi wajah secara _real-time_ yang canggih! Aplikasi ini menggunakan model deteksi wajah **RetinaFace (TensorFlow)** dan model klasifikasi emosi berbasis **ONNX** untuk menganalisis ekspresi langsung dari webcam Anda.

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow](https://img.shields.io/badge/AI-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![Model Format: ONNX](https://img.shields.io/badge/AI-ONNX-brightgreen.svg)](https://onnx.ai/)
[![License: MIT ](https://img.shields.io/badge/License-MIT-yellow.svg)](#lisensi) ---

## ✨ Fitur Utama

* 👁️ **Deteksi Wajah Akurat**: Menggunakan RetinaFace (dari TensorFlow SavedModel) untuk performa deteksi wajah yang cepat dan presisi.
* 😊 **Klasifikasi Emosi Multi-Kelas**: Model ONNX mampu mengenali 7 kelas emosi dasar (lihat [Struktur Emosi](#-struktur-emosi)).
* ⏱️ **Proses Real-Time**: Analisis langsung dari input webcam, lengkap dengan _bounding box_ di sekitar wajah dan label emosi yang terdeteksi.
* 💻 **Penggunaan Mudah**: Cukup satu perintah untuk menjalankan seluruh aplikasi dan melihat hasilnya.
* 💡 **Fleksibel**: Opsi untuk menyesuaikan path model jika diperlukan.

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman**: Python
* **Deteksi Wajah**: RetinaFace (implementasi TensorFlow)
* **Klasifikasi Emosi**: Model ONNX (misalnya, berbasis MobileNetV3 atau arsitektur serupa)
* **Machine Learning Frameworks**: TensorFlow (untuk RetinaFace), ONNX Runtime (untuk model emosi)
* **Library Utama**: OpenCV, NumPy

---

## 📋 Persyaratan Sistem

* Python 3.8 atau versi lebih baru.
* Webcam yang berfungsi (internal maupun eksternal).
* Sistem operasi: Windows, Linux, atau macOS (dengan Python dan dependensi terinstal).

---

## 🚀 Instalasi Langkah-demi-Langkah

1.  **Clone Repositori Ini:**
    Buka terminal atau command prompt, lalu jalankan:
    ```sh
    git clone https://github.com/pangkywara/face-emotion.git
    cd face-emotion
    ```

2.  **Buat dan Aktifkan Virtual Environment** (Sangat Direkomendasikan):
    ```sh
    python -m venv env
    ```
    * Untuk Linux/macOS:
        ```sh
        source env/bin/activate
        ```
    * Untuk Windows:
        ```sh
        .\env\Scripts\activate
        ```

3.  **Install Semua Dependensi:**
    Pastikan virtual environment Anda aktif, lalu jalankan:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Siapkan Model AI:** 🧠
    > **PENTING:** Pastikan file dan folder model AI ditempatkan sesuai struktur berikut di dalam folder proyek Anda:
    >
    > * **Model Klasifikasi Emosi (ONNX):**
    >     File model ONNX harus diletakkan di:
    >     ```
    >     <nama-folder-proyek>/QCS/me_models/mobilenetv3_mixed_weighted_fp16.onnx
    >     ```
    >     Jika Anda menggunakan nama atau path file ONNX yang berbeda, Anda perlu menyesuaikannya melalui argumen `--onnx_model_path` saat menjalankan aplikasi.
    >
    > * **Model Deteksi Wajah RetinaFace (TensorFlow SavedModel):**
    >     Folder model RetinaFace (yang berisi file `saved_model.pb` dan folder `variables`) harus diletakkan di:
    >     ```
    >     <nama-folder-proyek>/dodosd/model/retinaface_mobilefacenet025_tf/
    >     ```
    >
    > *Donwload Model di sini --> https://drive.google.com/drive/folders/1sclDGNjEwFj04R0ytpjHq6KftaN9FU5m?usp=sharing*

---

## ▶️ Cara Menjalankan Aplikasi

Setelah semua instalasi dan penyiapan model selesai, jalankan aplikasi dari terminal (pastikan virtual environment aktif):

```sh
python realtime_emotion_onnx.py
