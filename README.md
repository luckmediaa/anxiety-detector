# LuckMedia Project: RnD Anxiety Face Detector

Sistem **Computer Vision** berbasis *Deep Learning* untuk mendeteksi wajah serta mengklasifikasikan emosi dan jenis kelamin (gender) secara *real-time* melalui *webcam*. Proyek ini mencakup dua modul utama: *inference pipeline* untuk prediksi langsung dan *data pipeline* untuk memuat serta memproses dataset pelatihan.

## 🚀 Fitur Utama
- **Real-Time Detection**: Mendeteksi multi-wajah dalam satu frame secara instan.
- **Dual Classification**: Memprediksi 7 kelas emosi dan 2 kelas gender secara serentak.
- **Prediction Smoothing**: Menggunakan metode statistik untuk menghilangkan *flickering* (kedipan label teks) pada aliran video langsung.
- **Robust Data Loader**: Pipeline pemrosesan dataset yang menangani pemfilteran wajah kotor, *one-hot encoding*, dan pembagian set validasi.

---

## 🧠 Algoritma

Proyek ini menggabungkan teknik Computer Vision klasik dan arsitektur Modern Deep Learning:

### 1. Deteksi Wajah: Algoritma Viola-Jones (Haar Cascade)
Menggunakan fitur *Haar-like* untuk memindai kontras gelap-terang pada gambar. Sistem menggunakan pendekatan *Sliding Window* dan *Image Pyramid* untuk mendeteksi wajah di berbagai skala (jarak).
*   **Scale Factor (1.3):** Gambar diperkecil 30% pada setiap tahap pemindaian.
*   **Min Neighbors (5):** Membutuhkan minimal 5 deteksi tumpang tindih untuk mengesahkan *bounding box* wajah.

### 2. Klasifikasi Gambar: Convolutional Neural Network (CNN)
Sistem memotong area wajah yang terdeteksi, menerapkan *preprocessing* (normalisasi nilai piksel dan penyesuaian dimensi), lalu memasukannya ke dalam model CNN yang sudah dilatih:
*   **Emosi (mini-XCEPTION):** Menggunakan arsitektur *Depthwise Separable Convolution* untuk menekan jumlah parameter sehingga model sangat ringan dan cepat di- *render* secara *real-time*.
*   **Gender (Simple CNN):** Arsitektur CNN standar berlapis untuk mengekstraksi fitur visual pembeda gender.

---

## 🧮 Rumus & Perhitungan Matematis

Di dalam kode program, terdapat beberapa fungsi matematis dan logika yang berperan penting:

### 1. Kelas Prediksi Maksimum (Argmax)
Output dari lapisan akhir model (Softmax) berupa probabilitas untuk setiap kelas. Algoritma mengambil kelas dengan probabilitas tertinggi.

$$ \hat{y} = \text{argmax}_c P(y=c | \mathbf{x}) $$

*Di mana y&#771; adalah label final, dan c adalah indeks kelas.*

### 2. Penstabilan Prediksi (Mode / Modus Smoothing)
Mata manusia rentan melihat teks yang berubah sangat cepat per milidetik. Sistem menyimpan antrean hasil prediksi dalam *sliding window* berukuran 10 *frame* terakhir ($W=10$), lalu mencari nilai Modus (nilai yang paling sering muncul).

$$ y_{display} = \text{Mo}(y_{t-9}, y_{t-8}, \dots, y_t) $$

### 3. Pembagian Data Latih dan Validasi
Pada skrip pemroses dataset, jumlah sampel dibagi menggunakan perhitungan proporsional:

N<sub>train</sub> = &lfloor; N<sub>total</sub> &times; (1 - validation_split) &rfloor;

---

## 📚 Dataset yang Digunakan

Skrip `DataManager` secara native mendukung pemrosesan untuk tiga dataset publik berstandar riset berikut:

1.  **FER2013 (Facial Expression Recognition 2013)**
    *   **Tujuan:** Klasifikasi Emosi.
    *   **Deskripsi:** Terdiri dari puluhan ribu gambar wajah resolusi 48x48 piksel grayscale. Terdapat 7 kelas label: *Angry, Disgust, Anxiety, Happy, Sad, Surprise, Neutral*.
2.  **IMDB-WIKI Crop**
    *   **Tujuan:** Klasifikasi Gender.
    *   **Deskripsi:** Dataset skala besar berisi gambar wajah selebritas berserta metadatanya. Program ini menggunakan logika penyaringan (*Boolean Masking*) otomatis untuk membuang gambar tanpa wajah (`face_score < 3`) atau gambar foto grup (`second_face_score`).
3.  **KDEF (Karolinska Directed Emotional Faces)**
    *   **Tujuan:** Klasifikasi Emosi.
    *   **Deskripsi:** Kumpulan gambar ekspresi emosi manusia yang diambil dari berbagai sudut, berguna untuk meningkatkan keandalan deteksi wajah dari samping (*profile view*).

---

## 🛠️ Instalasi & Persiapan

1. Pastikan Anda telah menginstal **Python 3.7+**.
2. Kloning repositori ini ke dalam direktori lokal Anda.
3. Instal semua dependensi yang diperlukan dengan perintah berikut:

```bash
pip install -r requirements.txt