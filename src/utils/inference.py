import cv2
import matplotlib.pyplot as plt
import numpy as np
# Keras image preprocessing digunakan untuk memuat gambar dari disk dengan mudah
from keras.preprocessing import image


def load_image(image_path, grayscale=False, target_size=None):
    """
    Fungsi untuk memuat gambar dari penyimpanan lokal ke dalam format NumPy array.
    """
    # Menentukan mode warna berdasarkan argumen boolean grayscale
    color_mode = "grayscale" if grayscale else "rgb"

    # Memuat gambar menggunakan modul Keras. 
    # Fungsi ini juga otomatis menangani proses 'resize' jika target_size diberikan
    pil_image = image.load_img(
        image_path, color_mode=color_mode, target_size=target_size
    )
    
    # Mengonversi format gambar PIL (Python Imaging Library) menjadi NumPy array
    return image.img_to_array(pil_image)


def load_detection_model(model_path):
    """
    Memuat model deteksi objek Haar Cascade berbasis file XML.
    """
    # cv2.CascadeClassifier adalah class OpenCV untuk algoritma Viola-Jones
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model


def detect_faces(detection_model, gray_image_array):
    """
    Mengeksekusi model Haar Cascade untuk mencari wajah pada gambar abu-abu.
    """
    # Parameter 1.3: scaleFactor (memperkecil gambar 30% per iterasi untuk deteksi multi-skala)
    # Parameter 5: minNeighbors (butuh min 5 deteksi berdekatan untuk mengesahkan sebuah wajah)
    # Mengembalikan array berupa daftar koordinat wajah: [[x, y, width, height], ...]
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)


def draw_bounding_box(face_coordinates, image_array, color):
    """
    Menggambar kotak persegi panjang pada gambar asli.
    """
    # Unpacking tuple/list koordinat: x (sumbu horizontal), y (vertikal), w (lebar), h (tinggi)
    x, y, w, h = face_coordinates
    
    # cv2.rectangle membutuhkan titik kiri-atas (x, y) dan titik kanan-bawah (x+w, y+h)
    # Angka 2 di akhir adalah ketebalan garis (thickness)
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    """
    Memperluas area kotak (bounding box) wajah untuk memberikan margin (padding).
    Ini penting agar seluruh kepala (seperti rambut dan dagu) ikut masuk saat di-crop.
    """
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    
    # Titik awal (x, y) digeser lebih jauh ke kiri dan atas
    # Titik akhir ditambahkan lebar/tinggi asli, lalu ditambah offset lagi ke kanan/bawah
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def draw_text(
    coordinates,
    image_array,
    text,
    color,
    x_offset=0,
    y_offset=0,
    font_scale=2,
    thickness=2,
):
    """
    Fungsi pembantu untuk menulis teks di atas gambar.
    """
    # Mengambil koordinat x dan y saja dari array menggunakan slicing [:2]
    x, y = coordinates[:2]
    
    # Menempatkan teks menggunakan OpenCV
    cv2.putText(
        image_array,
        text,
        (x + x_offset, y + y_offset), # Posisi teks dengan tambahan penyesuaian (offset)
        cv2.FONT_HERSHEY_SIMPLEX,     # Jenis font (San-serif standar OpenCV)
        font_scale,                   # Ukuran font
        color,                        # Warna font (Tuple BGR/RGB)
        thickness,                    # Ketebalan teks
        cv2.LINE_AA,                  # Tipe garis Anti-Aliased agar teks terlihat lebih halus/tidak bergerigi
    )


def get_colors(num_classes):
    """
    Menghasilkan daftar warna unik secara otomatis berdasarkan jumlah kelas/kategori.
    Sangat berguna jika Anda memiliki banyak label (misal: 10 emosi) tanpa harus menentukan warna manual.
    """
    # np.linspace(0, 1, num_classes) membagi jarak 0 hingga 1 sama rata.
    # plt.cm.hsv memetakan nilai tersebut ke roda warna HSV (Hue, Saturation, Value).
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    
    # Warna matplotlib berada di rentang 0.0 - 1.0 (float).
    # Gambar digital (OpenCV) menggunakan format 8-bit berentang 0 - 255.
    # Sehingga array dikalikan 255 untuk mengonversinya ke format warna standar.
    colors = np.asarray(colors) * 255
    return colors