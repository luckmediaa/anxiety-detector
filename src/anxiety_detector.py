# Mengimpor fungsi 'mode' untuk mencari nilai yang paling sering muncul dari sekumpulan data
from statistics import mode

# cv2 adalah OpenCV, library utama untuk manipulasi gambar dan computer vision
import cv2 
# load_model digunakan untuk memuat arsitektur dan bobot model Deep Learning (Keras)
from keras.models import load_model 
# NumPy digunakan untuk operasi matriks dan array numerik berkinerja tinggi
import numpy as np 

# Mengimpor modul kustom (utils). Base Kode agar kode utama tetap bersih
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# ==========================================
# 1. PARAMETER INISIALISASI
# ==========================================
# Path (lokasi) file model yang sudah dilatih (pre-trained models)
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'

# Memuat label teks (misal: 'Senang', 'Sedih', 'Pria', 'Wanita') berdasarkan dataset pelatihannya
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX # Jenis font yang akan digambar pada layar

# ==========================================
# 2. HYPER-PARAMETERS
# ==========================================
# frame_window: Jumlah frame sebelumnya yang diingat untuk menstabilkan prediksi (mengurangi flickering/kedip)
frame_window = 10 
# Offsets: Margin tambahan (padding) di sekitar kotak wajah sebelum dipotong untuk dimasukkan ke model CNN
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# ==========================================
# 3. MEMUAT MODEL (LOADING MODELS)
# ==========================================
# Memuat model Haar Cascade untuk mendeteksi lokasi wajah
face_detection = load_detection_model(detection_model_path)
# Memuat model CNN untuk klasifikasi. compile=False karena kita hanya melakukan inferensi (prediksi), bukan training
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# Mengambil ukuran input yang dibutuhkan oleh masing-masing model CNN (misal: 48x48 pixel atau 64x64 pixel)
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# ==========================================
# 4. INISIALISASI LIST UNTUK SMOOTHING (MODUS)
# ==========================================
# List ini berfungsi sebagai "antrean" yang menyimpan prediksi frame sebelumnya
gender_window = []
emotion_window = []

# ==========================================
# 5. MEMULAI VIDEO STREAMING (MAIN LOOP)
# ==========================================
cv2.namedWindow('window_frame') # Membuat jendela GUI aplikasi
video_capture = cv2.VideoCapture(0) # Mengakses webcam default komputer (index 0)

while True: # Looping tak terbatas selama aplikasi berjalan
    # Membaca frame dari webcam. bgr_image adalah gambar dalam format Blue-Green-Red (standar OpenCV)
    bgr_image = video_capture.read()[1]
    
    # Konversi warna gambar karena model yang berbeda membutuhkan format warna yang berbeda
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) # Grayscale untuk deteksi wajah dan emosi
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)   # RGB untuk klasifikasi gender

    # Menjalankan Haar Cascade untuk mendapatkan koordinat wajah di dalam gambar
    faces = detect_faces(face_detection, gray_image)

    # Looping untuk setiap wajah yang terdeteksi di dalam frame
    for face_coordinates in faces:

        # --- PRE-PROCESSING UNTUK PREDIKSI GENDER ---
        # Menambahkan margin pada koordinat wajah, lalu memotong gambar (cropping)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        # --- PRE-PROCESSING UNTUK PREDIKSI EMOSI ---
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        
        # Mengecilkan/membesarkan potongan wajah agar ukurannya pas dengan input layer CNN
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue # Jika terjadi error (misal wajah terlalu di pinggir dan ukurannya 0), lewati wajah ini

        # Standarisasi nilai pixel (misal: membagi dengan 255)
        gray_face = preprocess_input(gray_face, False)
        # Menambahkan dimensi batch di awal: dari (48, 48) menjadi (1, 48, 48) -> syarat Keras
        gray_face = np.expand_dims(gray_face, 0)
        # Menambahkan dimensi channel di akhir: dari (1, 48, 48) menjadi (1, 48, 48, 1) -> gambar grayscale (1 channel)
        gray_face = np.expand_dims(gray_face, -1)
        
        # --- PREDIKSI EMOSI ---
        # Memprediksi probabilitas emosi, lalu argmax mengambil indeks dengan skor tertinggi
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg] # Menerjemahkan indeks ke teks (misal: 'Marah')
        emotion_window.append(emotion_text) # Menyimpan hasil ke dalam list window

        # --- PREDIKSI GENDER ---
        # Setup dimensi dan pre-processing mirip seperti emosi, tapi untuk gambar RGB
        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)

        # --- SMOOTHING / PENSTABILAN PREDIKSI ---
        # Jika panjang list melebih frame_window (10), hapus data yang paling lama (index 0)
        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        
        try:
            # Mengambil nilai yang paling sering muncul dari 10 frame terakhir
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue # Lewati jika terjadi error (misal window kosong saat inisialisasi)

        # --- VISUALISASI ---
        # Mengatur warna berdasarkan gender. Index 0 adalah 'Wanita', Index 1 'Laki-laki'
        # Warna OpenCV menggunakan BGR, tetapi gambar ini sedang dikonversi ke RGB di baris 63,
        # Jadi (0, 0, 255) di RGB adalah warna Biru, (255, 0, 0) adalah Merah.
        if gender_text == gender_labels[0]:
            color = (0, 0, 255) 
        else:
            color = (255, 0, 0)

        # Menggambar kotak di wajah dan menempelkan teks hasil prediksi
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    # Mengembalikan gambar ke format BGR agar bisa dirender dengan benar oleh OpenCV imshow
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image) # Menampilkan frame ke layar
    
    # Deteksi input keyboard. Jika tombol 'q' ditekan, hentikan looping
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan dan melepaskan resource memori (hardware webcam dan GUI) saat aplikasi ditutup
video_capture.release()
cv2.destroyAllWindows()