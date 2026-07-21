# Modul SciPy untuk membaca file dari MATLAB (.mat)
from scipy.io import loadmat
# Pandas untuk manipulasi data tabular (CSV)
import pandas as pd
# NumPy untuk operasi matriks matematika dan komputasi numerik
import numpy as np
# Untuk mengacak urutan data saat membagi dataset pelatihan
from random import shuffle
import os
import cv2

class DataManager(object):
    """
    Kelas (Object-Oriented) untuk mengelola dan memuat dataset 
    klasifikasi emosi (fer2013, KDEF) atau klasifikasi gender (imdb).
    """
    def __init__(self, dataset_name='imdb',
                 dataset_path=None, image_size=(48, 48)):
        
        # Inisialisasi konfigurasi dasar dataset
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        
        # Logika untuk menentukan lokasi (path) otomatis jika pengguna tidak memberikannya
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = '../datasets/KDEF/'
        else:
            # Error handling jika nama dataset tidak didukung
            raise Exception(
                    'Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        # Factory method pattern yang mengarahkan eksekusi ke fungsi spesifik sesuai dataset
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()
        return ground_truth_data

    def _load_imdb(self):
        # Skor minimal agar wajah dianggap valid (menyaring gambar blur/kotor)
        face_score_treshold = 3
        # Memuat file MATLAB
        dataset = loadmat(self.dataset_path)
        
        # Mengekstrak array langsung dari struktur data bawaan file .mat
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        
        # --- TEKNIK BOOLEAN MASKING ---
        # Membuat array True/False untuk memfilter data tanpa menggunakan for-loop
        # 1. Pastikan wajah utama terlihat cukup jelas (skor > 3)
        face_score_mask = face_score > face_score_treshold
        # 2. Pastikan tidak ada wajah kedua di gambar (menghindari foto grup)
        second_face_score_mask = np.isnan(second_face_score)
        # 3. Pastikan label gendernya valid (bukan NaN/Kosong)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        
        # Menggabungkan semua syarat dengan gerbang logika AND
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        
        # Menerapkan filter ke dalam array nama gambar dan label gender
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
            
        # Mengembalikan format kamus (Dictionary) dengan key=nama file, value=label gender
        return dict(zip(image_names, gender_classes))

    def _load_fer2013(self):
        # Membaca dataset CSV menggunakan Pandas
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist() # Kolom piksel masih berupa teks string panjang
        width, height = 48, 48
        faces = []
        
        # Mengubah data teks kembali menjadi bentuk gambar (array 2D)
        for pixel_sequence in pixels:
            # Memecah string berdasarkan spasi, mengubahnya menjadi integer
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            # Membentuk ulang (reshape) list 1D menjadi matriks 2D berukuran 48x48
            face = np.asarray(face).reshape(width, height)
            # Menyesuaikan resolusi (resize) sesuai permintaan inisialisasi class
            face = cv2.resize(face.astype('uint8'), self.image_size)
            # Konversi ke float32, tipe data standar yang diminta oleh arsitektur Deep Learning
            faces.append(face.astype('float32'))
            
        faces = np.asarray(faces)
        # Menambahkan dimensi channel: (Total_Data, 48, 48) -> (Total_Data, 48, 48, 1)
        faces = np.expand_dims(faces, -1)
        
        # --- ONE-HOT ENCODING ---
        # INFO: Jika Anda memakai Pandas modern, ganti .as_matrix() menjadi .values 
        # Mengubah 1 kolom kategori menjadi N kolom biner
        emotions = pd.get_dummies(data['emotion']).as_matrix() 
        return faces, emotions

    def _load_KDEF(self):
        # Memuat map konversi dari teks label ke angka indeks (misal 'AN' -> 0)
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        # Menelusuri seluruh direktori dan subdirektori untuk mencari file .jpg
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        
        # Inisialisasi matriks kosong (zero matrix) untuk kecepatan komputasi alokasi memori
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        
        for file_arg, file_path in enumerate(file_paths):
            # Membaca gambar langsung dalam mode Grayscale
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            
            # KDEF menyimpan kode emosi pada penamaan filenya di karakter indeks ke-4 dan 5
            # Contoh nama file: BF33AN.jpg -> Emosinya adalah 'AN' (Angry)
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6] 
            
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                # Mengabaikan file yang penamaannya tidak standar/error
                continue
                
            # Manual One-Hot Encoding: Memberi nilai 1 hanya pada indeks emosi yang sesuai
            emotions[file_arg, emotion_arg] = 1
            
        faces = np.expand_dims(faces, -1)
        return faces, emotions


# ==============================================================
# HELPER FUNCTIONS (FUNGSI PEMBANTU DI LUAR KELAS)
# ==============================================================

def get_labels(dataset_name):
    # Mengembalikan konversi label dari angka ke teks (biasanya digunakan untuk visualisasi)
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'anxiety', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')

def get_class_to_arg(dataset_name='fer2013'):
    # Kebalikan dari get_labels, mengonversi teks ke angka indeks 
    # (biasanya digunakan pada saat membaca data target sebelum training)
    if dataset_name == 'fer2013':
        return {'angry': 0, 'disgust': 1, 'anxiety': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    elif dataset_name == 'imdb':
        return {'woman': 0, 'man': 1}
    elif dataset_name == 'KDEF':
        return {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}
    else:
        raise Exception('Invalid dataset name')

def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    # Membagi dataset bentuk Dictionary (IMDB) menjadi subset Training dan Validation
    ground_truth_keys = sorted(ground_truth_data.keys())
    
    if do_shuffle is not False:
        shuffle(ground_truth_keys) # Pengacakan data agar distribusi fitur lebih seimbang
        
    training_split = 1 - validation_split
    # Mengalikan total data dengan rasio pelatihan (misal: 1000 * 0.8 = 800)
    num_train = int(training_split * len(ground_truth_keys))
    
    # Python slicing: Mengambil data indeks 0 sampai 800 untuk Train, sisanya untuk Validation
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def split_data(x, y, validation_split=.2):
    # Fungsi pembagi data general (Training/Validation) untuk struktur data array (FER2013/KDEF)
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    
    # Pemisahan untuk array X (Fitur/Gambar) dan Y (Label/Target)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data