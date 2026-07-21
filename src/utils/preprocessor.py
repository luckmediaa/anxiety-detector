import cv2
import numpy as np


def preprocess_input(x, v2=True):
    x = x.astype("float32")
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def _imread(image_name):
    # Membaca gambar menggunakan OpenCV
    img = cv2.imread(image_name)
    # Konversi dari BGR ke RGB agar sesuai dengan perilaku scipy lama
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _imresize(image_array, size):
    # scipy menggunakan format (height, width) -> size[0]=height, size[1]=width
    # cv2.resize membutuhkan format (width, height)
    return cv2.resize(image_array, (size[1], size[0]))


def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype="int")
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical