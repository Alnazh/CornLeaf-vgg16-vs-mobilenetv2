"""
data_loader.py

Modul untuk memuat dataset New Plant Diseases (Augmented) dari Kaggle
ke dalam format tf.data yang siap dipakai untuk training model VGG16
dan MobileNetV2.

Dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Catatan struktur folder:
Setelah diunduh dan diekstrak, struktur folder biasanya ter-nested dua kali
akibat proses zip di Kaggle, contoh:

New Plant Diseases Dataset(Augmented)/
    New Plant Diseases Dataset(Augmented)/
        train/
            Apple___Apple_scab/
            Apple___Black_rot/
            ...
        valid/
            Apple___Apple_scab/
            ...

Sesuaikan DATA_DIR di bawah dengan path hasil ekstraksi di komputer kamu.
"""

import tensorflow as tf

# Ubah path ini sesuai lokasi dataset di komputer/environment kamu
DATA_DIR = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = f"{DATA_DIR}/train"
VALID_DIR = f"{DATA_DIR}/valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Mengembalikan train_ds, val_ds, dan daftar nama kelas.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Optimasi pipeline
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    train_ds, val_ds, class_names = get_datasets()
    print(f"Jumlah kelas: {len(class_names)}")
    print(f"Contoh kelas: {class_names[:5]}")
