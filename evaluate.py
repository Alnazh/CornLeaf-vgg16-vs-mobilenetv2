import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import get_datasets

MODEL_NAMES = ["vgg16", "mobilenetv2"]
MODELS_DIR = "models"


def plot_training_curves(model_name):
    # Baca history dari CSV hasil training sebelumnya, tidak perlu training ulang
    history = pd.read_csv(f"{MODELS_DIR}/{model_name}_history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["accuracy"], label="Train Accuracy")
    axes[0].plot(history["epoch"], history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{model_name.upper()} - Akurasi")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["loss"], label="Train Loss")
    axes[1].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name.upper()} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/{model_name}_curves.png", dpi=150)
    plt.close()
    print(f"Kurva training {model_name} tersimpan di {MODELS_DIR}/{model_name}_curves.png")


def plot_confusion_matrix(model_name, val_ds, class_names):
    # Load model hasil training sebelumnya, hanya untuk prediksi (bukan training ulang)
    model = tf.keras.models.load_model(f"{MODELS_DIR}/{model_name}_best.keras")

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label Asli")
    ax.set_title(f"Confusion Matrix - {model_name.upper()}")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/{model_name}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix {model_name} tersimpan di {MODELS_DIR}/{model_name}_confusion_matrix.png")

    print(f"\nClassification Report - {model_name.upper()}")
    print(classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0))


def main():
    _, val_ds, class_names = get_datasets()

    for model_name in MODEL_NAMES:
        plot_training_curves(model_name)
        plot_confusion_matrix(model_name, val_ds, class_names)


if __name__ == "__main__":
    main()
