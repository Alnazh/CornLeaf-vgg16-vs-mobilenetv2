import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import get_datasets

MODEL_NAMES = ["vgg16", "mobilenetv2"]
MODEL_TITLES = {"vgg16": "VGG16", "mobilenetv2": "MobileNetV2"}
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def save_json(data, filename):
    with open(f"{OUTPUTS_DIR}/{filename}", "w") as f:
        json.dump(data, f, indent=2)


def plot_and_export_curves(model_name):
    # Baca history dari CSV hasil training, lalu simpan versi PNG dan JSON
    history = pd.read_csv(f"{MODELS_DIR}/{model_name}_history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["accuracy"], label="Train Accuracy")
    axes[0].plot(history["epoch"], history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{MODEL_TITLES[model_name]} - Akurasi")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["loss"], label="Train Loss")
    axes[1].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{MODEL_TITLES[model_name]} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/{model_name}_curves.png", dpi=150)
    plt.close()

    save_json({
        "epoch": history["epoch"].tolist(),
        "accuracy": history["accuracy"].round(4).tolist(),
        "val_accuracy": history["val_accuracy"].round(4).tolist(),
        "loss": history["loss"].round(4).tolist(),
        "val_loss": history["val_loss"].round(4).tolist(),
    }, f"{model_name}_history.json")

    print(f"Kurva dan history {model_name} tersimpan di {OUTPUTS_DIR}/")


def evaluate_model(model_name, val_ds, class_names):
    # Prediksi ke seluruh data validasi, hanya inferensi, bukan training ulang
    model = tf.keras.models.load_model(f"{MODELS_DIR}/{model_name}_best.keras")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    save_json({"labels": class_names, "matrix": cm.tolist()}, f"{model_name}_confusion_matrix.json")
    plot_confusion_matrix_png(cm, class_names, model_name)

    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    rows = [
        {"label": name, "precision": round(report[name]["precision"], 4),
         "recall": round(report[name]["recall"], 4), "f1_score": round(report[name]["f1-score"], 4),
         "support": int(report[name]["support"])}
        for name in class_names
    ]
    save_json({
        "rows": rows,
        "accuracy": round(report["accuracy"], 4),
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall": round(report["macro avg"]["recall"], 4),
            "f1_score": round(report["macro avg"]["f1-score"], 4),
        },
    }, f"{model_name}_classification_report.json")

    save_json({
        "accuracy": round(report["accuracy"], 4),
        "num_classes": len(class_names),
        "num_val_images": len(y_true),
    }, f"{model_name}_metrics.json")

    print(f"Evaluasi {model_name} selesai, akurasi validasi {report['accuracy']:.4f}")


def plot_confusion_matrix_png(cm, class_names, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label Asli")
    ax.set_title(f"Confusion Matrix - {MODEL_TITLES[model_name]}")

    threshold = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=text_color)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/{model_name}_confusion_matrix.png", dpi=150)
    plt.close()


def main():
    _, val_ds, class_names = get_datasets()

    for model_name in MODEL_NAMES:
        plot_and_export_curves(model_name)
        evaluate_model(model_name, val_ds, class_names)


if __name__ == "__main__":
    main()
