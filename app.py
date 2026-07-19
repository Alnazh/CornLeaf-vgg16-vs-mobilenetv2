import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
IMG_SIZE = (224, 224)

CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_LABELS = {
    "Blight": "Blight (Hawar Daun)",
    "Common_Rust": "Common Rust (Karat Daun)",
    "Gray_Leaf_Spot": "Gray Leaf Spot (Bercak Daun Abu-abu)",
    "Healthy": "Healthy (Sehat)",
}
CLASS_DESCRIPTIONS = {
    "Blight": "Bercak memanjang berwarna coklat keabuan pada permukaan daun.",
    "Common_Rust": "Bintik kecil berwarna coklat kemerahan menyerupai karat besi.",
    "Gray_Leaf_Spot": "Bercak persegi panjang berwarna abu-abu kecoklatan sejajar tulang daun.",
    "Healthy": "Daun tanpa gejala penyakit, berwarna hijau merata.",
}
CLASS_COUNTS = {"Blight": 1146, "Common_Rust": 1306, "Gray_Leaf_Spot": 574, "Healthy": 1162}
CLASS_RECOMMENDATIONS = {
    "Blight": "Segera pangkas daun yang terinfeksi dan pertimbangkan fungisida berbahan aktif triazol. Perbaiki jarak tanam agar sirkulasi udara lebih baik.",
    "Common_Rust": "Pantau perkembangan bintik karat pada daun bagian bawah. Fungisida berbahan aktif strobilurin efektif jika diaplikasikan pada gejala awal.",
    "Gray_Leaf_Spot": "Rotasi tanam dengan tanaman non-inang dan bersihkan sisa tanaman terinfeksi dari lahan, karena jamur ini bertahan pada residu jagung musim sebelumnya.",
    "Healthy": "Daun tidak menunjukkan gejala penyakit. Tetap lakukan pemantauan rutin dan jaga kelembapan lahan agar tidak berlebihan.",
}
MODEL_DISPLAY_NAMES = {"vgg16": "VGG16", "mobilenetv2": "MobileNetV2"}

# Model dimuat sekali saat server dijalankan, bukan setiap request
vgg16_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "vgg16_best.keras"))
mobilenetv2_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "mobilenetv2_best.keras"))


def load_json_file(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_and_prepare_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(img)
    return np.expand_dims(array, axis=0)


def predict_with_model(model, preprocess_fn, raw_image):
    processed = preprocess_fn(raw_image.copy())
    probs = model.predict(processed, verbose=0)[0]
    predicted_index = int(np.argmax(probs))
    predicted_name = CLASS_NAMES[predicted_index]
    return {
        "predicted_class": CLASS_LABELS[predicted_name],
        "confidence": round(float(probs[predicted_index]) * 100, 2),
        "recommendation": CLASS_RECOMMENDATIONS[predicted_name],
        "probabilities": [
            {"label": CLASS_LABELS[name], "value": round(float(p) * 100, 2)}
            for name, p in zip(CLASS_NAMES, probs)
        ],
    }


def format_pct(value):
    return f"{value * 100:.2f}%"


def build_training_captions(vgg_history, mobile_history):
    # Menjelaskan tren akurasi dan loss dari data history, bukan teks statis
    if not vgg_history or not mobile_history:
        return None, None

    v_val_start, v_val_end = vgg_history["val_accuracy"][0], vgg_history["val_accuracy"][-1]
    m_val_start, m_val_end = mobile_history["val_accuracy"][0], mobile_history["val_accuracy"][-1]
    v_gap = vgg_history["accuracy"][-1] - v_val_end
    m_gap = mobile_history["accuracy"][-1] - m_val_end

    accuracy_caption = (
        f"Akurasi validasi VGG16 naik dari {format_pct(v_val_start)} di epoch pertama menjadi "
        f"{format_pct(v_val_end)} di epoch terakhir, sedangkan MobileNetV2 naik dari {format_pct(m_val_start)} "
        f"menjadi {format_pct(m_val_end)}. Selisih akurasi data latih dan data validasi di epoch terakhir hanya "
        f"{format_pct(v_gap)} untuk VGG16 dan {format_pct(m_gap)} untuk MobileNetV2, jadi kedua model belum "
        f"menunjukkan tanda menghafal data latih secara berlebihan (overfitting)."
    )

    v_loss_start, v_loss_end = vgg_history["val_loss"][0], vgg_history["val_loss"][-1]
    m_loss_start, m_loss_end = mobile_history["val_loss"][0], mobile_history["val_loss"][-1]
    loss_caption = (
        f"Loss validasi VGG16 turun dari {v_loss_start:.4f} menjadi {v_loss_end:.4f}, dan MobileNetV2 turun dari "
        f"{m_loss_start:.4f} menjadi {m_loss_end:.4f}. Kedua kurva melandai di epoch akhir, artinya proses "
        f"training sudah cukup stabil dan menambah epoch lagi kemungkinan tidak banyak menurunkan kesalahan model."
    )
    return accuracy_caption, loss_caption


def build_comparison_caption(vgg_report, mobile_report):
    # Membandingkan akurasi kedua model secara langsung dari classification report
    if not vgg_report or not mobile_report:
        return None

    diff = mobile_report["accuracy"] - vgg_report["accuracy"]
    if diff > 0:
        winner_note = f"MobileNetV2 unggul tipis sekitar {format_pct(abs(diff))}"
    elif diff < 0:
        winner_note = f"VGG16 unggul tipis sekitar {format_pct(abs(diff))}"
    else:
        winner_note = "kedua model mencatat akurasi yang sama persis"

    return (
        f"VGG16 mencapai akurasi {format_pct(vgg_report['accuracy'])} dan MobileNetV2 mencapai "
        f"{format_pct(mobile_report['accuracy'])} pada data validasi, {winner_note}. Meski begitu, MobileNetV2 "
        f"memakai jauh lebih sedikit parameter daripada VGG16, sehingga tetap lebih hemat untuk dijalankan di "
        f"perangkat dengan sumber daya terbatas seperti ponsel."
    )


def build_confusion_caption(model_name, confusion):
    # Mencari pasangan kelas yang paling sering tertukar dari matriks konfusi
    if not confusion:
        return None

    matrix = confusion["matrix"]
    labels = confusion["labels"]
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(labels)))

    worst_true, worst_pred, worst_count = 0, 0, -1
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and matrix[i][j] > worst_count:
                worst_true, worst_pred, worst_count = i, j, matrix[i][j]

    true_label = CLASS_LABELS[labels[worst_true]]
    pred_label = CLASS_LABELS[labels[worst_pred]]
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    return (
        f"Dari {total} gambar validasi, {display_name} menebak benar {correct} gambar "
        f"({format_pct(correct / total)}). Kesalahan paling sering terjadi ketika gambar berlabel {true_label} "
        f"justru diprediksi sebagai {pred_label}, sebanyak {worst_count} kali. Ini masuk akal karena gejala "
        f"kedua kondisi tersebut sama-sama berupa bercak pada permukaan daun, sehingga polanya mirip di mata model."
    )


def build_report_caption(model_name, report):
    # Menyoroti kelas terlemah dan mengaitkannya dengan jumlah data validasinya
    if not report:
        return None

    weakest = min(report["rows"], key=lambda row: row["f1_score"])
    strongest = max(report["rows"], key=lambda row: row["f1_score"])
    smallest_support = min(report["rows"], key=lambda row: row["support"])
    weak_label = CLASS_LABELS[weakest["label"]]
    strong_label = CLASS_LABELS[strongest["label"]]
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    support_note = ""
    if smallest_support["label"] == weakest["label"]:
        support_note = (
            f" {weak_label} juga memiliki jumlah data validasi paling sedikit di antara empat kelas "
            f"({weakest['support']} gambar), jadi wajar bila model masih agak kesulitan mengenalinya."
        )

    return (
        f"Kelas dengan F1-Score terendah pada {display_name} adalah {weak_label} "
        f"({weakest['f1_score']:.4f}), sedangkan yang tertinggi adalah {strong_label} "
        f"({strongest['f1_score']:.4f}).{support_note}"
    )


def build_distribution_caption(labels, counts):
    # Menjelaskan ketimpangan jumlah gambar antar kelas dari data distribusi
    total = sum(counts)
    max_index = counts.index(max(counts))
    min_index = counts.index(min(counts))
    return (
        f"Dataset berisi total {total} gambar yang tersebar ke empat kelas. Kelas {labels[max_index]} memiliki "
        f"gambar terbanyak ({counts[max_index]} gambar), sedangkan kelas {labels[min_index]} memiliki gambar "
        f"paling sedikit ({counts[min_index]} gambar). Ketimpangan jumlah data seperti ini bisa membuat model "
        f"lebih sulit mengenali kelas yang datanya lebih sedikit dibanding kelas lain."
    )


def build_architecture_caption(vgg16_metrics, mobilenetv2_metrics):
    # Menjelaskan trade-off ukuran model versus akurasi menggunakan angka evaluasi asli
    if not vgg16_metrics or not mobilenetv2_metrics:
        return None

    diff = mobilenetv2_metrics["accuracy"] - vgg16_metrics["accuracy"]
    if diff >= 0:
        accuracy_note = f"MobileNetV2 justru lebih akurat {format_pct(abs(diff))} meski parameternya jauh lebih sedikit"
    else:
        accuracy_note = f"VGG16 lebih akurat {format_pct(abs(diff))}, tetapi dengan parameter sekitar 6 kali lipat lebih banyak"

    return (
        f"VGG16 memakai sekitar 14,7 juta parameter pada bagian feature extractor, sedangkan MobileNetV2 hanya "
        f"sekitar 2,4 juta. Pada percobaan ini {accuracy_note}, sehingga MobileNetV2 lebih masuk akal dipakai di "
        f"aplikasi mobile karena ukurannya jauh lebih ringan tanpa mengorbankan banyak akurasi."
    )


@app.route("/")
def index():
    vgg16_metrics = load_json_file("vgg16_metrics.json")
    mobilenetv2_metrics = load_json_file("mobilenetv2_metrics.json")
    vgg16_accuracy = f"{vgg16_metrics['accuracy'] * 100:.2f}%" if vgg16_metrics else None
    mobilenetv2_accuracy = f"{mobilenetv2_metrics['accuracy'] * 100:.2f}%" if mobilenetv2_metrics else None
    return render_template(
        "index.html", active_page="beranda",
        vgg16_accuracy=vgg16_accuracy, mobilenetv2_accuracy=mobilenetv2_accuracy,
    )


@app.route("/prediksi")
def prediksi():
    return render_template("prediksi.html", active_page="prediksi")


@app.route("/hasil")
def result_page():
    vgg16_metrics = load_json_file("vgg16_metrics.json")
    mobilenetv2_metrics = load_json_file("mobilenetv2_metrics.json")
    vgg16_accuracy = vgg16_metrics["accuracy"] if vgg16_metrics else None
    mobilenetv2_accuracy = mobilenetv2_metrics["accuracy"] if mobilenetv2_metrics else None
    return render_template(
        "result.html", active_page="prediksi",
        vgg16_accuracy=vgg16_accuracy, mobilenetv2_accuracy=mobilenetv2_accuracy,
    )


@app.route("/evaluasi")
def evaluasi():
    vgg16_metrics = load_json_file("vgg16_metrics.json")
    mobilenetv2_metrics = load_json_file("mobilenetv2_metrics.json")
    if vgg16_metrics:
        vgg16_metrics["accuracy_pct"] = f"{vgg16_metrics['accuracy'] * 100:.2f}%"
    if mobilenetv2_metrics:
        mobilenetv2_metrics["accuracy_pct"] = f"{mobilenetv2_metrics['accuracy'] * 100:.2f}%"

    vgg16_report = load_json_file("vgg16_classification_report.json")
    mobilenetv2_report = load_json_file("mobilenetv2_classification_report.json")
    vgg16_history = load_json_file("vgg16_history.json")
    mobilenetv2_history = load_json_file("mobilenetv2_history.json")
    vgg16_confusion = load_json_file("vgg16_confusion_matrix.json")
    mobilenetv2_confusion = load_json_file("mobilenetv2_confusion_matrix.json")

    history_available = bool(vgg16_history and mobilenetv2_history)
    comparison_available = bool(vgg16_report and mobilenetv2_report)
    vgg16_confusion_exists = os.path.isfile(os.path.join(OUTPUTS_DIR, "vgg16_confusion_matrix.png"))
    mobilenetv2_confusion_exists = os.path.isfile(os.path.join(OUTPUTS_DIR, "mobilenetv2_confusion_matrix.png"))

    accuracy_caption, loss_caption = build_training_captions(vgg16_history, mobilenetv2_history)
    comparison_caption = build_comparison_caption(vgg16_report, mobilenetv2_report)
    vgg16_confusion_caption = build_confusion_caption("vgg16", vgg16_confusion)
    mobilenetv2_confusion_caption = build_confusion_caption("mobilenetv2", mobilenetv2_confusion)
    vgg16_report_caption = build_report_caption("vgg16", vgg16_report)
    mobilenetv2_report_caption = build_report_caption("mobilenetv2", mobilenetv2_report)
    architecture_caption = build_architecture_caption(vgg16_metrics, mobilenetv2_metrics)

    return render_template(
        "evaluasi.html", active_page="evaluasi",
        vgg16_metrics=vgg16_metrics, mobilenetv2_metrics=mobilenetv2_metrics,
        vgg16_report=vgg16_report, mobilenetv2_report=mobilenetv2_report,
        history_available=history_available, comparison_available=comparison_available,
        vgg16_confusion_exists=vgg16_confusion_exists,
        mobilenetv2_confusion_exists=mobilenetv2_confusion_exists,
        accuracy_caption=accuracy_caption, loss_caption=loss_caption,
        comparison_caption=comparison_caption,
        vgg16_confusion_caption=vgg16_confusion_caption,
        mobilenetv2_confusion_caption=mobilenetv2_confusion_caption,
        vgg16_report_caption=vgg16_report_caption,
        mobilenetv2_report_caption=mobilenetv2_report_caption,
        architecture_caption=architecture_caption,
    )


@app.route("/dataset")
def dataset_page():
    report = load_json_file("vgg16_classification_report.json")
    val_counts_by_label = {row["label"]: row["support"] for row in report["rows"]} if report else {}

    classes = []
    total_images_for_pct = sum(CLASS_COUNTS[name] for name in CLASS_NAMES)
    for name in CLASS_NAMES:
        total = CLASS_COUNTS[name]
        val_count = val_counts_by_label.get(name)
        train_count = total - val_count if val_count is not None else None
        classes.append({
            "name": name, "label": CLASS_LABELS[name], "description": CLASS_DESCRIPTIONS[name],
            "count": total, "train_count": train_count, "val_count": val_count,
            "pct": round(total / total_images_for_pct * 100, 1),
        })

    labels = [CLASS_LABELS[name] for name in CLASS_NAMES]
    counts = [CLASS_COUNTS[name] for name in CLASS_NAMES]
    distribution_caption = build_distribution_caption(labels, counts)

    total_images = sum(counts)
    total_val = sum(val_counts_by_label.values()) if val_counts_by_label else None
    total_train = total_images - total_val if total_val is not None else None
    train_pct = round(total_train / total_images * 100, 1) if total_train is not None else None
    val_pct = round(total_val / total_images * 100, 1) if total_val is not None else None
    return render_template(
        "dataset.html", active_page="dataset", classes=classes, distribution_caption=distribution_caption,
        total_images=total_images, total_train=total_train, total_val=total_val,
        train_pct=train_pct, val_pct=val_pct,
    )


@app.route("/tentang")
def tentang():
    return render_template("tentang.html", active_page="tentang")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar yang diunggah"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    raw_image = load_and_prepare_image(filepath)
    vgg16_result = predict_with_model(vgg16_model, preprocess_vgg16, raw_image)
    mobilenet_result = predict_with_model(mobilenetv2_model, preprocess_mobilenet, raw_image)

    os.remove(filepath)

    return jsonify({"vgg16": vgg16_result, "mobilenetv2": mobilenet_result})


@app.route("/api/history/<model_name>")
def api_history(model_name):
    data = load_json_file(f"{model_name}_history.json")
    if data is None:
        return jsonify({"error": "belum ada data"}), 404
    return jsonify(data)


@app.route("/api/comparison")
def api_comparison():
    vgg16 = load_json_file("vgg16_classification_report.json")
    mobilenetv2 = load_json_file("mobilenetv2_classification_report.json")
    if not vgg16 or not mobilenetv2:
        return jsonify({"error": "belum ada data"}), 404

    return jsonify({
        "labels": ["Akurasi", "Precision", "Recall", "F1-Score"],
        "vgg16": [vgg16["accuracy"], vgg16["macro_avg"]["precision"],
                  vgg16["macro_avg"]["recall"], vgg16["macro_avg"]["f1_score"]],
        "mobilenetv2": [mobilenetv2["accuracy"], mobilenetv2["macro_avg"]["precision"],
                         mobilenetv2["macro_avg"]["recall"], mobilenetv2["macro_avg"]["f1_score"]],
    })


@app.route("/api/dataset-distribution")
def api_dataset_distribution():
    return jsonify({
        "labels": [CLASS_LABELS[name] for name in CLASS_NAMES],
        "counts": [CLASS_COUNTS[name] for name in CLASS_NAMES],
    })


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
