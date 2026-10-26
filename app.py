"""Flask application for visual conveyor-belt damage inspections."""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

try:
    from PIL import Image
except ImportError:
    Image = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
DATABASE_PATH = BASE_DIR / "database" / "inspections.db"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MODEL_MISSING_MESSAGE = (
    "AI model not loaded. Please place the trained YOLO model at model/best.pt"
)

DAMAGE_CLASSES = {"hole", "puncture", "tear", "impact damage", "patch work"}
SEVERE_CLASSES = {"hole", "puncture", "tear"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "local-development-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["MAX_IMAGE_PIXELS"] = 40_000_000
app.logger.setLevel(logging.INFO)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

_model = None
_model_checked = False
_model_error = None


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                image_name TEXT NOT NULL,
                upload_file TEXT NOT NULL,
                result_file TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                defect_count INTEGER NOT NULL,
                confidence REAL,
                risk_level TEXT NOT NULL,
                risk_score INTEGER,
                condition_label TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_available INTEGER NOT NULL,
                error_message TEXT
            )
            """
        )


def get_model():
    """Load trained weights on first use; keep the UI available without them."""
    global _model, _model_checked, _model_error
    if _model_checked:
        return _model

    _model_checked = True
    if not MODEL_PATH.is_file():
        _model_error = MODEL_MISSING_MESSAGE
        return None
    try:
        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))
    except ImportError:
        _model_error = "Ultralytics is not installed. Install dependencies from requirements.txt."
    except Exception as error:
        app.logger.exception("Could not load YOLO model")
        _model_error = f"The YOLO model could not be loaded: {error}"
    return _model


def parse_record(row):
    record = dict(row)
    record["detections"] = json.loads(record.pop("detections_json"))
    record["model_available"] = bool(record["model_available"])
    return record


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(path):
    """Reject corrupt or unsupported image data before it reaches the model."""
    if Image is not None:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            return width > 0 and height > 0 and width * height <= app.config["MAX_IMAGE_PIXELS"]
        except Exception:
            return False

    try:
        import cv2
    except ImportError:
        return False
    image = cv2.imread(str(path))
    return image is not None and image.size > 0


def assess_risk(detections):
    """Return a rule-based damage-risk estimate, not a rupture-time forecast."""
    damage_detections = [
        item for item in detections if item["class_name"].strip().lower() in DAMAGE_CLASSES
    ]
    severe_count = sum(
        item["class_name"].strip().lower() in SEVERE_CLASSES for item in damage_detections
    )

    if not damage_detections:
        risk, score, condition = "LOW", 8, "NORMAL"
    elif severe_count >= 2:
        risk, score, condition = "VERY HIGH", 95, "CRITICAL"
    elif severe_count == 1:
        risk, score, condition = "HIGH", 72, "CRITICAL"
    else:
        risk, score, condition = "MEDIUM", 42, "WARNING"

    if not damage_detections:
        recommendation = "No visible damage was detected. Continue routine inspection."
    elif len(damage_detections) >= 2:
        recommendation = "Schedule an immediate detailed inspection of all detected damage locations."
    else:
        label = damage_detections[0]["class_name"].strip().lower()
        recommendations = {
            "hole": "Inspect the affected belt section and consider immediate maintenance.",
            "puncture": "Inspect the puncture location and check whether the damage is expanding.",
            "tear": "Inspect the tear immediately and check belt tension and alignment.",
            "impact damage": "Inspect the affected region for structural damage.",
            "patch work": "Check the repaired section for further deterioration.",
        }
        recommendation = recommendations.get(
            label, "Inspect the affected belt section and schedule maintenance as needed."
        )

    return {
        "damage_detections": damage_detections,
        "risk_level": risk,
        "risk_score": score,
        "condition_label": condition,
        "recommendation": recommendation,
    }


def run_detection(image_path, result_path):
    model = get_model()
    if model is None:
        return None, _model_error or MODEL_MISSING_MESSAGE
    try:
        import cv2
    except ImportError:
        return None, "OpenCV is not installed. Install dependencies from requirements.txt."

    predictions = model.predict(source=str(image_path), verbose=False)
    prediction = predictions[0]
    detections = []
    if prediction.boxes is not None:
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            class_names = prediction.names
            class_name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else class_names[class_id]
            coordinates = [round(float(value), 1) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_name": str(class_name),
                    "confidence": round(float(box.conf.item()), 4),
                    "confidence_percent": round(float(box.conf.item()) * 100, 1),
                    "bbox": coordinates,
                    "is_damage": str(class_name).strip().lower() in DAMAGE_CLASSES,
                }
            )

    annotated = prediction.plot()
    if not cv2.imwrite(str(result_path), annotated):
        raise OSError("The annotated image could not be saved.")
    return detections, None


def save_inspection(
    inspection_id,
    created_at,
    image_name,
    upload_file,
    result_file,
    detections,
    assessment,
    model_available,
    error_message=None,
):
    damage_detections = assessment.get("damage_detections", [])
    confidence = max((item["confidence"] for item in damage_detections), default=None)
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO inspections (
                id, created_at, image_name, upload_file, result_file, detections_json,
                defect_count, confidence, risk_level, risk_score, condition_label,
                recommendation, model_available, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                created_at,
                image_name,
                upload_file,
                result_file,
                json.dumps(detections),
                len(damage_detections),
                confidence,
                assessment.get("risk_level", "NOT ASSESSED"),
                assessment.get("risk_score"),
                assessment.get("condition_label", "MODEL REQUIRED"),
                assessment.get("recommendation", MODEL_MISSING_MESSAGE),
                int(model_available),
                error_message,
            ),
        )


def wants_json():
    return request.args.get("format") == "json" or request.accept_mimetypes.best == "application/json"


def get_recent_inspections(limit=8):
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM inspections ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [parse_record(row) for row in rows]


@app.errorhandler(413)
def upload_too_large(_error):
    message = f"Image is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
    if wants_json():
        return jsonify({"success": False, "error": message}), 413
    return render_template("inspect.html", error=message), 413


@app.route("/")
def index():
    recent = get_recent_inspections()
    with get_db_connection() as connection:
        total = connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    risk_counts = {level: 0 for level in ("LOW", "MEDIUM", "HIGH", "VERY HIGH")}
    for record in get_recent_inspections(limit=1000):
        if record["risk_level"] in risk_counts:
            risk_counts[record["risk_level"]] += 1
    latest = recent[0] if recent else None
    model_ready = get_model() is not None
    return render_template(
        "index.html",
        recent=recent[:5],
        latest=latest,
        total_inspections=total,
        risk_counts=risk_counts,
        model_ready=model_ready,
        model_message=None if model_ready else (_model_error or MODEL_MISSING_MESSAGE),
    )


@app.route("/inspect")
def inspect():
    model_ready = get_model() is not None
    return render_template(
        "inspect.html",
        model_ready=model_ready,
        model_message=None if model_ready else (_model_error or MODEL_MISSING_MESSAGE),
    )


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        message = "Choose an image file before analyzing the belt."
        if wants_json():
            return jsonify({"success": False, "error": message}), 400
        return render_template("inspect.html", error=message, model_ready=get_model() is not None), 400

    original_name = secure_filename(uploaded.filename)
    if not original_name or not allowed_file(original_name):
        message = "Unsupported file type. Upload a JPG, PNG, BMP, or WEBP image."
        if wants_json():
            return jsonify({"success": False, "error": message}), 400
        return render_template("inspect.html", error=message, model_ready=get_model() is not None), 400

    inspection_id = uuid.uuid4().hex
    extension = Path(original_name).suffix.lower()
    stored_name = f"{inspection_id}{extension}"
    result_name = f"{inspection_id}.jpg"
    upload_path = UPLOAD_DIR / stored_name
    result_path = RESULT_DIR / result_name
    uploaded.save(upload_path)

    if not validate_image(upload_path):
        upload_path.unlink(missing_ok=True)
        message = "The uploaded file is not a valid image or its dimensions are too large."
        if wants_json():
            return jsonify({"success": False, "error": message}), 400
        return render_template("inspect.html", error=message, model_ready=get_model() is not None), 400

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    detections, error_message = None, None
    try:
        detections, error_message = run_detection(upload_path, result_path)
    except Exception:
        app.logger.exception("YOLO inference failed")
        error_message = "Analysis could not be completed. Check the image and trained model, then try again."

    if detections is None:
        detections = []
        assessment = {
            "damage_detections": [],
            "risk_level": "NOT ASSESSED",
            "risk_score": None,
            "condition_label": "MODEL REQUIRED",
            "recommendation": error_message or MODEL_MISSING_MESSAGE,
        }
        result_name = stored_name
    else:
        assessment = assess_risk(detections)

    save_inspection(
        inspection_id,
        created_at,
        original_name,
        stored_name,
        result_name,
        detections,
        assessment,
        detections is not None and error_message is None,
        error_message,
    )
    record = {
        "id": inspection_id,
        "created_at": created_at,
        "image_name": original_name,
        "upload_file": stored_name,
        "result_file": result_name,
        "detections": detections,
        "defect_count": len(assessment["damage_detections"]),
        "confidence": max(
            (item["confidence"] for item in assessment["damage_detections"]), default=None
        ),
        **{key: value for key, value in assessment.items() if key != "damage_detections"},
        "model_available": detections is not None and error_message is None,
        "error_message": error_message,
    }

    if wants_json():
        image_kind = "results" if record["model_available"] else "uploads"
        image_filename = record["result_file"] if record["model_available"] else record["upload_file"]
        return jsonify(
            {
                "success": True,
                "record": record,
                "annotated_image": url_for("media_file", kind=image_kind, filename=image_filename),
                "results_url": url_for("results", inspection_id=inspection_id),
            }
        )
    return redirect(url_for("results", inspection_id=inspection_id))


@app.route("/results/<inspection_id>")
def results(inspection_id):
    with get_db_connection() as connection:
        row = connection.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
    if row is None:
        abort(404)
    record = parse_record(row)
    record["annotated_url"] = url_for(
        "media_file",
        kind="results" if record["model_available"] else "uploads",
        filename=record["result_file"] if record["model_available"] else record["upload_file"],
    )
    record["uploaded_url"] = url_for("media_file", kind="uploads", filename=record["upload_file"])
    return render_template("results.html", record=record)


@app.route("/history", methods=["GET"])
def history():
    filters = {
        "date": request.args.get("date", "").strip(),
        "damage": request.args.get("damage", "").strip(),
        "risk": request.args.get("risk", "").strip().upper(),
    }
    with get_db_connection() as connection:
        rows = connection.execute("SELECT * FROM inspections ORDER BY created_at DESC").fetchall()
    records = [parse_record(row) for row in rows]
    if filters["date"]:
        records = [record for record in records if record["created_at"].startswith(filters["date"])]
    if filters["risk"]:
        records = [record for record in records if record["risk_level"] == filters["risk"]]
    if filters["damage"]:
        search = filters["damage"].casefold()
        records = [
            record
            for record in records
            if any(search in item["class_name"].casefold() for item in record["detections"])
        ]
    if wants_json():
        return jsonify({"inspections": records, "count": len(records)})
    return render_template("history.html", records=records, filters=filters)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    with get_db_connection() as connection:
        rows = connection.execute("SELECT upload_file, result_file, model_available FROM inspections").fetchall()
        connection.execute("DELETE FROM inspections")
    for row in rows:
        (UPLOAD_DIR / row["upload_file"]).unlink(missing_ok=True)
        if row["model_available"]:
            (RESULT_DIR / row["result_file"]).unlink(missing_ok=True)
    if wants_json():
        return jsonify({"success": True, "deleted": len(rows)})
    flash(f"Cleared {len(rows)} inspection record(s).", "success")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/media/<kind>/<path:filename>")
def media_file(kind, filename):
    directories = {"uploads": UPLOAD_DIR, "results": RESULT_DIR}
    directory = directories.get(kind)
    if directory is None:
        abort(404)
    return send_from_directory(directory, filename)


init_database()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)