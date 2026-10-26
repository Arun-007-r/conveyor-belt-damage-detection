# Intelligent Conveyor Belt Monitoring System

A local Flask application for detecting visible conveyor-belt damage in uploaded images with a trained Ultralytics YOLO model. The application presents a transparent, rule-based **Potential Joint Failure Risk** assessment from visible damage; it does not predict when a joint will rupture.

## Project objective

Help maintenance personnel review belt images, see detected classes and bounding boxes, and prioritize follow-up inspections. The web interface works without model weights so the dashboard, upload flow, history, and architecture pages remain available while model setup is in progress.

## Dataset

The uploaded Roboflow YOLOv8 dataset is extracted to `dataset/conveyor-belt-damage/` and the original ZIP is kept in the project root. It contains 922 labeled images in train, valid, and test splits and the following eight classes:

| Class | Use in the application |
| --- | --- |
| Hole | Damage |
| Human | Context only |
| Other Objects | Context only |
| Puncture | Damage |
| Roller | Context only |
| Tear | Damage |
| impact damage | Damage |
| patch work | Damage |

The archive metadata identifies the dataset as CC BY 4.0. Retain its attribution and license when redistributing it. Dataset labels are not model weights; the archive does **not** include a trained `best.pt` file.

## YOLO model

The app loads `model/best.pt` with Ultralytics when a prediction is requested. Place your trained weights at that exact path. The class names embedded in the weights should match the dataset class order above. If weights are absent, the app displays:

> AI model not loaded. Please place the trained YOLO model at model/best.pt

No mock or sample detections are generated in this state. The included dataset can be used to train a model separately; training is not performed by the web application.

## Installation

In the VS Code terminal, from the project folder, run these commands in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. On a machine with an NVIDIA GPU, install the PyTorch build suited to its CUDA version before installing Ultralytics, following the official PyTorch selector.

## Run in VS Code

1. Open this folder in VS Code and select the `.venv` Python interpreter.
2. Add the trained weights as `model/best.pt`.
3. In the VS Code terminal, run:

```powershell
python app.py
```

4. Open <http://127.0.0.1:5000> in your browser. The app creates `database/inspections.db` automatically on its first run.

The app can also start before the weights are installed; inference remains disabled and the missing-model message is shown.

## How to inspect an image

Open **Image inspection**, choose a JPG, JPEG, PNG, BMP, or WEBP image up to 16 MB, preview it, and select **Analyze belt**. When the model is available, the server validates the image, runs YOLO inference, saves an annotated image, separates damage classes from context classes, calculates a rule-based risk category, and writes the inspection to SQLite. The results page shows all detections, confidence, bounding boxes in pixel coordinates, the annotated image, condition, risk, and maintenance guidance.

## Risk assessment and limitations

- No detected damage: LOW / NORMAL.
- Patch work or impact damage only: MEDIUM / WARNING.
- One hole, puncture, or tear: HIGH / CRITICAL.
- Two or more severe detections: VERY HIGH / CRITICAL.

The percentage on screen is a **rule-based damage-risk indicator**, not a calibrated probability. Risk summarizes visible detections and is not a direct prediction of future joint rupture or its timing. Performance depends on the quality, coverage, and calibration of the supplied model and images. Human, Other Objects, and Roller detections are shown for context but do not increase the damage risk. This tool does not replace an in-person safety inspection.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Dashboard |
| GET | `/inspect` | Image upload and preview |
| POST | `/predict` | Multipart image upload and YOLO inference; returns JSON when requested with `Accept: application/json` |
| GET | `/results/<inspection_id>` | Inspection results page |
| GET | `/history` | History page; add `?format=json` for inspection records |
| POST | `/clear-history` | Clear saved records and app-generated inspection images |
| GET | `/about` | System architecture and limitations |

Example API request:

```powershell
curl.exe -X POST -H "Accept: application/json" -F "image=@conveyor.jpg" http://127.0.0.1:5000/predict
```

## Future improvements

- Real-time camera monitoring
- IoT sensor integration
- Conveyor belt vibration monitoring
- Temperature monitoring
- Belt tension monitoring
- Time-series prediction
- Automatic alerts
- Cloud deployment