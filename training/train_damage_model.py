"""
Fine-tune YOLOv8n-seg on CarDD for car damage instance segmentation.

Requires training/data.yaml (produced by training/prepare_dataset.py).

Run from the repo root (GPU strongly recommended):
    python training/train_damage_model.py
"""
import json
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "training", "data.yaml")
MODEL_PATH = os.path.join(ROOT, "models", "best.pt")
METRICS_PATH = os.path.join(ROOT, "training", "reports", "metrics.json")

EPOCHS = 50
IMG_SIZE = 640


def main():
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=os.path.join(ROOT, "training", "runs"),
        name="cardd_yolov8n_seg",
    )

    metrics = model.val(data=DATA_YAML)

    summary = {
        "map50": round(float(metrics.seg.map50), 4),
        "map50_95": round(float(metrics.seg.map), 4),
        "per_class_map50": {
            name: round(float(ap), 4)
            for name, ap in zip(metrics.names.values(), metrics.seg.ap50)
        },
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
    }
    print("Damage model metrics:", summary)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    shutil.copy(best_weights, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
