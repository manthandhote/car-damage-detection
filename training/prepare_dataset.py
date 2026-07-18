"""
Download the CarDD dataset (via its FiftyOne/Hugging Face mirror
'harpreetsahota/CarDD') and export it into YOLO segmentation format for
Ultralytics training.

This dataset's exact label-field name isn't something we can hardcode with
certainty in advance (Hugging Face dataset schemas vary by uploader), so this
script prints the dataset's field schema before exporting. If the auto-detect
below picks the wrong field, re-run with:
    python training/prepare_dataset.py --label-field <field_name>
using the field name printed in the "Available label fields" list.

Run from the repo root (needs `pip install fiftyone ultralytics`):
    python training/prepare_dataset.py
"""
import argparse
import os
import shutil

import fiftyone as fo
import fiftyone.utils.huggingface as fouh
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(ROOT, "training", "datasets", "cardd_yolo")
DATA_YAML_PATH = os.path.join(ROOT, "training", "data.yaml")

CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]


def detect_label_field(dataset):
    """Pick the first field holding Detections/Polylines/Segmentation labels."""
    schema = dataset.get_field_schema(flat=True)
    label_types = (
        fo.Detections, fo.Polylines, fo.Polyline,
        fo.Segmentation, fo.Detection,
    )
    candidates = []
    for name, field in schema.items():
        doc_type = getattr(field, "document_type", None)
        if doc_type is not None and issubclass(doc_type, label_types):
            candidates.append(name)

    print("Available label fields:", candidates or "(none found - inspect manually)")
    if not candidates:
        raise RuntimeError(
            "Could not auto-detect a label field. Inspect `dataset.get_field_schema()` "
            "and re-run with --label-field <name>."
        )

    # Prefer a field that looks like it carries instance masks (needed for
    # YOLO segmentation training) over plain bounding-box fields.
    for preferred in ("segmentations", "segmentation", "masks"):
        if preferred in candidates:
            return preferred
    return candidates[0]


def verify_polygon_labels(export_dir, sample_size=20):
    """Sanity-check that exported labels are YOLO-seg polygons, not plain boxes.

    A YOLO detection label line is "class x y w h" (5 fields). A YOLO
    segmentation label line is "class x1 y1 x2 y2 ..." (odd count > 5,
    variable length per polygon). FiftyOne's mask->polygon export
    (use_masks=True) has known edge cases (see
    https://github.com/voxel51/fiftyone/issues/6421) where it silently falls
    back to box-only output, which trains a broken 'segment' model - so we
    check this locally right after export instead of finding out 10+ minutes
    into a training run.
    """
    train_labels_dir = os.path.join(export_dir, "labels", "train")
    label_files = [f for f in os.listdir(train_labels_dir) if f.endswith(".txt")][:sample_size]
    if not label_files:
        raise RuntimeError(f"No label files found in {train_labels_dir}")

    box_only_count = 0
    for fname in label_files:
        with open(os.path.join(train_labels_dir, fname)) as f:
            for line in f:
                fields = line.strip().split()
                if fields and len(fields) == 5:
                    box_only_count += 1

    if box_only_count:
        raise RuntimeError(
            f"{box_only_count} label line(s) in a sample of {len(label_files)} files look like "
            "plain bounding boxes (5 fields), not segmentation polygons (>5 fields). "
            "The mask->polygon export likely failed for some instances - do not proceed to "
            "training yet. Inspect training/datasets/cardd_yolo/labels/train/*.txt manually."
        )
    print(f"Verified {len(label_files)} label files look like polygon (segmentation) format.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-field", default=None, help="Override auto-detected label field name")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    print("Loading CarDD from Hugging Face (this downloads ~500MB-1GB, may take a while)...")
    dataset = fouh.load_from_hub("harpreetsahota/CarDD", name="CarDD", persistent=False)
    print(dataset)

    label_field = args.label_field or detect_label_field(dataset)
    print(f"Using label field: {label_field}")

    dataset.shuffle(seed=42)
    val_count = int(len(dataset) * args.val_fraction)
    val_view = dataset[:val_count]
    train_view = dataset[val_count:]

    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)

    for split_name, view in [("train", train_view), ("val", val_view)]:
        view.export(
            export_dir=EXPORT_DIR,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=CLASS_NAMES,
            split=split_name,
            use_masks=True,  # convert instance masks to polygons instead of plain boxes
        )

    verify_polygon_labels(EXPORT_DIR)

    data_yaml = {
        "path": EXPORT_DIR,
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    os.makedirs(os.path.dirname(DATA_YAML_PATH), exist_ok=True)
    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Wrote {DATA_YAML_PATH}")
    print(f"Train: {len(train_view)} images, Val: {len(val_view)} images")


if __name__ == "__main__":
    main()
