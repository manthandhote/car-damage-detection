from .config import CONFIDENCE_THRESHOLD


def predict(image, model):
    """Run damage detection/segmentation on a PIL image.

    Returns (annotated_image_bgr, detections) where detections is a list of
    dicts: {"class_name": str, "confidence": float}.
    """
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    annotated_image = result.plot()  # BGR numpy array with boxes/masks/labels drawn

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_name": model.names[class_id],
                "confidence": round(float(box.conf.item()), 3),
            })

    return annotated_image, detections
