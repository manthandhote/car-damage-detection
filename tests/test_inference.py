import os

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "best.pt")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="models/best.pt not present")
def test_model_loads_and_predicts_on_sample_image():
    from utils.inference import predict
    from utils.models import load_damage_model

    model = load_damage_model()

    sample_dir = os.path.join(ROOT, "data", "sample_images")
    image_extensions = (".jpg", ".jpeg", ".png")
    sample_images = (
        [f for f in os.listdir(sample_dir) if f.lower().endswith(image_extensions)]
        if os.path.isdir(sample_dir) else []
    )
    if not sample_images:
        pytest.skip("no sample images in data/sample_images")

    image = Image.open(os.path.join(sample_dir, sample_images[0])).convert("RGB")
    annotated_image, detections = predict(image, model)

    assert annotated_image is not None
    assert isinstance(detections, list)
