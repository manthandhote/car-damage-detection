import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "best.pt")
PRICE_MODEL_PATH = os.path.join(ROOT, "models", "price_model.pkl")
CAR_DATA_PATH = os.path.join(ROOT, "data", "Cardetails.csv")

# CarDD's 6 fine-grained damage categories (order must match data.yaml
# produced by training/prepare_dataset.py).
CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]

CONFIDENCE_THRESHOLD = 0.25
