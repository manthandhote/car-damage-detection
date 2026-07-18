import pickle as pk

import streamlit as st
from ultralytics import YOLO

from .config import MODEL_PATH, PRICE_MODEL_PATH


@st.cache_resource
def load_damage_model():
    """Load the trained YOLOv8-seg damage detection model."""
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_price_model():
    """Load the trained car price regression pipeline."""
    with open(PRICE_MODEL_PATH, "rb") as f:
        return pk.load(f)
