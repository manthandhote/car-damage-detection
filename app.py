import pandas as pd
import streamlit as st
from PIL import Image

from utils.inference import predict
from utils.models import load_damage_model, load_price_model
from utils.preprocessing import get_brand_name, preprocess_input
from utils.pricing import compute_discount
from utils.config import CAR_DATA_PATH

st.set_page_config(
    page_title="Car Damage Detection & Price Estimator",
    page_icon="🔧",
    layout="wide",
)

st.markdown("<h1>🔧 Car Damage Detection & Price Estimator</h1>", unsafe_allow_html=True)
st.markdown(
    "Enter your car's specifications for a base price estimate, then upload a "
    "photo to detect damage (dent, scratch, crack, glass shatter, lamp broken, "
    "tire flat) with a YOLOv8 instance segmentation model fine-tuned on "
    "[CarDD](https://arxiv.org/abs/2211.00945) - the final price is adjusted "
    "based on the type and amount of damage found."
)


@st.cache_data
def load_car_data():
    cars_data = pd.read_csv(CAR_DATA_PATH)
    cars_data["name"] = cars_data["name"].apply(get_brand_name)
    return cars_data


try:
    price_model = load_price_model()
    damage_model = load_damage_model()
except Exception as e:
    st.error(f"Failed to load prediction models: {e}")
    st.stop()

try:
    cars_data = load_car_data()
except Exception as e:
    st.error(f"Failed to load car dataset: {e}")
    st.stop()

st.subheader("🚘 Car Specifications")
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox("Car Brand", cars_data["name"].unique(), key="brand")
    year = st.slider("Manufactured Year", 1994, 2024, 2015, key="year")
    km_driven = st.slider("Kilometers Driven", 11, 200000, 50000, key="km_driven")
    fuel = st.selectbox("Fuel Type", cars_data["fuel"].unique(), key="fuel")
    seller_type = st.selectbox("Seller Type", cars_data["seller_type"].unique(), key="seller_type")

with col2:
    transmission = st.selectbox("Transmission", cars_data["transmission"].unique(), key="transmission")
    owner = st.selectbox("Owner", cars_data["owner"].unique(), key="owner")
    mileage = st.slider("Mileage (km/l)", 10, 40, 20, key="mileage")
    engine = st.slider("Engine (CC)", 700, 5000, 1500, key="engine")
    max_power = st.slider("Max Power (bhp)", 0, 200, 100, key="max_power")
    seats = st.slider("Number of Seats", 2, 10, 5, key="seats")

if st.button("📊 Calculate Base Price"):
    try:
        input_df = preprocess_input(name, year, km_driven, fuel, seller_type,
                                     transmission, owner, mileage, engine, max_power, seats)
        base_price = price_model.predict(input_df)[0]
        st.session_state.base_price = round(base_price, 2)
        st.balloons()
    except Exception as e:
        st.error(f"Couldn't calculate a price for this combination: {e}")

if "base_price" in st.session_state:
    st.markdown("---")
    st.subheader("💰 Base Price")
    st.markdown(f"**₹ {st.session_state.base_price:,}**")

    st.markdown("---")
    st.subheader("🔍 Damage Assessment")

    uploaded_file = st.file_uploader("Upload a photo of the car", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Couldn't read that image, please upload a valid JPG/PNG: {e}")
            image = None

        if image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)

            if st.button("🔎 Detect Damage & Final Price"):
                try:
                    with st.spinner("Analyzing image..."):
                        annotated_image, detections = predict(image, damage_model)
                        discount, breakdown = compute_discount(detections)
                        final_price = round(st.session_state.base_price * (1 - discount), 2)

                    with col2:
                        st.markdown("**Detected Damage**")
                        st.image(annotated_image, channels="BGR", use_container_width=True)

                    st.markdown("---")
                    st.subheader("💵 Final Price")

                    if breakdown:
                        st.dataframe(
                            pd.DataFrame(breakdown).rename(columns={
                                "category": "Damage Type", "count": "Count", "contribution": "Discount",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.warning(
                            f"Detected damage reduces the price by {discount * 100:.1f}% "
                            f"(₹ {st.session_state.base_price - final_price:,.2f})."
                        )
                    else:
                        st.success("No damage detected - price unchanged.")

                    st.markdown(f"**Final Estimated Price: ₹ {final_price:,}**")
                except Exception as e:
                    st.error(f"Damage assessment failed: {e}")
