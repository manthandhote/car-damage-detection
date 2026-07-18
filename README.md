# 🔧 Car Damage Detection & Price Estimator

A computer-vision + ML app: enter a car's specifications for a base price
estimate, upload a photo, and get back the image annotated with
bounding boxes/segmentation masks for each damaged region - plus a final
price adjusted for the type and amount of damage found.

Unlike a simple "damaged / not damaged" classifier, this identifies **what**
kind of damage and **where** it is, then reflects that in the price instead
of a flat discount.

## How it works

1. **Base price** - a scikit-learn `Pipeline` (one-hot encoding +
   `GradientBoostingRegressor`) trained on ~8,100 used-car listings predicts
   a base price from brand, year, kilometers driven, fuel type, seller type,
   transmission, ownership history, mileage, engine size, max power, and
   seat count.
2. **Damage detection** - a YOLOv8n-seg model fine-tuned on
   [CarDD](https://arxiv.org/abs/2211.00945) (Wang et al., IEEE Transactions
   on Intelligent Transportation Systems, 2023) detects and segments 6
   damage categories: dent, scratch, crack, glass shatter, lamp broken, tire
   flat.
3. **Final price** - the base price is reduced by a severity-weighted
   discount computed from the detected damage (see below).

## Price discount logic

There's no dataset mapping "this damage" to "this price impact," so this is
a documented heuristic reflecting relative real-world repair cost, not a
learned function:

| Damage type | Base discount (single instance) |
|---|---|
| Glass shatter | 10% |
| Crack | 8% |
| Dent | 5% |
| Lamp broken | 4% |
| Scratch | 3% |
| Tire flat | 3% |

Repeated instances of the same type add diminishing extra discount
(`weight * (1 + 0.3 * (count - 1))`), capped at 2x the base weight per
category. The total discount across all categories is capped at 50%. See
[utils/pricing.py](utils/pricing.py) for the implementation and
[tests/test_pricing.py](tests/test_pricing.py) for the exact behavior.

## Model performance

| Price model (GradientBoostingRegressor) | Value |
|---|---|
| R² | 0.975 |
| MAE | ₹86,480 |

| Damage model (YOLOv8n-seg) | mAP50 | mAP50-95 |
|---|---|---|
| Overall | 0.662 | 0.505 |
| Glass shatter | 0.977 | - |
| Tire flat | 0.968 | - |
| Lamp broken | 0.839 | - |
| Scratch | 0.512 | - |
| Dent | 0.490 | - |
| Crack | 0.185 | - |

Crack detection is noticeably weaker than the other categories - it has the
fewest training instances and is the thinnest, lowest-contrast damage type to
segment accurately. Full metrics:
[training/reports/metrics.json](training/reports/metrics.json).

## Project structure

```
app.py                          # Streamlit app: specs -> base price -> photo -> final price
utils/
  models.py                     # cached loaders for both models
  inference.py                  # damage detection prediction
  preprocessing.py              # feature dataframe builder for the price pipeline
  pricing.py                    # severity-weighted discount logic
  config.py                     # paths, class names, confidence threshold
training/
  prepare_dataset.py            # downloads CarDD, converts to YOLO-seg format
  train_damage_model.py         # trains YOLOv8n-seg, saves models/best.pt + metrics
  reports/metrics.json          # last training run's metrics
models/
  best.pt                       # trained YOLO damage detection weights
  price_model.pkl               # trained price regression pipeline
data/
  Cardetails.csv                # used-car listings dataset (price model training + UI dropdowns)
  sample_images/                # example photos used by tests/README
tests/                          # pytest unit + smoke tests
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Training / retraining the damage model

Needs a GPU for reasonable training time (CPU works but is much slower).

```bash
pip install -r training/requirements-training.txt
python training/prepare_dataset.py
python training/train_damage_model.py
```

The price model's training script lives in the sibling
[car-price-estimator](https://github.com/manthandhote/car-price-estimator)
repo (`training/train_price_model.py`) - `models/price_model.pkl` here is a
copy of its output.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest -v
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

**Live demo:** _add link here after deploying_

## License

MIT — see [LICENSE](LICENSE).
