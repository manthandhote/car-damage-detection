import pandas as pd

FEATURE_COLUMNS = ['name', 'fuel', 'seller_type', 'transmission', 'owner',
                    'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

def preprocess_input(name, year, km_driven, fuel, seller_type,
                    transmission, owner, mileage, engine, max_power, seats):
    """Build a raw-feature DataFrame for the trained model pipeline.

    The saved model (see training/train_price_model.py) is a scikit-learn
    Pipeline with its own OneHotEncoder step, so no manual label encoding is
    needed here - the pipeline handles categorical columns (and unseen
    categories) internally.
    """
    return pd.DataFrame(
        [[name, fuel, seller_type, transmission, owner,
          year, km_driven, mileage, engine, max_power, seats]],
        columns=FEATURE_COLUMNS,
    )

def get_brand_name(car_name):
    """Extract brand name from full car name"""
    return car_name.split(' ')[0].strip()
