import joblib
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(filepath):
    logger.info("Loading model for prediction")
    model = joblib.load("artifacts/model.joblib")

    import pandas as pd
    df = pd.read_csv(filepath).drop(columns="Failure Type").head(5)
    preds = model.predict(df)

    logger.info(f"Predictions: {preds}")
    return preds
