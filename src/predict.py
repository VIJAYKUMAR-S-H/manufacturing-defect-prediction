import pandas as pd

from src.utils import load_object
from src.logger import logger


class PredictionPipeline:

    def __init__(self):
        self.model_path = "models/model.pkl"
        self.preprocessor_path = "models/preprocessor.pkl"

    def predict(self, input_data: dict):

        logger.info("Loading model and preprocessor")

        model = load_object(self.model_path)
        preprocessor = load_object(self.preprocessor_path)

        input_df = pd.DataFrame([input_data])

        # Timestamp was removed during preprocessing
        if "Timestamp" in input_df.columns:
            input_df = input_df.drop(columns=["Timestamp"])

        transformed_data = preprocessor.transform(input_df)

        prediction = model.predict(transformed_data)

        logger.info("Prediction Successful")

        return prediction[0]