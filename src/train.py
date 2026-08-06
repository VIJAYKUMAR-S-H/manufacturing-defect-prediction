import os
import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.logger import logger
from src.utils import save_object


class ModelTrainer:

    def __init__(self):
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.metrics_path = os.path.join(self.model_dir, "metrics.json")

    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        logger.info("Starting Model Training...")

        model = LinearRegression()

        model.fit(X_train, y_train)

        logger.info("Model Training Completed")

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        metrics = {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2 Score": float(r2),
        }

        print("\n========== MODEL EVALUATION ==========")
        print(f"MAE       : {mae:.4f}")
        print(f"MSE       : {mse:.4f}")
        print(f"RMSE      : {rmse:.4f}")
        print(f"R2 Score  : {r2:.4f}")

        save_object(model, self.model_path)

        with open(self.metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Model and Metrics Saved Successfully")

        return model