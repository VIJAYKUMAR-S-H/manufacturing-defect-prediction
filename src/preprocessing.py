import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logger
from src.utils import save_object


class DataPreprocessing:
    """
    Handles data preprocessing for training and prediction.
    """

    def __init__(self):
        self.train_path = "data/processed/train.csv"
        self.test_path = "data/processed/test.csv"

        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.preprocessor_path = os.path.join(
            self.model_dir,
            "preprocessor.pkl"
        )

    def initiate_preprocessing(self):

        logger.info("Loading train and test datasets")

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Target column
        target_column = "Parts_Per_Hour"

        # Drop timestamp since it is not useful for Linear Regression
        if "Timestamp" in train_df.columns:
            train_df = train_df.drop(columns=["Timestamp"])
            test_df = test_df.drop(columns=["Timestamp"])

        # Split features and target
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Numerical and categorical columns
        numerical_columns = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()

        logger.info(f"Numerical Columns: {numerical_columns}")
        logger.info(f"Categorical Columns: {categorical_columns}")

        # Numerical Pipeline
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical Pipeline
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

        logger.info("Fitting preprocessing pipeline")

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        save_object(preprocessor, self.preprocessor_path)

        logger.info("Preprocessor saved successfully")

        print("\nPreprocessing Completed Successfully")
        print("Training Shape :", X_train_processed.shape)
        print("Testing Shape  :", X_test_processed.shape)

        return (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
        )