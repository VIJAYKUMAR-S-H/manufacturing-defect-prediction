import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger


class DataIngestion:
    """
    Handles loading the dataset and splitting it into
    training and testing datasets.
    """

    def __init__(self):
        self.raw_data_path = "data/raw/manufacturing_dataset_1000_samples.csv"
        self.processed_data_dir = "data/processed"

        os.makedirs(self.processed_data_dir, exist_ok=True)

    def initiate_data_ingestion(self):
        """
        Reads the dataset, splits it into train/test sets,
        and saves them into the processed folder.
        """

        logger.info("Starting data ingestion")

        # Read dataset
        df = pd.read_csv(self.raw_data_path)

        logger.info(f"Dataset loaded successfully")
        logger.info(f"Dataset Shape : {df.shape}")

        print("\nDataset Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        # Train-Test Split
        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_path = os.path.join(
            self.processed_data_dir,
            "train.csv"
        )

        test_path = os.path.join(
            self.processed_data_dir,
            "test.csv"
        )

        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        logger.info("Train and Test files saved successfully")

        return train_path, test_path