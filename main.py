from src.logger import logger
from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessing
from src.train import ModelTrainer


def main():

    logger.info("Manufacturing Defect Prediction Project Started")

    # Data Ingestion
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()

    # Data Preprocessing
    preprocessing = DataPreprocessing()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = preprocessing.initiate_preprocessing()

    # Model Training
    trainer = ModelTrainer()

    trainer.train(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nProject Executed Successfully")


if __name__ == "__main__":
    main()