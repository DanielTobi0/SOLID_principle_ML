import logging

from config.settings import DATA_PATH, TARGET

from data.loader import DataLoader, CSVReader
from data.cleaner import DefaultFiller, DataCleaner, DefaultColumnSelector
from data.transformation import (
    LabelEncoderWrapper,
    MinMaxScalerWrapper
)
from data.splitting import DefaultDataSplitting
from models.logistic_regression import LogisticRegressionWrapper
from evaluation.evaluator import F1SCORE


class MLPipeline:
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Logging is configured.")

    def load_data(self):
        self.logger.info("Loading data...")
        csv_reader = CSVReader()
        loader = DataLoader(reader=csv_reader)
        data = loader.load_data(self.data_path)
        if data is None:
            self.logger.error("Data loading failed")
            raise ValueError("Data loading failed")
        return data

    def clean_and_transform(self, data):
        column_selector = DefaultColumnSelector()
        filler = DefaultFiller()

        cleaner = DataCleaner(data, column_selector, filler)
        cleaned_data = cleaner.clean_up()

        cat_cols = column_selector.get_categorical_columns(cleaned_data)
        # num_cols = column_selector.get_numerical_columns(data)

        encoded_data = LabelEncoderWrapper().apply_encoding(cleaned_data, cat_cols)
        scaled_data = MinMaxScalerWrapper().apply_scaler(
            encoded_data, encoded_data.columns.tolist()
        )
        return scaled_data

    def split_data(self, data):
        x_train, x_test, y_train, y_test = DefaultDataSplitting().data_split(
            data, target=TARGET
        )
        return x_train, x_test, y_train, y_test

    def train_and_evaluate(self, x_train, x_test, y_train, y_test):
        lr = LogisticRegressionWrapper()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)

        acc = F1SCORE()
        print(acc.evaluate(y_test, y_pred))


    def run(self):
        data = self.load_data()
        transformed_data = self.clean_and_transform(data)
        x_train, x_test, y_train, y_test = self.split_data(transformed_data)
        self.train_and_evaluate(x_train, x_test, y_train, y_test)
        self.logger.info("ML pipeline completed!")


if __name__ == "__main__":
    pipeline = MLPipeline(DATA_PATH, TARGET)
    pipeline.run()
