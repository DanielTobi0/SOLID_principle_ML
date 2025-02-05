"""
ML Models 
"""

from abc import abstractmethod, ABC
import pandas as pd
from sklearn.linear_model import LogisticRegression
from models.base import BaseModel


class LogisticRegressionWrapper(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, x, y):
        return self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == "__main__":
    from data.loader import DataLoader, CSVReader
    from data.cleaner import DefaultFiller, DataCleaner, DefaultColumnSelector
    from data.transformation import (
        LabelEncoderWrapper,
        StandardScalerWrapper,
        MinMaxScalerWrapper
    )
    from config.settings import target, data_path
    from data.splitting import DefaultDataSplitting

    csv_reader = CSVReader()
    csv_loader = DataLoader(csv_reader)
    data = csv_loader.load_data(data_path)

    column_selector = DefaultColumnSelector()
    filler = DefaultFiller()

    cleaner = DataCleaner(data, column_selector, filler)
    cleaned_data = cleaner.clean_up()

    cat_cols = column_selector.get_categorical_columns(data)
    num_cols = column_selector.get_numerical_columns(data)

    encoder = LabelEncoderWrapper().apply_encoding(cleaned_data, cat_cols)
    scaler = MinMaxScalerWrapper().apply_scaler(
        cleaned_data, cleaned_data.columns.tolist()
    )

    # print(target)
    # print(cleaned_data)
    x_train, x_test, y_train, y_test = DefaultDataSplitting().data_split(
        cleaned_data, target=target
    )
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    lr = LogisticRegressionWrapper()
    lr.fit(x_train, y_train)
    print(lr.predict(x_test))
