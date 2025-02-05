"""
Data transformation, which contains normalization of categorical variables
"""

from abc import abstractmethod, ABC
from typing import List
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
)


class Scaler(ABC):
    """Abstract base class for data scaling"""

    @abstractmethod
    def apply_scaler(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        pass


class StandardScalerWrapper(Scaler):
    def apply_scaler(self, df, columns: List[str] = None) -> pd.DataFrame:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df


class MinMaxScalerWrapper(Scaler):
    def apply_scaler(self, df, columns: List[str] = None) -> pd.DataFrame:
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df


class Encoder(ABC):
    """Abstract base class for data scaling"""

    @abstractmethod
    def apply_encoding(
            self, df: pd.DataFrame, columns: List[str] = None
    ) -> pd.DataFrame:
        pass


class LabelEncoderWrapper(Encoder):
    def apply_encoding(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        if columns:
            encoder = LabelEncoder()
            for col in columns:
                df[col] = encoder.fit_transform(df[col])
        return df


class GetDummiesWrapper(Encoder):
    def apply_encoding(self, df, columns=None):
        if columns:
            df[columns] = pd.get_dummies(df[columns], dtype=int, drop_first=True)
        return df


if __name__ == "__main__":
    from config.settings import DATA_PATH
    from loader import DataLoader, CSVReader
    from cleaner import DefaultFiller, DataCleaner, DefaultColumnSelector

    csv_reader = CSVReader()
    csv_loader = DataLoader(csv_reader)
    data = csv_loader.load_data(DATA_PATH)

    column_selector = DefaultColumnSelector()
    filler = DefaultFiller()

    cleaner = DataCleaner(data, column_selector, filler)
    cleaned_data = cleaner.clean_up()

    cat_cols = column_selector.get_categorical_columns(data)
    num_cols = column_selector.get_numerical_columns(data)

    encoder_ = LabelEncoderWrapper().apply_encoding(cleaned_data, cat_cols)
    scaler_ = StandardScalerWrapper().apply_scaler(cleaned_data, cleaned_data.columns.tolist())

    print(cleaned_data)
