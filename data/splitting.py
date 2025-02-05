"""
Data splitting
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitting(ABC):
    """Abstraction class for data splitting"""

    @abstractmethod
    def data_split(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]],
        target: str = "",
        test_size=0.2,
        random_state=0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


class DefaultDataSplitting(DataSplitting):
    """create train test split"""

    def data_split(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        target: str = "",
        test_size=0.2,
        random_state=0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if columns is None:
            columns = [col for col in df.columns if col != target]

        x_train, x_test, y_train, y_test = train_test_split(
            df[columns],
            df[target],
            test_size=test_size,
            random_state=random_state,
            stratify=df[target],
        )
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    from loader import DataLoader, CSVReader
    from cleaner import DefaultFiller, DataCleaner,DefaultColumnSelector
    from transformation import (
        LabelEncoderWrapper,
        StandardScalerWrapper,
    )
    from config.settings import target, data_path

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
    scaler = StandardScalerWrapper().apply_scaler(
        cleaned_data, cleaned_data.columns.tolist()
    )

    # print(target)
    # print(cleaned_data)
    x_train, x_test, y_train, y_test = DefaultDataSplitting().data_split(
        cleaned_data, target=target
    )
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
