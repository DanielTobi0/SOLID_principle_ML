"""
following the SOLID priciple, of assigning a single classs to a single task
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class ColumnSelector(ABC):
    """abstract class for columns selection"""
    @abstractmethod
    def get_categorical_columns(self, df: pd.DataFrame) -> List:
        pass

    @abstractmethod
    def get_numerical_columns(self, df: pd.DataFrame) -> List:
        pass


class DefaultColumnSelector(ColumnSelector):
    def get_categorical_columns(self, df: pd.DataFrame) -> List:
        """Get categorical features"""
        return df.select_dtypes(include=["object"]).columns.tolist()

    def get_numerical_columns(self, df: pd.DataFrame) -> List:
        """Get numerical features"""
        return df.select_dtypes(include=["number"]).columns.tolist()


class MissingValueFiller(ABC):
    @abstractmethod
    def fill(self, df: pd.DataFrame, columns: List, strategy: str) -> pd.DataFrame:
        pass


class DefaultFiller(MissingValueFiller):
    def __init__(self):
        self.num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

    def fill(self, df: pd.DataFrame, columns: List, strategy: str) -> pd.DataFrame:
        """fill missing values in the specified columns based on strategy."""
        if not columns:
            return df

        if strategy == "numerical" and columns:
            # print(f"numerical columns: {columns}")
            df[columns] = self.num_imputer.fit_transform(df[columns])
        
        elif strategy == "categorical" and columns:
            # print(f"object columns: {columns}")
            df[columns] = self.cat_imputer.fit_transform(df[columns])

        return df


class DataCleaner:
    def __init__(
        self,
        df: pd.DataFrame,
        column_selector: ColumnSelector,
        filler: MissingValueFiller,
    ):
        self.df = df
        self.column_selector = column_selector
        self.filler = filler

    def clean_up(self) -> pd.DataFrame:
        """Clean up missing values in categorical and numerical columns"""
        cat_cols = self.column_selector.get_categorical_columns(self.df)
        num_cols = self.column_selector.get_numerical_columns(self.df)

        self.df = self.filler.fill(self.df, num_cols, strategy="numerical")
        self.df = self.filler.fill(self.df, cat_cols, strategy="categorical")

        return self.df


if __name__ == "__main__":
    from config.settings import data_path
    from data.loader import DataLoader, CSVReader

    csv_reader = CSVReader()
    csv_loader = DataLoader(csv_reader)
    data = csv_loader.load_data(data_path)

    column_selector = DefaultColumnSelector()
    filler = DefaultFiller()

    cleaner = DataCleaner(data, column_selector, filler)
    cleaned_data = cleaner.clean_up()
    print(cleaned_data)