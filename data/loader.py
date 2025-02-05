"""
Data loader class
"""

from abc import abstractmethod, ABC
import pandas as pd


class IDataReader(ABC):
    """abstraction to load data from file"""

    @abstractmethod
    def read(self, path: str):
        pass


class CSVReader(IDataReader):
    """read csv file"""

    def read(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"File not found: {path}.")
        except pd.errors.EmptyDataError:
            print(f"The file at {path} is empty.")
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")

    
class ExcelReader(IDataReader):
    """read excel file"""

    def read(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_excel(path)
        except FileNotFoundError:
            print(f"File not found: {path}.")
        except pd.errors.EmptyDataError:
            print(f"The file at {path} is empty.")
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")


class DataLoader:
    def __init__(self, reader: IDataReader):
        self.reader = reader

    def load_data(self, path: str) -> pd.DataFrame:
        return self.reader.read(path)


if __name__ == "__main__":
    from config.settings import data_path
    # data_path = (
    #     "C:/Users/HomePC/Desktop/learn/SOLID principle ML/data/raw/Titanic-Dataset.csv"
    # )
    
    csv_reader = CSVReader()
    csv_loader = DataLoader(csv_reader)
    print(csv_loader.load_data(data_path))
