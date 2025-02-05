from abc import abstractmethod, ABC
import pandas as pd


class BaseModel(ABC):
    """Base abstraction for ML models"""

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame):
        pass