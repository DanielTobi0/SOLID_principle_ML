"""
ML model evaluation on data
"""

from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score, f1_score


class BaseEvaluation(ABC):
    """Abstraction for evaluation"""

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass


class Accuracy(BaseEvaluation):
    """check for accuracy"""

    def evaluate(self, y_true, y_pred):
        return f"Accuracy score: {accuracy_score(y_true, y_pred)}"


class F1SCORE(BaseEvaluation):
    """f1 score"""

    def evaluate(self, y_true, y_pred):
        return f"F1 Score: {f1_score(y_true, y_pred)}"


if __name__ == "__main__":
    from data.loader import DataLoader, CSVReader
    from data.cleaner import DefaultFiller, DataCleaner, DefaultColumnSelector
    from data.transformation import (
        LabelEncoderWrapper,
        MinMaxScalerWrapper
    )
    from config.settings import TARGET, DATA_PATH
    from data.splitting import DefaultDataSplitting
    from models.logistic_regression import LogisticRegressionWrapper

    csv_reader = CSVReader()
    csv_loader = DataLoader(csv_reader)
    data = csv_loader.load_data(DATA_PATH)

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

    x_train, x_test, y_train, y_test = DefaultDataSplitting().data_split(
        cleaned_data, target=TARGET
    )

    lr = LogisticRegressionWrapper()
    lr.fit(x_train, y_train)
    y_pred_ = lr.predict(x_test)

    acc = F1SCORE()
    print(acc.evaluate(y_test, y_pred_))
