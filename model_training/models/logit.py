from sklearn.linear_model import LogisticRegression
from .interface import RegressorInterface
import pandas as pd
import pickle


class Logit_(RegressorInterface):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, save_folder_path: str, random_state: int = 1):
        self.__random_state = random_state
        self.__X = X
        self.__y = y.values.ravel()
        self.__save_folder_path = save_folder_path

    def fit_regression(self):
        dict_results = {}
        logit = LogisticRegression()
        logit.fit(self.__X, self.__y)
        dict_results['logit'] = {
            "Accuracy": None,
            "Best params": logit.coef_
        }

        path_model= f'{self.__save_folder_path}/logit_model.pkl'
        pickle.dump(logit, open(path_model, "wb"))
        print(f"The model is training ......")
        print(f"Done!!! The model was trained!!!")
        return logit, None, None
