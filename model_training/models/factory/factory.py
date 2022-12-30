from typing import Type
from ..interface import RegressorInterface
from ..rf import RandomForest_
from ..xgb import XGB_
from ..logit import Logit_



class Factory():
    def __init__(self, classificator_name: str, X, y, save_folder_path:str) -> None:
        self.__classificator_name = classificator_name
        self.__X = X
        self.__y = y
        self.__save_folder_path=save_folder_path

    def create_regressor(self) -> Type[RegressorInterface]:
        print(f'Regressor name: {self.__classificator_name}')
        return self.__select_regressor(regressor_name=self.__classificator_name)

    def __load_xgb_regressor(self) -> any:
        regressor = XGB_(X=self.__X, y=self.__y, save_folder_path=self.__save_folder_path)
        return regressor

    def __load_logit_regressor(self) -> any:
        regressor = Logit_(X=self.__X, y=self.__y, save_folder_path=self.__save_folder_path)
        return regressor

    def __load_rf_regressor(self) -> any:
        regressor = RandomForest_(X=self.__X, y=self.__y, save_folder_path=self.__save_folder_path)
        return regressor

    def __select_regressor(self, regressor_name: str) -> Type[RegressorInterface]:
        switcher = {
            'rf': self.__load_rf_regressor,
            'xgb': self.__load_xgb_regressor,
            'logit': self.__load_logit_regressor,
        }
        regressor = switcher.get(regressor_name)
        return regressor()
