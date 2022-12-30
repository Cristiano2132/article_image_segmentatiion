import numpy as np
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from xgboost import XGBClassifier
from .interface import RegressorInterface
import pandas as pd
import pickle


class XGB_(RegressorInterface):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, save_folder_path: str, random_state: int = 1):
        self.__random_state = random_state
        self.__X = X
        self.__y = y
        self.__save_folder_path = save_folder_path

    def __get_space(self) -> list:
        space = [Integer(1, 5, name='max_depth'),
                 Real(10 ** -5, 10 ** 0, "log-uniform", name='learning_rate')]
        return space

    def __get_space_params(self) -> list:
        space_params = ['max_depth', 'learning_rate']
        return space_params

    def __make_objective_function(self) -> any:
        classifier = XGBClassifier(random_state=self.__random_state,
                                   eval_metric='logloss',
                                   use_label_encoder=False,
                                   num_parallel_tree=5
                                   )
        space = self.__get_space()

        @use_named_args(space)
        def objective(**params):
            classifier.set_params(**params)
            return -np.mean(cross_val_score(classifier, self.__X, self.__y, cv=3, n_jobs=-1,
                                            scoring="roc_auc"))
        return objective

    def __find_optimal_params(self) -> dict:
        print(f"Wait: Finding the best parameters .....")
        obj = self.__make_objective_function()
        space = self.__get_space()
        space_params = self.__get_space_params()
        res_gp = gp_minimize(func=obj, dimensions=space,
                             n_calls=20, random_state=self.__random_state)
        best_params = dict(zip(space_params, res_gp.x))
        dict_results = {}
        dict_results['xgb'] = {
            "Accuracy": res_gp.fun,
            "Best params": best_params
        }
        print(f'Trainin acuracy: {res_gp.fun}\nBest params: {best_params}')
        return dict_results

    def fit_regression(self):
        dict_results = self.__find_optimal_params()
        best_params = dict_results.get('xgb').get('Best params')
        train_acuracy = dict_results.get('xgb').get("Accuracy")
        best_model = XGBClassifier(max_depth=best_params.get('max_depth'),
                                   learning_rate=best_params.get(
                                       'learning_rate'),
                                   eval_metric='logloss',
                                   use_label_encoder=False,
                                   num_parallel_tree=5
                                   )
        path_model_xgb = f'{self.__save_folder_path}/xgb_model.pkl'
        best_model.fit(self.__X, self.__y)
        pickle.dump(best_model, open(path_model_xgb, "wb"))
        print(f"The model is training with the best parameters ......")
        print(f"Done!!! The model was trained with the best parameters !!!")
        return best_model, train_acuracy, best_params
