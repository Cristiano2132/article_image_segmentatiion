import numpy as np
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.ensemble import RandomForestClassifier
from .interface import RegressorInterface
import pandas as pd
import pickle


class RandomForest_(RegressorInterface):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, save_folder_path: str, random_state: int = 1):
        self.__random_state = random_state
        self.__X = X
        self.__y = y.values.ravel()
        self.__save_folder_path = save_folder_path

    def __get_space(self) -> list:
        space = [Integer(10, 200, name='n_estimators'),
                 Integer(1, 1000, name='min_samples_leaf'),
                 Integer(1, 100, name='max_depth'),
                 Categorical(['gini', 'entropy'], name='criterion')
                 ]

        return space

    def __get_space_params(self) -> list:
        space_params = ['n_estimators',
                        'min_samples_leaf', 'max_depth', 'criterion']
        return space_params

    def __make_objective_function(self) -> any:
        classifier = RandomForestClassifier(random_state=0, n_jobs=-1)
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
        dict_results['rf'] = {
            "Accuracy": res_gp.fun,
            "Best params": best_params
        }
        print(f'Trainin acuracy: {res_gp.fun}\nBest params: {best_params}')
        return dict_results

    def fit_regression(self):
        dict_results = self.__find_optimal_params()
        best_params = dict_results.get('rf').get('Best params')
        train_acuracy = dict_results.get('rf').get("Accuracy")
        best_model = RandomForestClassifier(n_estimators=best_params.get('n_estimators'),
                                            min_samples_leaf=best_params.get(
                                                'min_samples_leaf'),
                                            max_depth=best_params.get(
                                                'max_depth'),
                                            criterion=best_params.get(
                                                'criterion'),
                                            random_state=self.__random_state,
                                            n_jobs=-1)
        path_model_rf = f'{self.__save_folder_path}/rf_model.pkl'
        best_model.fit(self.__X, self.__y)
        pickle.dump(best_model, open(path_model_rf, "wb"))
        print(f"The model is training with the best parameters ......")
        print(f"Done!!! The model was trained with the best parameters !!!")
        return best_model, train_acuracy, best_params
