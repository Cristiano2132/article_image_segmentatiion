import pickle
import pandas as pd

class LogitClassifier():
    def __init__(self, df: pd.DataFrame, model_path: str):
        self.__df = df
        self.__model_path = model_path

    def rate_vagetation(self):
        df_xr = self.__df
        model_loaded = pickle.load(open(self.__model_path, "rb"))
        df_xr['mask'] = model_loaded.predict(df_xr)
        dict_replace = {0:'Vegetation', 1:'Soil'}
        df_xr.replace({'mask':dict_replace}, inplace=True)
        return df_xr['mask']
