from .kmeans import GetKmeans
import pandas as pd


class ClassifyVegetationKmeans:
    def __init__(self, df: pd.DataFrame, get_indices = True) -> None:
        self.__df = df
        self.__n_cluster=2
        self.__get_indices = get_indices

    def rate_vagetation(self):
        if self.__get_indices:
            self.__df['cluster']=GetKmeans(df=self.__df[['MGVRI', 'GLI', 'MPRI', 'RGVBI', 'ExG', 'VEG']], n_cluster=self.__n_cluster).get_kmeans()
        else:
            self.__df['cluster']=GetKmeans(df=self.__df[['blue', 'green', 'red']], n_cluster=self.__n_cluster).get_kmeans()
        df_resume = self.__df.groupby(['cluster'])[['green']].mean()
        idx_max = df_resume['green'].idxmax()
        class_vegetation = df_resume.reset_index().iloc[idx_max].cluster
        idx_min = df_resume['green'].idxmin()
        class_soil = df_resume.reset_index().iloc[idx_min].cluster
        dict_replace = {class_vegetation: 'Vegetation', class_soil: 'Soil'}
        self.__df.replace({'cluster': dict_replace}, inplace=True)
        return self.__df['cluster']
