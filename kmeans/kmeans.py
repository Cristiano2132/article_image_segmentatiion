from sklearn.cluster import KMeans
import pandas as pd


class GetKmeans:
    def __init__(self, df: pd.DataFrame, n_cluster:int=2) -> None:
        self.__df = df
        self.n_cluster=2
    def get_kmeans(self):
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(self.__df)
        self.__df['cluster'] = model.labels_
        return self.__df['cluster']

