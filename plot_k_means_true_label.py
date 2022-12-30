import string
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import ClassifyVegetationKmeans
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from tools.confuion_matrix_for_grid import My_Confusion_Matrix

plt.style.use('seaborn-poster')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.edgecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['patch.edgecolor'] = '#ffffff'
plt.rcParams['patch.facecolor'] = '#ffffff'
plt.rcParams['savefig.edgecolor'] = '#ffffff'
plt.rcParams['savefig.facecolor'] = '#ffffff'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def structurer_data(path_data: str, indices=False):
    df = pd.read_csv(path_data, index_col=0)
    dic_replace = {'Vegetacao': 0, 'Solo': 1}
    df['target'] = df['target'].replace(dic_replace)
    df = df.set_index(['x', 'y']).dropna()
    df = df.assign(MGVRI=lambda x: (x.green ** 2 - x.red ** 2) / (x.green ** 2 + x.red ** 2))
    df = df.assign(GLI=lambda x: (2 * x.green - x.red - x.blue) / (2 * x.green + x.red + x.blue))
    df = df.assign(MPRI=lambda x: (x.green - x.red) / (x.green + x.red))
    df = df.assign(RGVBI=lambda x: (x.green - x.red * x.blue) / (x.green ** 2 + x.red * x.blue))
    df = df.assign(ExG=lambda x: (2 * x.green - x.red - x.blue) / (x.green + x.red + x.blue))
    df = df.assign(VEG=lambda x: (x.green) / (x.red ** 0.667 + x.blue ** (1 - 0.667)))
    df = df.dropna()
    y = df['target']
    X = df[['MGVRI', 'GLI', 'MPRI', 'RGVBI', 'ExG', 'VEG', 'red', 'blue', 'green']]
    return X, y

if __name__ == '__main__':
    path_data = 'data/output/all_targets.csv'
    X, y = structurer_data(path_data=path_data, indices=True)
    teste = [True, False]
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=2, height_ratios=[5, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    my_axes = [ax0, ax1, ax2, ax3]
    for n,condition in enumerate(teste):
        my_axes[n*2].text(-0.1, 1.1, string.ascii_uppercase[n], transform=my_axes[n*2].transAxes, 
            size=20, weight='bold')
        y_pred_kmeans_indice = ClassifyVegetationKmeans(df=X, get_indices=condition).rate_vagetation()
        dict_replace = {'Vegetation': 0, 'Soil': 1}
        y_pred_kmeans_indice.replace(dict_replace, inplace=True)

        cf_matrix = confusion_matrix(y, y_pred_kmeans_indice)
        group_names = ['TruePositive', 'FalseNegative', 'FalsePositive', 'TrueNegative']


        cm = My_Confusion_Matrix(
            ax0=my_axes[n*2], ax1=my_axes[n*2+1],
            cf=cf_matrix,
            group_names=group_names,
            cmap='Blues',
            figsize=(10, 12),
            categories=['Vegetation', 'Soil'],

        )
        if condition:
            cm.xlabel = 'K-means Clusters (indexes)'
        else:
            cm.xlabel = 'K-means Clusters (rgb)'
        cm.ylabel = 'True Label'
        cm.make_confusion_matrix()
        plt.tight_layout()
    plt.show()
    fig.savefig(f'data/output/fig_kmeans/kmeans_cm.png', dpi=600)

    