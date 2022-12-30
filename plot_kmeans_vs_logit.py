import glob
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import rasterio
from tools.confuion_matrix_for_grid import My_Confusion_Matrix
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from tools import LogitClassifier, RasterTools, create_rgb_img, plot_mask
from kmeans import ClassifyVegetationKmeans

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


if __name__ == '__main__':
    df_results = pd.DataFrame(columns=['logit_classify','kmeans_classify_rgb', 'kmeans_classify_indices'])
    model_path = 'data/output/rgb_indices/logit_model.pkl'
    folder_path = 'data/input/'


#  Matriz de confusao e máscaras
    for count,file in enumerate(glob.glob(folder_path + '/*.tif')):
        raster = rasterio.open(file)
        b = raster.bounds
        bounds_extent = np.asarray([b.left, b.right, b.bottom, b.top])
        red = raster.read(1)
        green = raster.read(2)
        blue = raster.read(3)
        tool = RasterTools(raster_path=file)
        df = tool.structur_dataframe()
        df['cluster'] = ClassifyVegetationKmeans(df=df, get_indices=True).rate_vagetation()
        list_indices = ['MGVRI', 'GLI', 'MPRI', 'RGVBI', 'ExG', 'VEG']
        list_bands = ['blue', 'green', 'red']
        df['logit_classify'] = LogitClassifier(
            model_path=model_path, df=df[list_indices]).rate_vagetation()
        
    
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(nrows=3, ncols=2, width_ratios=[
                      1, 1], height_ratios=[5, 5, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1:2, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 1])

        cf_matrix = confusion_matrix(df.cluster, df.logit_classify)
        group_names = ['TruePositive', 'FalseNegative', 'FalsePositive', 'TrueNegative']

        ax0.text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax0.transAxes, size=20, weight='bold')
        ax1.text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax1.transAxes, size=20, weight='bold')
        ax2.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax2.transAxes, size=20, weight='bold')
        ax3.text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax3.transAxes, size=20, weight='bold')
        cm = My_Confusion_Matrix(
            ax0=ax3, ax1=ax4,
            cf=cf_matrix,
            group_names=group_names,
            cmap='Blues',
            figsize=(10, 12),
            categories=['Vegetation', 'Soil'],

        )
        cm.xlabel = 'K-$means^+$ Label'
        cm.ylabel = 'Logistic Regression Label'
        cm.make_confusion_matrix()
        dict_replace = {'Vegetation': 1, 'Soil': 0}
        df.replace({'cluster': dict_replace,
                   'logit_classify': dict_replace}, inplace=True)
        data_xr = df[['red', 'green', 'blue',
                      'cluster', 'logit_classify']].to_xarray()
        attrs = xr.open_rasterio(file).attrs
        data_xr.attrs = attrs
        data_xr = data_xr.reindex(y=list(reversed(data_xr.y)))
        plot_mask(data_xr, mask='cluster', ax=ax0, red=red, green=green,
                  blue=blue, bounds_extent=bounds_extent, title='K-$means^+$ mask')
        plot_mask(data_xr, mask='logit_classify', ax=ax1, red=red, green=green,
                  blue=blue, bounds_extent=bounds_extent, title='Logistic Regression mask')
        create_rgb_img(red, green, blue, ax=ax2, bounds_extent=bounds_extent)
        ax4.xaxis.set_visible(False)
        ax4.yaxis.set_visible(False)
        ax2.set_title("Original image")
        plt.tight_layout()
        ax2.set_axis_off()
        fig.savefig(f'data/output/fig_kmeans/kmeans_vs_logit_{count}.png', dpi=100)
    
