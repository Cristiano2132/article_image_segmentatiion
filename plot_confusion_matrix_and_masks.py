import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tools import create_rgb_img
from matplotlib.gridspec import GridSpec

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
    
    folder_path = 'data/input/'
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(nrows=2, ncols=3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    axes = [ax0, ax1, ax2, ax3, ax4, ax5]

    #  Matriz de confusao e máscaras
    for count,file in enumerate(glob.glob(folder_path + '/*.tif')):
        raster = rasterio.open(file)
        b = raster.bounds
        bounds_extent = np.asarray([b.left, b.right, b.bottom, b.top])
        red = raster.read(1)
        green = raster.read(2)
        blue = raster.read(3)
        create_rgb_img(red, green, blue, ax=axes[count], bounds_extent=bounds_extent)
        axes[count].axis('off')

    for i, label in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
        axes[i].text(-0.1, 1.15, label, transform=axes[i].transAxes,
        fontsize=16, fontweight='bold', va='top', ha='right')
    fig.savefig(f'data/output/fig_01.png', dpi=600)
    
