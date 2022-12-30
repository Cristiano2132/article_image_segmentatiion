import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rasterio.plot import adjust_band
from rasterio.plot import reshape_as_image
from matplotlib.colors import ListedColormap



plt.style.use('seaborn-poster')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.edgecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['patch.edgecolor'] = '#ffffff'
plt.rcParams['patch.facecolor'] = '#ffffff'
plt.rcParams['savefig.edgecolor'] = '#ffffff'
plt.rcParams['savefig.facecolor'] = '#ffffff'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16


def create_rgb_img(red, green, blue, ax, bounds_extent):
    rgb = [red, green, blue]
    rgb_norm = adjust_band(rgb)  # normalize bands to range between 1.0 to 0.0
    rgb_reshaped = reshape_as_image(rgb_norm).astype('f4')  # reshape to [rows, cols, bands]
    rgb_reshaped = np.array(rgb_reshaped)
    masked = np.ma.masked_where(np.nan_to_num(red, nan=0) != 0, np.nan_to_num(red, nan=0))
    ax.imshow(rgb_reshaped, extent=bounds_extent)
    ax.imshow(masked, alpha=1, cmap=ListedColormap(['white']), extent=bounds_extent) #remove o fundo preto do rgb

def plot_mask(data_xr, mask:str, ax, red, green, blue, bounds_extent, title):
    where_is_0 = (data_xr[mask].values==0)
    data_xr2 = xr.where(
        where_is_0,data_xr, np.nan
        )
    create_rgb_img(red, green, blue, ax=ax, bounds_extent=bounds_extent)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(title)
    ax.imshow(data_xr2.cluster.values, cmap=ListedColormap([ 'black']),extent=bounds_extent, alpha=1)

