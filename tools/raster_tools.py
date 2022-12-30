import numpy as np
import numpy as np
import xarray as xr
import rasterio

class RasterTools():
    def __init__(self, raster_path: str):
        self.__raster_path = raster_path

    def get_metadata(self):
        raster = rasterio.open(self.__raster_path)
        out_meta = raster.meta.copy()
        return out_meta

    def structur_dataframe(self):
        data_xr = xr.open_rasterio(self.__raster_path)
        attrs = data_xr.attrs
        data_xr = xr.Dataset({'values': data_xr})
        dict_names = {1: 'red', 2: 'green', 3: 'blue'}
        df_xr = data_xr.to_dataframe().reset_index().replace(
            {'band': dict_names}).set_index(['band', 'y', 'x'])
        df_xr = df_xr.pivot_table(
            index=['y', 'x'], columns='band', values='values').rename_axis(columns=None)
        df_xr = df_xr.reset_index().set_index(['y', 'x'])
        mask_is_nan = (df_xr.red == 0)
        df_xr[mask_is_nan] = np.nan
        df_xr = df_xr.assign(MGVRI=lambda x: (
            x.green ** 2 - x.red ** 2) / (x.green ** 2 + x.red ** 2))
        df_xr = df_xr.assign(GLI=lambda x: (
            2 * x.green - x.red - x.blue) / (2 * x.green + x.red + x.blue))
        df_xr = df_xr.assign(MPRI=lambda x: (
            x.green - x.red) / (x.green + x.red))
        df_xr = df_xr.assign(RGVBI=lambda x: (
            x.green - x.red * x.blue) / (x.green ** 2 + x.red * x.blue))
        df_xr = df_xr.assign(ExG=lambda x: (
            2 * x.green - x.red - x.blue) / (x.green + x.red + x.blue))
        df_xr = df_xr.assign(VEG=lambda x: (x.green) /
                             (x.red ** 0.667 + x.blue ** (1 - 0.667)))
        return df_xr.dropna()
