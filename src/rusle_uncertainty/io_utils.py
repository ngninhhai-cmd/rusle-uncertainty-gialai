"""Raster I/O and reprojection utilities."""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def read_raster(path, nodata=None):
    """Read raster as float32 with NoData converted to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        arr = np.where(arr < -1e30, np.nan, arr)
        return arr, src.profile.copy(), src.transform, src.crs, src.shape


def write_raster(path, arr, profile, nodata=-9999):
    """Write raster with NoData handling."""
    profile = profile.copy()
    profile.update(dtype="float32", nodata=nodata, compress="LZW")
    arr_out = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_out, 1)
    return path


def resample_to_reference(src_arr, src_transform, src_crs,
                          ref_profile, ref_shape,
                          method=Resampling.bilinear):
    """Resample array to match a reference profile."""
    dst_arr = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr, destination=dst_arr,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
        resampling=method,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst_arr


def get_pixel_area_ha(path):
    """Return pixel area in hectares."""
    with rasterio.open(path) as src:
        px = abs(src.transform[0])
        py = abs(src.transform[4])
        return (px * py) / 10_000