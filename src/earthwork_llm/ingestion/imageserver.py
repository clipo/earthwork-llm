"""Self-contained USGS 3DEP ImageServer bare-earth DEM fetch (any CRS).

Fetches a square bare-earth elevation window centred on a coordinate directly
from the seamless USGS 3D Elevation Program (3DEP) ImageServer — no API key, no
local data store. This is the single external dependency for the validation and
review tooling: everything downstream runs on the array it returns.
"""
from __future__ import annotations

import time

import numpy as np
import requests
from rasterio.io import MemoryFile

USGS_3DEP_IMAGESERVER = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
_NODATA = -999999.0


def fetch_dem(
    center_x: float,
    center_y: float,
    size_px: int,
    crs_epsg: int = 26915,
    resolution_m: float = 1.0,
    timeout: int = 30,
    retries: int = 5,
) -> np.ndarray:
    """Return a ``size_px`` × ``size_px`` bare-earth DEM (float32, NaN nodata)
    centred on ``(center_x, center_y)`` in the given CRS (EPSG code).

    Defaults to EPSG:26915 (UTM Zone 15N) at 1 m so distances are true metres.
    Transient ImageServer errors (502/timeout) are retried with backoff.
    """
    half = size_px * resolution_m / 2.0
    minx, miny = center_x - half, center_y - half
    maxx, maxy = center_x + half, center_y + half
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": crs_epsg,
        "imageSR": crs_epsg,
        "size": f"{size_px},{size_px}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": _NODATA,
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    for attempt in range(retries):
        try:
            raw = requests.get(USGS_3DEP_IMAGESERVER, params=params, timeout=timeout).content
            with MemoryFile(raw) as mf, mf.open() as ds:
                arr = ds.read(1).astype("float32")
            arr[arr == _NODATA] = np.nan
            arr[arr < -1e30] = np.nan
            return arr
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


# --- Compatibility shim -----------------------------------------------------
# A minimal drop-in for the window-request interface used by the validation
# and review scripts, so they depend only on this module.
from dataclasses import dataclass  # noqa: E402


@dataclass
class WindowRequest:
    """A square DEM window centered on a projected coordinate."""

    center_x: float
    center_y: float
    utm_crs: str          # e.g. "EPSG:26915"
    resolution_m: float
    size_px: int


class Usgs3depImageServerSource:
    """Thin object wrapper around :func:`fetch_dem`."""

    def fetch_window(self, req: "WindowRequest") -> np.ndarray:
        """Fetch the DEM window described by ``req`` via :func:`fetch_dem`."""
        epsg = int(str(req.utm_crs).split(":")[-1])
        return fetch_dem(
            req.center_x, req.center_y, req.size_px,
            crs_epsg=epsg, resolution_m=req.resolution_m,
        )
