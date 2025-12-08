"""
Data classes for MVN processing.
"""

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio import transform as rasterio_transform


@dataclass
class ObservationData:
    """Bundled observation data for MVN processing."""

    locations: np.ndarray  # (n_obs, 2) array of [easting, northing]
    vs30: np.ndarray  # (n_obs,) measured vs30 values
    model_vs30: np.ndarray  # (n_obs,) model vs30 at observation locations
    model_stdv: np.ndarray  # (n_obs,) model stdv at observation locations
    residuals: np.ndarray  # (n_obs,) log residuals: log(vs30 / model_vs30)
    omega: np.ndarray  # (n_obs,) noise weights (if noisy=True)
    uncertainty: np.ndarray  # (n_obs,) observation uncertainties

    def __post_init__(self) -> None:
        """
        Validate data consistency.

        Raises
        ------
        AssertionError
            If array lengths don't match.
        """
        n = len(self.locations)
        assert len(self.vs30) == n, "vs30 length must match locations"
        assert len(self.model_vs30) == n, "model_vs30 length must match locations"
        assert len(self.model_stdv) == n, "model_stdv length must match locations"
        assert len(self.residuals) == n, "residuals length must match locations"
        assert len(self.omega) == n, "omega length must match locations"
        assert len(self.uncertainty) == n, "uncertainty length must match locations"


@dataclass
class PixelData:
    """Data for a single pixel being updated."""

    location: np.ndarray  # (2,) [easting, northing]
    vs30: float
    stdv: float
    index: int  # Index in full raster (for mapping back)


@dataclass
class MVNUpdateResult:
    """Result of MVN update for a single pixel."""

    updated_vs30: float
    updated_stdv: float
    n_observations_used: int
    min_distance: float
    pixel_index: int  # For mapping back to raster


@dataclass
class BoundingBoxResult:
    """Result of bounding box calculation."""

    mask: np.ndarray  # Boolean mask of affected pixels
    obs_to_grid_indices: list[np.ndarray]  # List of grid indices per observation
    n_affected_pixels: int


@dataclass
class RasterData:
    """Raster data wrapper using rasterio."""

    vs30: np.ndarray  # Band 1
    stdv: np.ndarray  # Band 2
    transform: rasterio.transform.Affine
    crs: rasterio.crs.CRS
    nodata: float | None
    valid_mask: np.ndarray  # Boolean mask of valid pixels
    valid_flat_indices: np.ndarray  # Flat indices of valid pixels

    @classmethod
    def from_file(cls, path: Path) -> "RasterData":
        """
        Load raster data from file.

        Parameters
        ----------
        path : Path
            Path to GeoTIFF file.

        Returns
        -------
        RasterData
            Loaded raster data object.
        """
        with rasterio.open(path) as src:
            vs30 = src.read(1)
            stdv = src.read(2)
            nodata = src.nodatavals[0] if src.nodatavals else None

            if nodata is not None:
                valid_mask = vs30 != nodata
            else:
                valid_mask = ~np.isnan(vs30)

            valid_flat_indices = np.where(valid_mask.flatten())[0]

            return cls(
                vs30=vs30,
                stdv=stdv,
                transform=src.transform,
                crs=src.crs,
                nodata=nodata,
                valid_mask=valid_mask,
                valid_flat_indices=valid_flat_indices,
            )

    def get_coordinates(self) -> np.ndarray:
        """
        Get coordinates for valid pixels.

        Returns
        -------
        ndarray
            (N_valid, 2) array of [easting, northing] coordinates.
        """
        valid_rows, valid_cols = np.where(self.valid_mask)
        rows_valid = valid_rows.astype(float) + 0.5
        cols_valid = valid_cols.astype(float) + 0.5
        xs, ys = rasterio_transform.xy(self.transform, rows_valid, cols_valid)
        return np.column_stack((np.array(xs), np.array(ys))).astype(np.float64)

    def write_updated(
        self, path: Path, updated_vs30: np.ndarray, updated_stdv: np.ndarray
    ) -> None:
        """
        Write updated raster to file.

        Parameters
        ----------
        path : Path
            Output file path.
        updated_vs30 : ndarray
            Updated vs30 values (1D array for valid pixels only).
        updated_stdv : ndarray
            Updated stdv values (1D array for valid pixels only).
        """
        # Initialize with original values
        output_vs30 = self.vs30.copy()
        output_stdv = self.stdv.copy()

        # Update only modified pixels
        output_vs30.flat[self.valid_flat_indices] = updated_vs30
        output_stdv.flat[self.valid_flat_indices] = updated_stdv

        # Write using rasterio
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=self.vs30.shape[0],
            width=self.vs30.shape[1],
            count=2,
            dtype=self.vs30.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
        ) as dst:
            dst.write(output_vs30, 1)
            dst.write(output_stdv, 2)

