"""
Configuration classes for MVN processing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Vs30MapConfig:
    """Configuration for VS30 map processing."""

    models_to_process: list[str]
    max_dist_m: float
    max_points: int
    cov_reduc: float
    noisy: bool
    phi_geology: float
    phi_terrain: float
    total_memory_gb: float
    compute_bayesian_update: bool  # If True, compute Bayesian update from observations; if False, use loaded CSV values directly
    observations_file: (
        str  # Path to observations CSV file (relative to resources directory)
    )
    output_dir: str  # Output directory for MVN updated rasters (relative to base_path)
    geology_mean_and_standard_deviation_per_category_file: (
        str  # Path to geology model CSV file (relative to resources directory)
    )
    terrain_mean_and_standard_deviation_per_category_file: (
        str  # Path to terrain model CSV file (relative to resources directory)
    )
    terrain_vs30_mean_stddev_filename: (
        str  # Output filename for terrain MVN updated raster
    )
    geology_vs30_mean_stddev_filename: (
        str  # Output filename for geology MVN updated raster
    )
    model_nodata: float  # NoData value for model rasters (from config.yaml)
    n_prior: int = 3  # For Bayesian updates
    min_sigma: float = 0.5  # Minimum stdv for Bayesian updates

    @classmethod
    def from_yaml(cls, path: Path) -> "Vs30MapConfig":
        """
        Load configuration from YAML file with validation.

        Parameters
        ----------
        path : Path
            Path to YAML configuration file.

        Returns
        -------
        Vs30MapConfig
            Validated configuration object.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises
        ------
        AssertionError
            If any configuration value is invalid.
        """
        assert self.max_dist_m > 0, "max_dist_m must be positive"
        assert self.max_points > 0, "max_points must be positive"
        assert self.cov_reduc >= 0, "cov_reduc must be non-negative"
        assert self.phi_geology > 0, "phi_geology must be positive"
        assert self.phi_terrain > 0, "phi_terrain must be positive"
        assert self.total_memory_gb > 0, "total_memory_gb must be positive"
        assert isinstance(self.compute_bayesian_update, bool), (
            "compute_bayesian_update must be a boolean"
        )
        assert all(m in ["geology", "terrain"] for m in self.models_to_process), (
            "models_to_process must contain only 'geology' and/or 'terrain'"
        )
        assert len(self.observations_file) > 0, "observations_file must be non-empty"
        assert len(self.output_dir) > 0, "output_dir must be non-empty"
        assert len(self.geology_mean_and_standard_deviation_per_category_file) > 0, (
            "geology_mean_and_standard_deviation_per_category_file must be non-empty"
        )
        assert len(self.terrain_mean_and_standard_deviation_per_category_file) > 0, (
            "terrain_mean_and_standard_deviation_per_category_file must be non-empty"
        )
        assert len(self.terrain_vs30_mean_stddev_filename) > 0, (
            "terrain_vs30_mean_stddev_filename must be non-empty"
        )
        assert len(self.geology_vs30_mean_stddev_filename) > 0, (
            "geology_vs30_mean_stddev_filename must be non-empty"
        )

    def get_phi(self, model_name: str) -> float:
        """
        Get phi value for specified model.

        Parameters
        ----------
        model_name : str
            Model name ("geology" or "terrain").

        Returns
        -------
        float
            Phi value for the model.

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        if model_name == "geology":
            return self.phi_geology
        elif model_name == "terrain":
            return self.phi_terrain
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_output_filename(self, model_name: str) -> str:
        """
        Get output filename for specified model.

        Parameters
        ----------
        model_name : str
            Model name ("geology" or "terrain").

        Returns
        -------
        str
            Output filename for the model.

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        if model_name == "geology":
            return self.geology_vs30_mean_stddev_filename
        elif model_name == "terrain":
            return self.terrain_vs30_mean_stddev_filename
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_model_id_column(self, model_name: str) -> str:
        """
        Get ID column name for specified model.

        Parameters
        ----------
        model_name : str
            Model name ("geology" or "terrain").

        Returns
        -------
        str
            ID column name ("gid" for geology, "tid" for terrain).

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        if model_name == "geology":
            return "gid"
        elif model_name == "terrain":
            return "tid"
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_model_csv_path(self, model_name: str) -> str:
        """
        Get CSV path for specified model.

        Parameters
        ----------
        model_name : str
            Model name ("geology" or "terrain").

        Returns
        -------
        str
            Path to model CSV file (relative to resources directory).

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        if model_name == "geology":
            return self.geology_mean_and_standard_deviation_per_category_file
        elif model_name == "terrain":
            return self.terrain_mean_and_standard_deviation_per_category_file
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_output_csv_path(
        self,
        model_name: str,
        base_path: Path,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Determine output CSV path for updated model values.

        Parameters
        ----------
        model_name : str
            Model name ("geology" or "terrain").
        base_path : Path
            Base path for finding input CSV files.
        output_dir : Path, optional
            Directory for output CSV files. If None, writes to same directory as input CSV
            with "updated_" prefix before filename.

        Returns
        -------
        Path
            Output CSV file path.

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        csv_path_rel = self.get_model_csv_path(model_name)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_filename = Path(csv_path_rel).name
            # Remove "updated_" prefix if present, then add it
            if csv_filename.endswith(".csv"):
                base_name = csv_filename[:-4]
                if base_name.startswith("updated_"):
                    base_name = base_name[8:]  # Remove "updated_" prefix
                return output_dir / f"updated_{base_name}.csv"
            else:
                return output_dir / csv_filename
        else:
            # Default: same directory as input with updated_ prefix
            resources_dir = base_path / "vs30" / "resources"
            input_path = resources_dir / csv_path_rel
            # Remove "updated_" prefix if already present
            stem = input_path.stem
            if stem.startswith("updated_"):
                stem = stem[8:]
            return input_path.parent / ("updated_" + stem + input_path.suffix)

    def get_model_raster_path(self, model_name: str, base_path: Path) -> Path:
        """
        Get raster path for specified model, generating it if needed.

        Parameters
        ----------
        model_name : str
            Model name ("geology" or "terrain").
        base_path : Path
            Base path for finding raster files.

        Returns
        -------
        Path
            Path to model raster file.

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        if model_name == "geology":
            return base_path / "vs30map" / "geology.tif"
        elif model_name == "terrain":
            raster_path = base_path / "vs30map" / "terrain.tif"
            # Generate terrain.tif if it doesn't exist
            if not raster_path.exists():
                csv_path = self.terrain_mean_and_standard_deviation_per_category_file
                _generate_terrain_raster(
                    raster_path, csv_path, base_path, self.model_nodata
                )
            return raster_path
        else:
            raise ValueError(f"Unknown model: {model_name}")


def _generate_terrain_raster(
    output_path: Path, csv_path: str, base_path: Path, model_nodata: float
) -> None:
    """
    Generate terrain.tif raster from IwahashiPike.tif and terrain category model values.

    This creates a 2-band GeoTIFF with vs30 (band 1) and stdv (band 2) by
    mapping terrain IDs from IwahashiPike.tif to terrain category model values.

    Parameters
    ----------
    output_path : Path
        Path where terrain.tif should be created.
    csv_path : str
        Path to terrain model CSV file (relative to resources directory).
    base_path : Path
        Base path for finding IwahashiPike.tif.
    model_nodata : float
        NoData value to use for output raster (from config).
    """
    import rasterio

    from vs30.model import load_model_values_from_csv

    # Get IwahashiPike.tif path
    iwahashi_path = base_path / "vs30" / "data" / "IwahashiPike.tif"
    if not iwahashi_path.exists():
        raise FileNotFoundError(
            f"IwahashiPike.tif not found at {iwahashi_path}. "
            "Cannot generate terrain.tif without source terrain ID raster."
        )

    # Load terrain IDs from IwahashiPike.tif
    with rasterio.open(iwahashi_path) as src:
        terrain_id_map = src.read(1)
        transform = src.transform
        crs = src.crs

    # Get model values from CSV
    # mean_and_stddev_vs30_per_category shape: (16, 2) where each row is [vs30_mean, vs30_stdv] for terrain IDs 1-16
    # Row 0 corresponds to terrain ID 1, row 1 to terrain ID 2, ..., row 15 to terrain ID 16
    mean_and_stddev_vs30_per_category = load_model_values_from_csv(csv_path)

    # Initialize output arrays with nodata value
    # vs30_array shape: same as terrain_id_map (e.g., (height, width))
    # stdv_array shape: same as terrain_id_map (e.g., (height, width))
    vs30_array = np.full_like(terrain_id_map, model_nodata, dtype=np.float64)
    stdv_array = np.full_like(terrain_id_map, model_nodata, dtype=np.float64)

    # Create boolean mask to identify valid terrain IDs in the map
    # Shape: (height, width) - boolean array, same shape as terrain_id_map
    # True = terrain ID is between 1 and len(mean_and_stddev_vs30_per_category) (inclusive), which means it's valid
    # False = terrain ID is outside range 1-16 (includes 0, negative values, nodata, values > 16)
    is_valid_terrain_id_in_map = (terrain_id_map >= 1) & (
        terrain_id_map <= len(mean_and_stddev_vs30_per_category)
    )

    # Extract the actual terrain ID values from valid pixels
    # Shape: (n_valid_pixels,) - 1D array of terrain ID integers (values 1-16)
    valid_terrain_ids = terrain_id_map[is_valid_terrain_id_in_map]

    # Convert terrain IDs to array indices for indexing into mean_and_stddev_vs30_per_category array
    # Terrain IDs are 1-indexed (1-16), but mean_and_stddev_vs30_per_category array rows are 0-indexed (0-15)
    # Shape: (n_valid_pixels,) - 1D array of array indices (values 0-15)
    # Example: terrain_id=1 -> index=0, terrain_id=2 -> index=1, terrain_id=16 -> index=15
    valid_terrain_ids_as_indices = valid_terrain_ids - 1

    # Map terrain IDs to vs30 values
    # NumPy boolean indexing: vs30_array[is_valid_terrain_id_in_map] creates a 1D view
    # of all pixels where the mask is True (in row-major order). The right-hand side
    # must have the same number of elements, which it does because:
    # - is_valid_terrain_id_in_map has n_valid_pixels True values
    # - valid_terrain_ids_as_indices has n_valid_pixels elements
    # - mean_and_stddev_vs30_per_category[valid_terrain_ids_as_indices, 0] extracts n_valid_pixels values
    # NumPy assigns them element-by-element: first True pixel gets first value, etc.
    vs30_array[is_valid_terrain_id_in_map] = mean_and_stddev_vs30_per_category[
        valid_terrain_ids_as_indices, 0
    ]

    # Map terrain IDs to stdv values using same boolean indexing as before
    stdv_array[is_valid_terrain_id_in_map] = mean_and_stddev_vs30_per_category[
        valid_terrain_ids_as_indices, 1
    ]

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write 2-band GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=terrain_id_map.shape[0],
        width=terrain_id_map.shape[1],
        count=2,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=model_nodata,
        compress="deflate",
        tiled=True,
        bigtiff="yes",
    ) as dst:
        dst.write(vs30_array, 1)
        dst.write(stdv_array, 2)
        dst.set_band_description(1, "Vs30")
        dst.set_band_description(2, "Standard Deviation")
