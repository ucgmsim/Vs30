"""
Configuration classes for MVN processing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class MVNConfig:
    """Configuration for MVN processing."""

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
    def from_yaml(cls, path: Path) -> "MVNConfig":
        """
        Load configuration from YAML file with validation.

        Parameters
        ----------
        path : Path
            Path to YAML configuration file.

        Returns
        -------
        MVNConfig
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


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    name: str
    phi: float
    prior_values: np.ndarray  # (n_categories, 2) array of [vs30, stdv]
    raster_path: Path
    id_column: str  # Column name for model ID in observations

    @classmethod
    def from_mvn_config(
        cls, mvn_config: MVNConfig, model_name: str, base_path: Path
    ) -> "ModelConfig":
        """
        Create ModelConfig from MVNConfig and model name.

        Parameters
        ----------
        mvn_config : MVNConfig
            MVN configuration object.
        model_name : str
            Model name ("geology" or "terrain").
        base_path : Path
            Base path for finding raster files.

        Returns
        -------
        ModelConfig
            Configured model configuration object.
        """
        # Import here to avoid circular imports
        from vs30.model import load_model_values_from_csv

        if model_name == "geology":
            id_column = "gid"
            raster_path = base_path / "vs30map" / "geology.tif"
            csv_path = mvn_config.geology_mean_and_standard_deviation_per_category_file
        elif model_name == "terrain":
            id_column = "tid"
            raster_path = base_path / "vs30map" / "terrain.tif"
            csv_path = mvn_config.terrain_mean_and_standard_deviation_per_category_file

            # Generate terrain.tif if it doesn't exist
            if not raster_path.exists():
                _generate_terrain_raster(
                    raster_path, csv_path, base_path, mvn_config.model_nodata
                )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load model values from CSV file specified in config
        prior_values = load_model_values_from_csv(csv_path)

        return cls(
            name=model_name,
            phi=mvn_config.get_phi(model_name),
            prior_values=prior_values,
            raster_path=raster_path,
            id_column=id_column,
        )


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
