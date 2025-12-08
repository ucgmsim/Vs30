"""
Configuration classes for MVN processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from vs30.mvn_data import RasterData


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
    update_mode: str  # "prior", "posterior_paper", "computed", or "custom"
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
        assert self.update_mode in [
            "prior",
            "posterior_paper",
            "computed",
            "custom",
        ], f"update_mode must be one of: prior, posterior_paper, computed, custom (got {self.update_mode})"
        assert all(
            m in ["geology", "terrain"] for m in self.models_to_process
        ), "models_to_process must contain only 'geology' and/or 'terrain'"

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
        if model_name == "geology":
            import vs30.model_geology as model_module

            id_column = "gid"
            raster_path = base_path / "vs30map" / "geology.tif"
        elif model_name == "terrain":
            import vs30.model_terrain as model_module

            id_column = "tid"
            raster_path = base_path / "vs30map" / "terrain.tif"
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load prior values
        prior_values = model_module.model_prior()

        return cls(
            name=model_name,
            phi=mvn_config.get_phi(model_name),
            prior_values=prior_values,
            raster_path=raster_path,
            id_column=id_column,
        )

