"""Configuration data structures for Vs30 calculations."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class GridConfig:
    """
    Grid domain and resolution parameters for raster-based Vs30 calculations.

    Defines the NZTM2000 (EPSG:2193) bounding box and pixel spacing for
    the output raster grid. Only used by the grid pipeline; the points
    pipeline does not need grid parameters.

    Attributes
    ----------
    grid_xmin : int
        Grid minimum X coordinate (NZTM, meters).
    grid_xmax : int
        Grid maximum X coordinate (NZTM, meters).
    grid_ymin : int
        Grid minimum Y coordinate (NZTM, meters).
    grid_ymax : int
        Grid maximum Y coordinate (NZTM, meters).
    grid_dx : int
        Grid X spacing (meters).
    grid_dy : int
        Grid Y spacing (meters).
    """

    grid_xmin: int
    grid_xmax: int
    grid_ymin: int
    grid_ymax: int
    grid_dx: int
    grid_dy: int

    @classmethod
    def from_yaml(cls, path: Path) -> "GridConfig":
        """
        Load grid configuration from a YAML file.

        The YAML file must contain keys: grid_xmin, grid_xmax, grid_ymin,
        grid_ymax, grid_dx, grid_dy. Other keys are ignored.

        Parameters
        ----------
        path : Path
            Path to the YAML configuration file.

        Returns
        -------
        GridConfig
            Grid configuration object.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "GridConfig":
        """
        Create a GridConfig from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing grid_xmin, grid_xmax, grid_ymin,
            grid_ymax, grid_dx, grid_dy keys.

        Returns
        -------
        GridConfig
            Grid configuration object.
        """
        return cls(
            grid_xmin=data["grid_xmin"],
            grid_xmax=data["grid_xmax"],
            grid_ymin=data["grid_ymin"],
            grid_ymax=data["grid_ymax"],
            grid_dx=data["grid_dx"],
            grid_dy=data["grid_dy"],
        )
