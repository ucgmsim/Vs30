"""
Model registry for standardized model interfaces.

This module provides a unified interface for geology and terrain models,
allowing them to be used interchangeably.
"""

import importlib
from dataclasses import dataclass
from typing import Callable

import numpy as np


# Standard column name used for model IDs in DataFrames
STANDARD_ID_COLUMN = "model_id"


@dataclass
class ModelInfo:
    """Information about a model type."""
    
    name: str
    module_name: str
    csv_config_key: str  # Key in config for CSV path
    
    def get_module(self):
        """Import and return the model module."""
        return importlib.import_module(self.module_name)
    
    def get_model_id_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get the model_id function from the module."""
        module = self.get_module()
        return module.model_id


# Registry mapping model names to their configurations
MODEL_REGISTRY = {
    "geology": ModelInfo(
        name="geology",
        module_name="vs30.model_geology",
        csv_config_key="geology_mean_and_standard_deviation_per_category_file",
    ),
    "terrain": ModelInfo(
        name="terrain",
        module_name="vs30.model_terrain",
        csv_config_key="terrain_mean_and_standard_deviation_per_category_file",
    ),
}


def get_model_info(model_name: str) -> ModelInfo:
    """
    Get model information from registry.
    
    Parameters
    ----------
    model_name : str
        Model name ("geology" or "terrain").
    
    Returns
    -------
    ModelInfo
        Model information object.
    
    Raises
    ------
    ValueError
        If model_name is not recognized.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]

