"""
Pytest configuration and shared test utilities.

This module contains fixtures and helper functions used across multiple test files.
"""

import vs30.config


def reset_default_config() -> None:
    """
    Reset the cached default configuration.

    Call this if you need to reload the default config from disk
    (e.g., after modifying config.yaml during testing).
    """
    vs30.config._default_config = None
