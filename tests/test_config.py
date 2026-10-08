"""Tests for model config loading and validation."""

import pytest
import yaml

from vs30 import config, constants

BUNDLED_YAML = constants.MODEL_VERSION_TO_CONFIG[
    constants.FixedModelVersion.FOSTER_2019_APPROX
]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"combination_method": "ratoi"}, "combination_method must be one of"),
        ({"combine_ratio": None}, "combine_ratio must be a number >= 0"),
        ({"combine_ratio": -0.5}, "combine_ratio must be a number >= 0"),
        ({"mvn": "false"}, "mvn must be true or false"),
        ({"noisey": False}, "unknown field.*noisey"),
        ({"geology_correlation": {"model": "exponential"}}, "geology_correlation.*phi"),
        (
            {"independent_observations_csv": "./my_obs.csv"},
            "bundled file name or an absolute path",
        ),
        ({"independent_observations_csv": "/nonexistent/my_obs.csv"}, "file not found"),
        ({"independent_observations_csv": None}, "need at least one observations CSV"),
    ],
)
def test_bad_config_values_are_rejected_when_loading(tmp_path, changes, message):
    """A bad value in a custom config raises a ValueError naming the problem, before any computation."""
    config_data = yaml.safe_load(
        BUNDLED_YAML.read_text(encoding=constants.DEFAULT_TEXT_ENCODING)
    )
    config_data.update(changes)
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(yaml.safe_dump(config_data))

    with pytest.raises(ValueError, match=message):
        config.load_config_from_yaml(custom_yaml)


def test_empty_config_is_rejected(tmp_path):
    """An empty config file raises a ValueError rather than a TypeError."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")

    with pytest.raises(ValueError, match="is empty"):
        config.load_config_from_yaml(empty_yaml)
