"""
Install script for the VsViewer package.

This installs the vs_calc library. The vs_api web service is packaged
separately (see vs_api/setup.py and vs_api/requirements.txt).
"""

from setuptools import setup

setup(
    name="VsViewer",
    version="1.0",
    # vs_calc.scripts has no __init__.py, so it must be listed explicitly
    packages=["vs_calc", "vs_calc.scripts"],
    url="https://github.com/ucgmsim/Vs30",
    description="Vs30 Web Calculator",
    install_requires=["numpy", "pandas"],
    # Only the vs_calc.scripts.plot_* scripts use matplotlib
    extras_require={"plot": ["matplotlib"]},
)
