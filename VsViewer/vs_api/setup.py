"""
Install script for the vs_api package.
"""
from setuptools import setup

setup(
    name="vs_api",
    version="1.0",
    packages=["vs_api"],
    url="https://github.com/ucgmsim/Vs30",
    description="VsViewer API",
    install_requires=["GDAL", "numpy", "pandas", "pyproj", "scikit-learn", "scipy", "matplotlib"],
)
