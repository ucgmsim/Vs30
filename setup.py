"""
Install script for the vs30 package.
recommended install method:
pip install --user .
"""
import os
from shutil import rmtree
import tarfile

from setuptools import setup

repo_data = os.path.join(os.path.dirname(__file__), "vs30", "data")
# remove old versions of shapefiles
# these won't exist when using pip because it copies repo into temp
for shape in ("coast", "qmap"):
    full_path = os.path.join(repo_data, shape)
    if os.path.isdir(full_path):
        rmtree(full_path)
# extract new versions
with tarfile.open(os.path.join(repo_data, "shapefiles.tar.xz")) as xz:
    xz.extractall(repo_data)

setup(
    name="Vs30",
    version="2.0",
    packages=["vs30"],
    url="https://github.com/ucgmsim/Vs30",
    description="NZ Vs30 Calculation",
    package_data={
        "vs30": [
            "data/*.csv",
            "data/*.ll",
            "data/*.ssv",
            "data/*.tif",
            "data/*.qgz",
            "data/coast/*",
            "data/qmap/*",
        ]
    },
    install_requires=["GDAL", "numpy", "pandas", "pyproj", "scikit-learn", "scipy"],
    scripts=["vs30calc.py"],
)
