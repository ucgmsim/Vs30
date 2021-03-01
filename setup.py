import os
from setuptools import setup
import sys

for shape in ("coast", "qmap"):
    if not os.path.isdir(
        os.path.join(os.path.dirname(__file__), "vs30", "data", shape)
    ):
        sys.exit("please extract vs30/data/shapefiles.tar.xz first")

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
