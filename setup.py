"""
Custom build script for shapefile extraction.
"""

import os
from shutil import rmtree
import tarfile

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithShapefiles(build_py):
    def run(self):
        repo_data = os.path.join(os.path.dirname(__file__), "vs30", "data")
        # remove old versions of shapefiles
        for shape in ("coast", "qmap"):
            full_path = os.path.join(repo_data, shape)
            if os.path.isdir(full_path):
                rmtree(full_path)
        # extract new versions
        with tarfile.open(os.path.join(repo_data, "shapefiles.tar.xz")) as xz:
            xz.extractall(repo_data)
        super().run()


setup(
    cmdclass={"build_py": BuildWithShapefiles},
)
