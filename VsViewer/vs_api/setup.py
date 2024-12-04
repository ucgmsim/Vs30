"""
Install script for the vs_api package.
"""

from setuptools import setup, find_packages

setup(
    name="vs_api",
    version="1.0",
    packages=find_packages(),
    url="https://github.com/ucgmsim/Vs30",
    description="VsViewer API",
    install_requires=["flask", "flask_cors", "numpy"],
)
