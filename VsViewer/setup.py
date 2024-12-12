"""
Install script for the VsViewer package.
"""

from setuptools import setup

setup(
    name="VsViewer",
    version="1.0",
    packages=["vs_calc", "vs_api"],
    url="https://github.com/ucgmsim/Vs30",
    description="Vs30 Web Calculator",
    install_requires=["numpy", "pandas", "pyyaml", "flask", "flask_cors", "uwsgi"],
)
