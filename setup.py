from setuptools import setup

setup(
    name="Vs30",
    version="2.0",
    packages=["vs30"],
    url="https://github.com/ucgmsim/Vs30",
    description="NZ Vs30 Calculation",
    package_data={"vs30": ["data/*.csv", "data/*.ll", "data/*.ssv"]},
    include_package_data=True,
    install_requires=["numpy", "osgeo", "pandas", "pyproj", "scipy", "sklearn"],
    scripts=["vs30calc.py"],
)
