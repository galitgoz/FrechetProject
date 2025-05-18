# setup.py
from setuptools import setup, find_packages

setup(
    name="frechet",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        # …any other deps
    ],
)
