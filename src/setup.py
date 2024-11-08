# setup.py
from setuptools import setup, find_packages

setup(
    name="olive_oil_dashboard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "tensorflow",
        "scikit-learn",
        # aggiungi altre dipendenze necessarie
    ]
)