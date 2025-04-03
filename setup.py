"""Setup script for the BB ML Pipeline package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bb_ml_pipeline",
    version="0.1.0",
    author="BB ML Pipeline Team",
    author_email="example@example.com",
    description="A pipeline for training, unboxing, and predicting LGBM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bb_ml_pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scikit-learn>=1.1.1",
        "lightgbm>=4.0.0",
        "optuna>=3.0.0",
        "shap>=0.40.0",
        "treeinterpreter>=0.2.2",
    ],
    entry_points={
        "console_scripts": [
            "bb_ml_pipeline=bb_ml_pipeline.__main__:main",
        ],
    },
) 