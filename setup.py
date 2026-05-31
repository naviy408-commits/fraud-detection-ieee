from setuptools import setup, find_packages

setup(
    name="fraud-detection-ieee",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "xgboost",
        "scikit-learn",
        "pandas",
        "numpy",
        "shap",
        "streamlit"
    ]
)
