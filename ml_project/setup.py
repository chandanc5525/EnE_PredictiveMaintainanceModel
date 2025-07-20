
from setuptools import setup, find_packages

setup(
    name="Predictive Maintaince MachineLearning Model",
    version="1.0.0",
    description="Predictive Model Its Multivariate Classification Model",
    author="chandan chaudhari",
    author_email="chaudhari.chandan22@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["pandas", "scikit-learn", "joblib", "mlflow","matplotlib","seaborn","loguru","pycaret"],
)
