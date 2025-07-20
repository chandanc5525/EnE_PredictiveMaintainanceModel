
from setuptools import setup, find_packages

setup(
    name="Predictive Maintainance Model",
    version="1.0.0",
    description="A predictive maintenance model that leverages historical sensor data and maintenance records to forecast potential equipment failures. This helps in planning maintenance proactively, minimizing unplanned downtime, and improving operational efficiency.",
    author="",
    author_email="chaudhari.chandan22@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["pandas", "scikit-learn", "joblib", "mlflow"],
)
