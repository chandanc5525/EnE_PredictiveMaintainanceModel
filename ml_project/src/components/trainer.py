from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes or clips outliers in numeric columns using the IQR method.
    Clips values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X).copy()
        for col in X_.select_dtypes(include='number').columns:
            q1 = X_[col].quantile(0.25)
            q3 = X_[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.multiplier * iqr
            upper = q3 + self.multiplier * iqr
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            X_[col] = X_[col].clip(lower, upper)
        return X_


def train(X_train, y_train, X_test, y_test):
    """
    Trains a RandomForestClassifier with preprocessing, outlier handling, and hyperparameter tuning.
    Logs experiment to MLflow, saves the best pipeline, and logs training time.
    """
    mlflow.set_experiment("Predictive_Maintenance_Model")

    # Identify feature types
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    logger.info(f"Numerical columns: {num_cols}")
    logger.info(f"Categorical columns: {cat_cols}")

    # Preprocessor: outlier removal + scale numeric + encode categorical
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("outlier_remover", OutlierRemover()),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Base model
    rf = RandomForestClassifier(random_state=42)

    # Pipeline: preprocess + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    # Extended hyperparameter tuning grid
    param_grid = {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__bootstrap": [True, False],
        "classifier__criterion": ["gini", "entropy"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    with mlflow.start_run():
        logger.info("Starting training with outlier handling and hyperparameter tuning...")

        start_time = time.time()

        grid_search.fit(X_train, y_train)

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        test_accuracy = best_pipeline.score(X_test, y_test)

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Test set accuracy: {test_accuracy:.4f}")

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("training_time_seconds", elapsed_time)
        mlflow.sklearn.log_model(best_pipeline, "model")

        # Save locally
        joblib.dump(best_pipeline, "artifacts/model.joblib")
        logger.info("Best pipeline saved to artifacts/model.joblib")

    logger.info("Training pipeline finished.")
