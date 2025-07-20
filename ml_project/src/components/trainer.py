from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import joblib
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(X_train, y_train, X_test, y_test):
    """
    Trains a RandomForestClassifier with preprocessing and hyperparameter tuning.
    Logs experiment to MLflow and saves the best pipeline.
    """
    mlflow.set_experiment("Predictive_Maintenance_Model")

    # Identify feature types
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    logger.info(f"Numerical columns: {num_cols}")
    logger.info(f"Categorical columns: {cat_cols}")

    # Preprocessor: scale numeric + encode categorical
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Base model
    rf = RandomForestClassifier(random_state=42)

    # Pipeline: preprocess + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    # Hyperparameter tuning grid
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    with mlflow.start_run():
        logger.info("Starting training with hyperparameter tuning...")

        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        test_accuracy = best_pipeline.score(X_test, y_test)

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Test set accuracy: {test_accuracy:.4f}")

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.sklearn.log_model(best_pipeline, "model")

        # Save locally
        joblib.dump(best_pipeline, "artifacts/model.joblib")
        logger.info("Best pipeline saved to artifacts/model.joblib")
