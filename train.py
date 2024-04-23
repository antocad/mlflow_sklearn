import click
import mlflow.projects.env_type
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

import sklearn
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


@click.command()
@click.option("--tracking_uri", default="", type=str)
def train(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    # We create the remote experiment (on the server, if it doesn't already exist)
    experiment = mlflow.set_experiment("test-sklearn-without-docker")
    # We create a run locally with the same experiment id as the one on the server
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id = run.info.run_id):
        # # Log artifacts
        # mlflow.log_artifact("conda.yaml")
        # mlflow.log_artifact("Dockerfile")
        # mlflow.log_artifact("MLproject")
        # mlflow.log_artifact("project.py")

        # Loading dataset
        dia_df = load_diabetes()
        X = dia_df.data
        y = dia_df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Normalizing y
        y_min = y_train.min()
        y_max = y_train.max()
        y_train = (y_max-y_train)/(y_max-y_min)
        y_test = (y_max-y_test)/(y_max-y_min)

        # Creating a pipeline
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', SVR()),
        ])

        # Training model
        pipeline.fit(X_train, y_train)

        # Evaluating model
        y_pred = pipeline.predict(X_test)
        y_pred = y_pred*(y_max - y_min) + y_min
        y_test = y_test*(y_max - y_min) + y_min
        rmse = np.sqrt(((y_test - y_pred)**2).mean())

        # Log materials to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_params({
            "y_min": y_min,
            "y_max": y_max
        })
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

if __name__ == "__main__":
    train()
