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
@click.option("--run", default="", type=str)
@click.option("--input", default="", type=str)
def inference(tracking_uri, run, input):
    mlflow.set_tracking_uri(tracking_uri)
    # Extracting Hparams and model's checkpoint from MLflow
    run_id = run.split('/')[1]
    run_mlflow = dict(mlflow.get_run(run_id))
    y_max = float(run_mlflow['data'].params['y_max'])
    y_min = float(run_mlflow['data'].params['y_min'])
    pipeline = mlflow.pyfunc.load_model(run)

    # Inference
    input = np.fromstring(input, sep=',').reshape(1, -1)
    y_pred = pipeline.predict(input)
    y_pred = y_pred*(y_max-y_min) + y_min
    print("X:", input)
    print("pred:", y_pred[0])


if __name__ == "__main__":
    inference()
