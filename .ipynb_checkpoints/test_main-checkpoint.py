import pytest
import joblib
import pandas as pd
from sklearn.metrics import recall_score
import numpy as np

DATA_FILE = "./data.csv"
MODEL_FILE = "./downloaded_models/model/model.pkl"
TARGET_COLUMN = "species"


@pytest.fixture(scope="session")
def input_dataset():
    raw_data = pd.read_csv(DATA_FILE)
    column_mapping = {
        "sepal_length": "sepal length (cm)",
        "sepal_width": "sepal width (cm)",
        "petal_length": "petal length (cm)",
        "petal_width": "petal width (cm)",
    }
    raw_data.rename(columns={
        "sepal_length": "sepal length (cm)",
        "sepal_width": "sepal width (cm)",
        "petal_length": "petal length (cm)",
        "petal_width": "petal width (cm)",
    }, inplace=True)
    
    return raw_data

@pytest.fixture(scope="session")
def inference_model():
    loaded_model = joblib.load(MODEL_FILE)
    return loaded_model

def test_data_integrity_check(input_dataset):
    required_features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    features_present = all(col in input_dataset.columns for col in required_features)
    assert features_present is True

def test_model_performance_metric(inference_model, input_dataset):
    df_features = input_dataset.drop(TARGET_COLUMN, axis=1, errors='ignore')
    target_labels = input_dataset[TARGET_COLUMN]
    predictions = inference_model.predict(df_features)
    retrieval_rate = recall_score(target_labels, predictions, average='macro')
    assert retrieval_rate > 0.85
