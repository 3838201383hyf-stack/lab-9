<<<<<<< HEAD
import joblib
import os

def test_model_exists():
    assert os.path.exists('assets/model.pkl'), "Model file not found"

def test_accuracy_threshold():
    with open('assets/accuracy.txt', 'r') as f:
        accuracy = float(f.read().strip())
    assert accuracy >= 0.90, f"Accuracy {accuracy} < 0.90"

def test_model_can_predict():
    model = joblib.load('assets/model.pkl')
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    assert prediction[0] in [0, 1, 2], "Invalid prediction"
=======
import json
import os
from pathlib import Path

import joblib


OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", "assets"))
MINIMUM_ACCURACY = float(os.getenv("MINIMUM_ACCURACY", "0.90"))


def load_metrics():
    metrics_path = OUTPUT_DIR / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def test_expected_artifacts_exist():
    expected_files = [
        OUTPUT_DIR / "iris_model.joblib",
        OUTPUT_DIR / "metrics.json",
        OUTPUT_DIR / "metrics.txt",
        OUTPUT_DIR / "confusion_matrix.png",
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Missing expected file: {file_path}"


def test_model_can_be_loaded():
    model_path = OUTPUT_DIR / "iris_model.joblib"
    model = joblib.load(model_path)
    assert hasattr(model, "predict")


def test_model_accuracy_is_high_enough():
    metrics = load_metrics()
    assert metrics["accuracy"] >= MINIMUM_ACCURACY


def test_confusion_matrix_shape():
    metrics = load_metrics()
    matrix = metrics["confusion_matrix"]
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    
>>>>>>> 503b77ed186bb3c706aa379cf9205ae5f342d33d
