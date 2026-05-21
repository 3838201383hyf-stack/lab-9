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