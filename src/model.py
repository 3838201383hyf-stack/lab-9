<<<<<<< HEAD
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('assets', exist_ok=True)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, 'assets/model.pkl')

with open('assets/accuracy.txt', 'w') as f:
    f.write(f"{accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('assets/confusion_matrix.png')

print(f"Model trained with accuracy: {accuracy:.4f}")
print("Files saved to assets/")
=======
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model on the Iris dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Directory where the model, metrics, and confusion matrix are saved.",
    )
    return parser.parse_args()


def train_model(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=1,
        stratify=iris.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    matrix = confusion_matrix(y_test, predictions)

    model_path = output_path / "iris_model.joblib"
    image_path = output_path / "confusion_matrix.png"
    metrics_json_path = output_path / "metrics.json"
    metrics_text_path = output_path / "metrics.txt"

    joblib.dump(model, model_path)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=iris.target_names,
    )
    figure, axis = plt.subplots(figsize=(6, 4.5))
    display.plot(ax=axis, cmap=plt.cm.Blues, colorbar=False)
    axis.set_title("Random Forest on Iris")
    figure.tight_layout()
    figure.savefig(image_path)
    plt.close(figure)

    metrics = {
        "model_name": "Random Forest Classifier",
        "dataset_name": "Iris",
        "accuracy": round(float(accuracy), 4),
        "accuracy_display": f"{accuracy:.2f}",
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "confusion_matrix": matrix.tolist(),
        "class_names": iris.target_names.tolist(),
    }

    metrics_json_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    metrics_text_path.write_text(
        "\n".join(
            [
                f"Model: {metrics['model_name']}",
                f"Dataset: {metrics['dataset_name']}",
                f"Accuracy: {metrics['accuracy_display']}",
                f"Train samples: {metrics['train_samples']}",
                f"Test samples: {metrics['test_samples']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Model saved to {model_path}")
    print(f"Accuracy: {metrics['accuracy_display']}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args.output_dir)
>>>>>>> 503b77ed186bb3c706aa379cf9205ae5f342d33d
