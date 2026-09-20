import sys
from pathlib import Path

import forge
from forge.exceptions import DataError


MODEL_PATH = Path(__file__).parent / "diabetes_classifier.forge"

# The artifact records how many features it needs, not what they are called;
# these names (the Pima Indians Diabetes columns, in training order) are only
# for the help text.
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
EXAMPLE = "6 148 72 35 0 33.6 0.627 50"


def usage():
    print("Usage: python predict.py <8 numbers>   (space- or comma-separated)")
    print("Features, in order: " + ", ".join(FEATURES))
    print("Raw values as in the dataset: 0 in Glucose/BloodPressure/SkinThickness/Insulin/BMI means")
    print("'not measured' and is handled by the model file itself.")
    print(f"Example: python predict.py {EXAMPLE}")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage()

    try:
        row = [float(v) for v in " ".join(sys.argv[1:]).replace(",", " ").split()]
    except ValueError:
        print("Error: every input must be a number.")
        usage()

    # The artifact was trained on the CPU, but choose the device explicitly:
    # Forge never moves a saved model between devices implicitly.
    predictor = forge.load_predictor(MODEL_PATH, device="cpu")

    expected = predictor.input_schema.feature_count
    if len(row) != expected:
        print(f"Error: expected {expected} numbers, got {len(row)}.")
        usage()

    try:
        result = predictor.predict([row])[0]
    except DataError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("Forge Tabular Classifier")
    print("------------------------")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.1%}")
    print("(A demonstration of the artifact workflow, not a medical tool.)")


if __name__ == "__main__":
    main()
