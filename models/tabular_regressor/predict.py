import sys
from pathlib import Path

import forge
from forge.exceptions import DataError


MODEL_PATH = Path(__file__).parent / "concrete_strength_regressor.forge"

# The artifact records how many features it needs, not what they are called;
# these names (the UCI Concrete Compressive Strength inputs, in training order)
# are only for the help text.
FEATURES = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
    "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age",
]
EXAMPLE = "540 0 0 162 2.5 1040 676 28"


def usage():
    print("Usage: python predict.py <8 numbers>   (space- or comma-separated)")
    print("Features, in order: " + ", ".join(FEATURES))
    print("Ingredients in kg per m^3 of concrete; Age in days.")
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
        result = predictor.predict([row])
    except DataError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("Forge Tabular Regressor")
    print("-----------------------")
    print(f"Predicted compressive strength: {float(result.numpy()[0, 0]):.1f} MPa")
    print("(A demonstration of the artifact workflow, not an engineering estimate.)")


if __name__ == "__main__":
    main()
