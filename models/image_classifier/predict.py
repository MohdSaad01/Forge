import sys
from pathlib import Path

import forge
from forge.exceptions import DataError


MODEL_PATH = Path(__file__).parent / "image_model.forge"


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    predictor = forge.load_predictor(MODEL_PATH)

    try:
        result = predictor.predict(image_path)
    except DataError:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)

    print("Forge Image Classifier")
    print("──────────────────────")
    print(f"Image: {image_path}")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.1%}")


if __name__ == "__main__":
    main()