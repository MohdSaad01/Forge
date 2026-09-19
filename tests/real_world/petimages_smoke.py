"""Real-world smoke test: ImageFolder -> train_image_classifier() -> artifact -> inference.

The maintenance-mode acceptance check described in
`docs/development/maintenance.md` (section 3). It is a standalone script, not a
pytest test: `tests/` collects only `test_*.py`, so this is never part of
`python -m pytest tests/` or CI. It needs a real cat/dog dataset that Forge
does not ship -- the ~25,000-image "PetImages" layout, `<root>/<class>/<n>.jpg`
with two classes -- which stays outside the repository.

```bash
python tests/real_world/petimages_smoke.py /path/to/petimages
# or:  FORGE_PETIMAGES_DIR=/path/to/petimages python tests/real_world/petimages_smoke.py
```

If no dataset directory is given, or it does not exist, the script says so and
exits 0 (a skip, not a pass and not a failure). Anything else that goes wrong
exits non-zero.

It copies a deterministic 160-image subset (the first 80 files of each class in
sorted filename order) into a temporary directory, trains 2 epochs on the CPU,
and checks that the artifact saves/verifies and that inference on one held-out
image per class (the file right after that class's subset) returns a valid
`ClassificationPrediction`. It is not an accuracy benchmark: the subset and
epoch count are chosen for speed, so the accuracy it reports means nothing.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import forge

PER_CLASS = 80
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _class_files(class_dir: Path) -> "list[Path]":
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in EXTENSIONS)


def main(argv: "list[str]") -> int:
    raw = argv[1] if len(argv) > 1 else os.environ.get("FORGE_PETIMAGES_DIR")
    if not raw or not Path(raw).is_dir():
        where = f"'{raw}'" if raw else "(none given)"
        print(f"SKIP: petimages dataset not found at {where}. "
              "Pass its directory as the first argument or set FORGE_PETIMAGES_DIR.")
        return 0

    root = Path(raw)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if len(class_dirs) != 2:
        print(f"FAIL: expected exactly 2 class directories under {root}, found {len(class_dirs)}.")
        return 1

    with tempfile.TemporaryDirectory(prefix="forge_petimages_smoke_") as tmp:
        tmp = Path(tmp)
        held_out = []
        for class_dir in class_dirs:
            files = _class_files(class_dir)
            if len(files) <= PER_CLASS:
                print(f"FAIL: class '{class_dir.name}' has only {len(files)} images; need more than {PER_CLASS}.")
                return 1
            target = tmp / "data" / class_dir.name
            target.mkdir(parents=True)
            for f in files[:PER_CLASS]:
                shutil.copy2(f, target / f.name)
            held_out.append((class_dir.name, files[PER_CLASS]))

        artifact = tmp / "smoke.forge"
        start = time.perf_counter()
        result = forge.train_image_classifier(
            tmp / "data", path=str(artifact), epochs=2, batch_size=16,
            image_size=(64, 64), device="cpu", seed=0, verbose=False,
        )
        elapsed = time.perf_counter() - start

        assert artifact.is_file(), "artifact was not written"
        assert result.classes == [d.name for d in class_dirs], result.classes
        assert result.dataset_size + len(result.skipped_images) == 2 * PER_CLASS, (
            result.dataset_size, len(result.skipped_images))

        info = forge.inspect_model(str(artifact))
        assert info.task == "classification" and info.classes == result.classes, info

        predictor = forge.load_predictor(str(artifact), device="cpu")
        for class_name, image in held_out:
            prediction = predictor.predict(image)
            assert prediction.label in result.classes, prediction
            assert 0.0 <= prediction.confidence <= 1.0, prediction
            assert forge.predict_model(str(artifact), image, device="cpu").label == prediction.label
            print(f"held-out {class_name}/{image.name}: {prediction.label} ({prediction.confidence:.1%})")

    print(f"OK: trained {result.dataset_size} images ({len(result.skipped_images)} skipped), "
          f"2 epochs on CPU in {elapsed:.1f}s, artifact verified, inference valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
