# Forge

**Forge** is a general deep learning framework for building, training, evaluating, and deploying neural-network models. It provides flexible data pipelines, model persistence, CPU/CUDA execution, and performance benchmarking with the core ML machinery implemented from the ground up.

## Status
Early development. See `docs/development/roadmap.md`.

## Philosophy
Forge is intended to be a real, usable framework rather than a wrapper around an existing deep-learning framework. It uses established numerical infrastructure where appropriate while keeping core ML machinery inside Forge.

## Command-line interface
A thin `forge`/`python -m forge` CLI exposes model/checkpoint inspection and
device conversion, and the existing benchmark suite, over Forge's public
persistence APIs (no new framework logic). See `docs/development/cli.md`.

## Development
Read `CLAUDE.md` and the relevant `docs/` files before contributing.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
