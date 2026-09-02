"""Milestone 19 tests: the `forge` command-line interface (`forge/cli/`).

Covers `--help`, model/checkpoint inspection (text and `--json`), model/
checkpoint device conversion (CPU<->CPU always; CPU<->CUDA when CUDA is
available), invalid-input error handling and exit codes, the read-only/
no-CUDA-required/no-RNG-mutation guarantees of `inspect`, and the
`forge benchmark` pass-through. Invokes `forge.cli.main.main()` directly
(no subprocess) -- exactly the function `python -m forge` and the installed
`forge` console script both call. See `docs/development/cli.md`.
"""

from __future__ import annotations

import json
import zipfile

import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.cli.main import main
from forge.nn import Linear, ReLU, Sequential
from forge.optim import SGD, Adam
from forge.serialization.archive import METADATA_ENTRY


# -- fixtures -----------------------------------------------------------


def _build_model() -> Sequential:
    forge.random.seed(0)
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 3))


@pytest.fixture()
def model_path(tmp_path):
    path = tmp_path / "model.forge"
    forge.save_model(_build_model(), str(path))
    return path


@pytest.fixture()
def checkpoint_path(tmp_path):
    model = _build_model()
    optimizer = Adam(model.parameters(), lr=0.01)
    x = forge.Tensor([[1.0, 2.0, 3.0, 4.0]])
    optimizer.zero_grad()
    out = model(x)
    out.sum().backward()
    optimizer.step()

    path = tmp_path / "ckpt.forge"
    forge.save_checkpoint(str(path), model, optimizer, epoch=2, global_step=10)
    return path


def _read_metadata(path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read(METADATA_ENTRY))


def _write_metadata(path, metadata: dict) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries[METADATA_ENTRY] = json.dumps(metadata).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


# -- help / exit codes ----------------------------------------------------


def test_top_level_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "model" in out and "checkpoint" in out and "benchmark" in out


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main(["nosuchcommand"])
    assert exc_info.value.code != 0


def test_missing_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main(["model"])
    assert exc_info.value.code != 0


# -- model inspect --------------------------------------------------------


def test_model_inspect_text_output(model_path, capsys):
    code = main(["model", "inspect", str(model_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Model: {model_path}" in out
    assert "Device: cpu" in out
    assert "Training: train" in out
    assert "Sequential" in out and "Linear" in out and "ReLU" in out
    assert "0.weight" in out and "shape=(4, 8)" in out
    assert "Total parameters: 67" in out


def test_model_inspect_json_output(model_path, capsys):
    code = main(["model", "inspect", str(model_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu"
    assert payload["training"] == "train"
    assert payload["total_parameters"] == 67
    names = {p["name"] for p in payload["parameters"]}
    assert names == {"0.weight", "0.bias", "2.weight", "2.bias"}
    types = {m["type"] for m in payload["modules"]}
    assert types == {"Sequential", "Linear", "ReLU"}


def test_model_inspect_is_read_only(model_path):
    """Inspecting must not mutate the saved file on disk."""
    before = model_path.read_bytes()
    main(["model", "inspect", str(model_path)])
    after = model_path.read_bytes()
    assert before == after


def test_model_inspect_missing_file_returns_nonzero(tmp_path, capsys):
    code = main(["model", "inspect", str(tmp_path / "nope.forge")])
    assert code != 0
    assert "Error" in capsys.readouterr().err


def test_model_inspect_malformed_archive_returns_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.forge"
    bad.write_bytes(b"not a zip file")
    code = main(["model", "inspect", str(bad)])
    assert code != 0
    assert "Error" in capsys.readouterr().err


def test_model_inspect_unsupported_format_version_returns_nonzero(model_path, capsys):
    metadata = _read_metadata(model_path)
    metadata["forge_format_version"] = 999
    _write_metadata(model_path, metadata)
    code = main(["model", "inspect", str(model_path)])
    assert code != 0
    assert "version" in capsys.readouterr().err


def test_model_inspect_reports_recorded_cuda_device_without_cuda_hardware(model_path, monkeypatch, capsys):
    """A CUDA-recorded archive must inspect cleanly even with CUDA unavailable."""
    metadata = _read_metadata(model_path)
    metadata["device"] = "cuda"
    _write_metadata(model_path, metadata)

    import forge.backend.cuda as cuda_module

    def _fail_if_called():
        raise AssertionError("inspect must never probe CUDA availability")

    monkeypatch.setattr(cuda_module, "is_cuda_available", _fail_if_called)

    code = main(["model", "inspect", str(model_path)])
    assert code == 0
    assert "Device: cuda" in capsys.readouterr().out


# -- checkpoint inspect -----------------------------------------------------


def test_checkpoint_inspect_text_output(checkpoint_path, capsys):
    code = main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Checkpoint: {checkpoint_path}" in out
    assert "Device: cpu" in out
    assert "Epoch: 2" in out
    assert "Global step: 10" in out
    assert "RNG state: present" in out
    assert "Optimizer: Adam" in out
    assert "lr: 0.01" in out
    assert "Optimizer parameters: 4 total, 4 with saved state" in out
    assert "Total parameters: 67" in out


def test_checkpoint_inspect_json_output(checkpoint_path, capsys):
    code = main(["checkpoint", "inspect", str(checkpoint_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["optimizer"]["type"] == "Adam"
    assert payload["epoch"] == 2
    assert payload["global_step"] == 10
    assert payload["rng_state_present"] is True
    assert payload["optimizer_parameters_total"] == 4
    assert payload["optimizer_parameters_with_state"] == 4


def test_checkpoint_inspect_does_not_mutate_rng_state(checkpoint_path):
    """The hard read-only requirement: `load_checkpoint()` restores RNG state as a
    documented side effect -- `inspect` must never call it for exactly this reason."""
    forge.random.seed(12345)
    before = forge.random.get_state()
    code = main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    after = forge.random.get_state()
    assert before == after


def test_checkpoint_inspect_missing_file_returns_nonzero(tmp_path, capsys):
    code = main(["checkpoint", "inspect", str(tmp_path / "nope.forge")])
    assert code != 0
    assert "Error" in capsys.readouterr().err


def test_checkpoint_inspect_malformed_archive_returns_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.forge"
    bad.write_bytes(b"garbage")
    code = main(["checkpoint", "inspect", str(bad)])
    assert code != 0


def test_checkpoint_inspect_no_saved_state_before_any_step(tmp_path, capsys):
    model = _build_model()
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "fresh.ckpt"
    forge.save_checkpoint(str(path), model, optimizer)

    code = main(["checkpoint", "inspect", str(path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Optimizer: SGD" in out
    assert "Optimizer parameters: 4 total, 0 with saved state" in out


# -- model convert ----------------------------------------------------------


def test_model_convert_cpu_to_cpu(model_path, tmp_path, capsys):
    output = tmp_path / "converted.forge"
    code = main(["model", "convert", str(model_path), "--device", "cpu", "--output", str(output)])
    assert code == 0
    assert "Converted" in capsys.readouterr().out
    assert output.is_file()

    reloaded = forge.load_model(str(output))
    assert reloaded.device.type == "cpu"
    names_before = {n for n, _ in _build_model().named_parameters()}
    names_after = {n for n, _ in reloaded.named_parameters()}
    assert names_before == names_after


def test_model_convert_invalid_device_choice_exits_nonzero(model_path, tmp_path):
    output = tmp_path / "x.forge"
    with pytest.raises(SystemExit) as exc_info:
        main(["model", "convert", str(model_path), "--device", "tpu", "--output", str(output)])
    assert exc_info.value.code != 0
    assert not output.exists()


def test_model_convert_missing_output_directory_returns_nonzero(model_path, tmp_path, capsys):
    output = tmp_path / "does_not_exist" / "x.forge"
    code = main(["model", "convert", str(model_path), "--device", "cpu", "--output", str(output)])
    assert code != 0
    assert "Error" in capsys.readouterr().err


def test_model_convert_cuda_unavailable_returns_clear_error(model_path, tmp_path, monkeypatch, capsys):
    import forge.backend.cuda as cuda_module

    monkeypatch.setattr(cuda_module, "is_cuda_available", lambda: False)
    output = tmp_path / "x.forge"
    code = main(["model", "convert", str(model_path), "--device", "cuda", "--output", str(output)])
    assert code != 0
    assert "CUDA" in capsys.readouterr().err
    assert not output.exists()


# -- checkpoint convert -------------------------------------------------------


def test_checkpoint_convert_cpu_to_cpu(checkpoint_path, tmp_path, capsys):
    output = tmp_path / "converted.ckpt"
    code = main(["checkpoint", "convert", str(checkpoint_path), "--device", "cpu", "--output", str(output)])
    assert code == 0
    assert "Converted" in capsys.readouterr().out
    assert output.is_file()

    reloaded = forge.load_checkpoint(str(output))
    assert reloaded.epoch == 2
    assert reloaded.global_step == 10
    assert isinstance(reloaded.optimizer, Adam)


def test_checkpoint_convert_cuda_unavailable_returns_clear_error(checkpoint_path, tmp_path, monkeypatch, capsys):
    import forge.backend.cuda as cuda_module

    monkeypatch.setattr(cuda_module, "is_cuda_available", lambda: False)
    output = tmp_path / "x.ckpt"
    code = main(["checkpoint", "convert", str(checkpoint_path), "--device", "cuda", "--output", str(output)])
    assert code != 0
    assert "CUDA" in capsys.readouterr().err
    assert not output.exists()


# -- benchmark pass-through --------------------------------------------------


def test_benchmark_forwards_arguments_unparsed(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv

    import benchmarks.run as benchmarks_run

    monkeypatch.setattr(benchmarks_run, "main", fake_main)

    code = main(["benchmark", "--categories", "forward", "--output", "out.json"])
    assert code == 0
    assert captured["argv"] == ["--categories", "forward", "--output", "out.json"]


def test_benchmark_missing_package_returns_clear_error(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "benchmarks.run" or name.startswith("benchmarks"):
            raise ImportError("no module named benchmarks")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(["benchmark"])
    assert code != 0
    assert "benchmark" in capsys.readouterr().err.lower()


# -- CUDA-hardware-verified conversion (skips cleanly without CUDA) ---------


pytestmark_cuda = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytestmark_cuda
def test_model_convert_cpu_to_cuda_to_cpu_round_trip(model_path, tmp_path):
    from forge.backend.cuda import CUDAStorage

    cuda_path = tmp_path / "cuda.forge"
    code = main(["model", "convert", str(model_path), "--device", "cuda", "--output", str(cuda_path)])
    assert code == 0

    cuda_model = forge.load_model(str(cuda_path))
    assert cuda_model.device.type == "cuda"
    for param in cuda_model.parameters():
        assert isinstance(param._data, CUDAStorage)

    cpu_path = tmp_path / "back_to_cpu.forge"
    code = main(["model", "convert", str(cuda_path), "--device", "cpu", "--output", str(cpu_path)])
    assert code == 0
    cpu_model = forge.load_model(str(cpu_path))
    assert cpu_model.device.type == "cpu"

    original = _build_model()
    orig_by_name = dict(original.named_parameters())
    for name, param in cpu_model.named_parameters():
        assert param.shape == orig_by_name[name].shape
        assert param.numpy().tolist() == orig_by_name[name].numpy().tolist()


@pytestmark_cuda
def test_checkpoint_convert_cpu_to_cuda_creates_real_cuda_state(checkpoint_path, tmp_path):
    from forge.backend.cuda import CUDAStorage

    cuda_path = tmp_path / "cuda.ckpt"
    code = main(["checkpoint", "convert", str(checkpoint_path), "--device", "cuda", "--output", str(cuda_path)])
    assert code == 0

    cuda_checkpoint = forge.load_checkpoint(str(cuda_path))
    assert cuda_checkpoint.model.device.type == "cuda"
    for param in cuda_checkpoint.model.parameters():
        assert isinstance(param._data, CUDAStorage)
    for state in cuda_checkpoint.optimizer.state.values():
        assert isinstance(state.m, CUDAStorage)
        assert isinstance(state.v, CUDAStorage)

    cpu_path = tmp_path / "back_to_cpu.ckpt"
    code = main(["checkpoint", "convert", str(cuda_path), "--device", "cpu", "--output", str(cpu_path)])
    assert code == 0
    cpu_checkpoint = forge.load_checkpoint(str(cpu_path))
    assert cpu_checkpoint.model.device.type == "cpu"
    assert cpu_checkpoint.epoch == 2
    assert cpu_checkpoint.global_step == 10
