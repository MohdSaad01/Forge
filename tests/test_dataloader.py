import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import DataError
from forge.data import DataLoader, Dataset, TensorDataset


def _range_dataset(n: int) -> TensorDataset:
    x = Tensor(np.arange(n, dtype=np.float64).reshape(n, 1))
    y = Tensor(np.arange(n, dtype=np.float64) * 10.0)
    return TensorDataset(x, y)


# -- batch size / partial batch / drop_last ---------------------------------


def test_batch_size_produces_expected_number_of_batches():
    ds = _range_dataset(10)
    loader = DataLoader(ds, batch_size=4)
    batches = list(loader)
    assert len(batches) == 3
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]


def test_drop_last_false_keeps_final_partial_batch():
    ds = _range_dataset(10)
    loader = DataLoader(ds, batch_size=4, drop_last=False)
    sizes = [b[0].shape[0] for b in loader]
    assert sizes == [4, 4, 2]
    assert len(loader) == 3


def test_drop_last_true_drops_final_partial_batch():
    ds = _range_dataset(10)
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    sizes = [b[0].shape[0] for b in loader]
    assert sizes == [4, 4]
    assert len(loader) == 2


def test_exact_division_produces_no_partial_batch_either_way():
    ds = _range_dataset(8)
    loader_keep = DataLoader(ds, batch_size=4, drop_last=False)
    loader_drop = DataLoader(ds, batch_size=4, drop_last=True)
    assert [b[0].shape[0] for b in loader_keep] == [4, 4]
    assert [b[0].shape[0] for b in loader_drop] == [4, 4]


# -- no-shuffle ordering ------------------------------------------------------


def test_no_shuffle_preserves_dataset_order():
    ds = _range_dataset(9)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    values = []
    for bx, _ in loader:
        values.extend(bx.numpy().ravel().tolist())
    assert values == list(range(9))


# -- shuffle behavior ---------------------------------------------------------


def test_shuffle_changes_order_relative_to_sequential():
    ds = _range_dataset(50)
    loader = DataLoader(ds, batch_size=50, shuffle=True, generator=np.random.default_rng(1))
    (bx, _), = list(loader)
    values = bx.numpy().ravel().tolist()
    assert values != list(range(50))
    assert sorted(values) == list(range(50))


def test_shuffle_changes_between_successive_iterations():
    ds = _range_dataset(50)
    loader = DataLoader(ds, batch_size=50, shuffle=True)
    (first_batch,) = [b[0].numpy().ravel().tolist() for b in loader]
    (second_batch,) = [b[0].numpy().ravel().tolist() for b in loader]
    assert first_batch != second_batch


# -- deterministic shuffle with supplied generator ----------------------------


def test_deterministic_shuffle_with_same_generator_seed_matches():
    ds = _range_dataset(20)
    loader_a = DataLoader(ds, batch_size=6, shuffle=True, generator=np.random.default_rng(123))
    loader_b = DataLoader(ds, batch_size=6, shuffle=True, generator=np.random.default_rng(123))

    order_a = [b[0].numpy().ravel().tolist() for b in loader_a]
    order_b = [b[0].numpy().ravel().tolist() for b in loader_b]
    assert order_a == order_b


def test_deterministic_shuffle_different_seeds_differ():
    ds = _range_dataset(30)
    loader_a = DataLoader(ds, batch_size=30, shuffle=True, generator=np.random.default_rng(1))
    loader_b = DataLoader(ds, batch_size=30, shuffle=True, generator=np.random.default_rng(2))

    (order_a,) = [b[0].numpy().ravel().tolist() for b in loader_a]
    (order_b,) = [b[0].numpy().ravel().tolist() for b in loader_b]
    assert order_a != order_b


# -- tuple samples: feature/target correspondence preserved through batching -


def test_tuple_samples_preserve_feature_target_correspondence():
    ds = _range_dataset(12)
    loader = DataLoader(ds, batch_size=5, shuffle=True, generator=np.random.default_rng(9))
    for bx, by in loader:
        np.testing.assert_allclose(bx.numpy().ravel() * 10.0, by.numpy())


def test_single_tensor_dataset_batches_to_single_tensor():
    x = Tensor(np.arange(6.0).reshape(6, 1))
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert isinstance(batch, Tensor)
    assert batch.shape == (2, 1)


# -- invalid configuration -----------------------------------------------


@pytest.mark.parametrize("bad_batch_size", [0, -1, 1.5, "4", True, False])
def test_invalid_batch_size_raises(bad_batch_size):
    ds = _range_dataset(5)
    with pytest.raises(DataError):
        DataLoader(ds, batch_size=bad_batch_size)


def test_invalid_shuffle_type_raises():
    ds = _range_dataset(5)
    with pytest.raises(DataError):
        DataLoader(ds, batch_size=2, shuffle="yes")


def test_invalid_drop_last_type_raises():
    ds = _range_dataset(5)
    with pytest.raises(DataError):
        DataLoader(ds, batch_size=2, drop_last="yes")


def test_dataloader_requires_dataset_supporting_len():
    class NoLen:
        def __getitem__(self, index):
            return Tensor([0.0])

    with pytest.raises(DataError):
        DataLoader(NoLen(), batch_size=2)


def test_dataloader_rejects_inconsistent_sample_structure():
    class Weird(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            if index == 0:
                return Tensor([1.0]), Tensor([2.0])
            return Tensor([1.0])

    loader = DataLoader(Weird(), batch_size=2)
    with pytest.raises(DataError):
        list(loader)


def test_dataloader_rejects_mismatched_component_shapes():
    class Weird(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return Tensor(np.zeros(3 if index == 0 else 4))

    loader = DataLoader(Weird(), batch_size=2)
    with pytest.raises(DataError):
        list(loader)
