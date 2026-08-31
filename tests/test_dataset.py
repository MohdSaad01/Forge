import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import DataError
from forge.nn import Linear
from forge.data import Dataset, Subset, TensorDataset, random_split

# -- base Dataset -------------------------------------------------------


def test_base_dataset_len_raises():
    with pytest.raises(DataError):
        len(Dataset())


def test_base_dataset_getitem_raises():
    with pytest.raises(DataError):
        Dataset()[0]


# -- TensorDataset: construction / validation ----------------------------


def test_tensor_dataset_length():
    x = Tensor(np.zeros((5, 3)))
    y = Tensor(np.zeros((5,)))
    ds = TensorDataset(x, y)
    assert len(ds) == 5


def test_tensor_dataset_rejects_mismatched_sample_counts():
    x = Tensor(np.zeros((5, 3)))
    y = Tensor(np.zeros((4,)))
    with pytest.raises(DataError):
        TensorDataset(x, y)


def test_tensor_dataset_rejects_no_tensors():
    with pytest.raises(DataError):
        TensorDataset()


def test_tensor_dataset_rejects_scalar_tensor():
    x = Tensor(np.zeros((5,)))
    scalar = Tensor(1.0)
    with pytest.raises(DataError):
        TensorDataset(x, scalar)


def test_tensor_dataset_rejects_empty_tensors():
    x = Tensor(np.zeros((0, 3)))
    with pytest.raises(DataError):
        TensorDataset(x)


def test_tensor_dataset_accepts_raw_array_like():
    ds = TensorDataset(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
    assert len(ds) == 2
    x0, y0 = ds[0]
    assert isinstance(x0, Tensor)
    np.testing.assert_allclose(x0.numpy(), [1.0, 2.0])


# -- TensorDataset: indexing / retrieval ----------------------------------


def test_tensor_dataset_getitem_returns_matching_features_and_target():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    y = Tensor(np.array([10.0, 20.0, 30.0]))
    ds = TensorDataset(x, y)

    xi, yi = ds[1]
    np.testing.assert_allclose(xi.numpy(), [3.0, 4.0])
    np.testing.assert_allclose(yi.numpy(), 20.0)


def test_tensor_dataset_single_tensor_returns_bare_sample():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    ds = TensorDataset(x)
    sample = ds[1]
    assert isinstance(sample, Tensor)
    np.testing.assert_allclose(sample.numpy(), 2.0)


def test_tensor_dataset_preserves_dtype_and_device():
    x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int64))
    ds = TensorDataset(x)
    sample = ds[0]
    assert sample.dtype == x.dtype
    assert sample.device == x.device


def test_tensor_dataset_negative_index():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    ds = TensorDataset(x)
    np.testing.assert_allclose(ds[-1].numpy(), 3.0)


def test_tensor_dataset_shape_preservation_for_multidim_features():
    x = Tensor(np.zeros((4, 3, 2)))
    ds = TensorDataset(x)
    assert ds[0].shape == (3, 2)


# -- TensorDataset: invalid index ------------------------------------------


def test_tensor_dataset_rejects_out_of_range_index():
    ds = TensorDataset(Tensor(np.zeros((3,))))
    with pytest.raises(DataError):
        ds[3]


def test_tensor_dataset_rejects_non_int_index():
    ds = TensorDataset(Tensor(np.zeros((3,))))
    with pytest.raises(DataError):
        ds["0"]


# -- TensorDataset: transform / target_transform ---------------------------


def test_tensor_dataset_applies_transform_to_features_only():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = Tensor(np.array([100.0, 200.0]))
    # A per-sample transform on a (2,) feature vector: double it.
    from forge.data.transforms import Lambda

    ds = TensorDataset(x, y, transform=Lambda(lambda t: t * 2.0))
    xi, yi = ds[0]
    np.testing.assert_allclose(xi.numpy(), [2.0, 4.0])
    np.testing.assert_allclose(yi.numpy(), 100.0)


def test_tensor_dataset_applies_target_transform_to_target_only():
    from forge.data.transforms import Lambda

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = Tensor(np.array([1.0, 0.0]))
    ds = TensorDataset(x, y, target_transform=Lambda(lambda t: t * 10.0))
    xi, yi = ds[0]
    np.testing.assert_allclose(xi.numpy(), [1.0, 2.0])
    np.testing.assert_allclose(yi.numpy(), 10.0)


def test_tensor_dataset_target_transform_requires_two_tensors():
    x = Tensor(np.zeros((3,)))
    from forge.data.transforms import Lambda

    with pytest.raises(DataError):
        TensorDataset(x, target_transform=Lambda(lambda t: t))


# -- Subset / random_split --------------------------------------------------


def test_subset_preserves_correspondence():
    x = Tensor(np.array([[0.0], [1.0], [2.0], [3.0]]))
    y = Tensor(np.array([0.0, 1.0, 2.0, 3.0]))
    ds = TensorDataset(x, y)
    subset = Subset(ds, [3, 1])
    assert len(subset) == 2
    xi0, yi0 = subset[0]
    np.testing.assert_allclose(xi0.numpy(), [3.0])
    np.testing.assert_allclose(yi0.numpy(), 3.0)


def test_random_split_sizes_and_disjoint_coverage():
    x = Tensor(np.arange(10.0).reshape(10, 1))
    ds = TensorDataset(x)
    rng = np.random.default_rng(0)
    train, test = random_split(ds, [7, 3], generator=rng)
    assert len(train) == 7
    assert len(test) == 3

    all_values = sorted(
        [float(train[i].numpy()[0]) for i in range(len(train))]
        + [float(test[i].numpy()[0]) for i in range(len(test))]
    )
    assert all_values == [float(v) for v in range(10)]


def test_random_split_deterministic_with_same_generator_seed():
    x = Tensor(np.arange(10.0).reshape(10, 1))
    ds = TensorDataset(x)
    a_train, a_test = random_split(ds, [6, 4], generator=np.random.default_rng(7))
    b_train, b_test = random_split(ds, [6, 4], generator=np.random.default_rng(7))
    assert a_train.indices == b_train.indices
    assert a_test.indices == b_test.indices


def test_random_split_rejects_mismatched_lengths():
    x = Tensor(np.zeros((10, 1)))
    ds = TensorDataset(x)
    with pytest.raises(DataError):
        random_split(ds, [6, 3])


def test_random_split_rejects_negative_length():
    x = Tensor(np.zeros((5, 1)))
    ds = TensorDataset(x)
    with pytest.raises(DataError):
        random_split(ds, [-1, 6])


# -- integration smoke: dataset item feeds a model ---------------------------


def test_tensor_dataset_sample_feeds_linear_model():
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    ds = TensorDataset(x)
    model = Linear(3, 2)
    sample = ds[0].reshape(1, 3)
    out = model(sample)
    assert out.shape == (1, 2)
