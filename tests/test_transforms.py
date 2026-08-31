import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import DataError
from forge.data.transforms import Compose, Flatten, Lambda, Normalize, Reshape, ToTensor, Transform


# -- base Transform / Compose -------------------------------------------


def test_base_transform_call_raises():
    with pytest.raises(DataError):
        Transform()(Tensor([1.0]))


def test_compose_applies_transforms_in_order():
    add_one = Lambda(lambda t: t + 1.0)
    times_two = Lambda(lambda t: t * 2.0)
    compose = Compose([add_one, times_two])
    result = compose(Tensor([1.0, 2.0]))
    np.testing.assert_allclose(result.numpy(), [4.0, 6.0])  # (x+1)*2


def test_compose_order_matters():
    add_one = Lambda(lambda t: t + 1.0)
    times_two = Lambda(lambda t: t * 2.0)
    forward = Compose([add_one, times_two])(Tensor([1.0]))
    backward = Compose([times_two, add_one])(Tensor([1.0]))
    assert forward.numpy()[0] != backward.numpy()[0]
    np.testing.assert_allclose(forward.numpy(), [4.0])
    np.testing.assert_allclose(backward.numpy(), [3.0])


def test_compose_rejects_non_callable_member():
    with pytest.raises(DataError):
        Compose([Lambda(lambda t: t), "not callable"])


def test_compose_empty_is_identity():
    x = Tensor([1.0, 2.0])
    result = Compose([])(x)
    assert result is x


# -- ToTensor ---------------------------------------------------------------


def test_to_tensor_converts_raw_array():
    t = ToTensor()(np.array([1.0, 2.0, 3.0]))
    assert isinstance(t, Tensor)
    np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])


def test_to_tensor_accepts_python_list():
    t = ToTensor()([1, 2, 3])
    assert isinstance(t, Tensor)


# -- Normalize ----------------------------------------------------------------


def test_normalize_scalar_mean_std():
    sample = Tensor([2.0, 4.0, 6.0])
    normalized = Normalize(mean=4.0, std=2.0)(sample)
    np.testing.assert_allclose(normalized.numpy(), [-1.0, 0.0, 1.0])


def test_normalize_per_channel_broadcast():
    sample = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    normalized = Normalize(mean=[1.0, 2.0], std=[2.0, 4.0])(sample)
    expected = (sample.numpy() - np.array([1.0, 2.0])) / np.array([2.0, 4.0])
    np.testing.assert_allclose(normalized.numpy(), expected)


def test_normalize_rejects_zero_std():
    with pytest.raises(DataError):
        Normalize(mean=0.0, std=0.0)


def test_normalize_rejects_non_tensor_sample():
    with pytest.raises(DataError):
        Normalize(mean=0.0, std=1.0)(np.array([1.0]))


# -- Reshape / Flatten ----------------------------------------------------


def test_reshape_transform():
    sample = Tensor(np.arange(6.0))
    reshaped = Reshape(2, 3)(sample)
    assert reshaped.shape == (2, 3)


def test_reshape_rejects_non_tensor_sample():
    with pytest.raises(DataError):
        Reshape(2, 3)([1, 2, 3, 4, 5, 6])


def test_flatten_transform():
    sample = Tensor(np.arange(6.0).reshape(2, 3))
    flat = Flatten()(sample)
    assert flat.shape == (6,)
    np.testing.assert_allclose(flat.numpy(), np.arange(6.0))


def test_flatten_rejects_non_tensor_sample():
    with pytest.raises(DataError):
        Flatten()([1, 2, 3])


# -- Lambda -------------------------------------------------------------------


def test_lambda_applies_arbitrary_function():
    double = Lambda(lambda t: t * 2.0)
    result = double(Tensor([1.0, 2.0]))
    np.testing.assert_allclose(result.numpy(), [2.0, 4.0])


def test_lambda_rejects_non_callable():
    with pytest.raises(DataError):
        Lambda("not callable")


# -- feature/target semantics via Compose in a realistic pipeline -------------


def test_compose_pipeline_is_deterministic_across_calls():
    pipeline = Compose([ToTensor(), Normalize(mean=0.0, std=2.0), Flatten()])
    a = pipeline(np.array([[2.0, 4.0]]))
    b = pipeline(np.array([[2.0, 4.0]]))
    np.testing.assert_allclose(a.numpy(), b.numpy())
    np.testing.assert_allclose(a.numpy(), [1.0, 2.0])
