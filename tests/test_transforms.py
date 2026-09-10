import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import DataError
from forge.data.transforms import (
    Compose,
    Flatten,
    Lambda,
    Normalize,
    Reshape,
    Resize,
    ToTensor,
    Transform,
)


def _image_tensor(channels, height, width, fill=100.0):
    return Tensor(np.full((channels, height, width), fill, dtype=np.float32))


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


# -- Resize -----------------------------------------------------------------


def test_resize_portrait_to_target():
    sample = _image_tensor(3, 90, 40)  # H=90, W=40
    resized = Resize((64, 64))(sample)
    assert resized.shape == (3, 64, 64)


def test_resize_landscape_to_target():
    sample = _image_tensor(3, 40, 90)  # H=40, W=90
    resized = Resize((64, 64))(sample)
    assert resized.shape == (3, 64, 64)


def test_resize_square_to_target():
    sample = _image_tensor(3, 50, 50)
    resized = Resize((64, 64))(sample)
    assert resized.shape == (3, 64, 64)


def test_resize_non_square_target():
    sample = _image_tensor(3, 50, 80)
    resized = Resize((32, 96))(sample)
    assert resized.shape == (3, 32, 96)


def test_resize_rgb_input_preserves_channel_count():
    sample = _image_tensor(3, 20, 30)
    resized = Resize((10, 10))(sample)
    assert resized.shape[0] == 3


def test_resize_grayscale_input_preserves_channel_count():
    sample = _image_tensor(1, 20, 30)
    resized = Resize((10, 10))(sample)
    assert resized.shape[0] == 1


def test_resize_output_dtype_matches_input():
    sample = _image_tensor(3, 20, 30)
    resized = Resize((10, 10))(sample)
    assert resized.dtype == sample.dtype


def test_resize_output_value_range_stays_within_uint8_bounds():
    sample = _image_tensor(3, 20, 30, fill=255.0)
    resized = Resize((10, 10))(sample)
    array = resized.numpy()
    assert array.min() >= 0.0
    assert array.max() <= 255.0


def test_resize_uniform_fill_is_preserved_by_bilinear_interpolation():
    sample = _image_tensor(3, 20, 30, fill=128.0)
    resized = Resize((10, 10))(sample)
    np.testing.assert_allclose(resized.numpy(), 128.0, atol=1.0)


def test_resize_is_deterministic():
    sample = Tensor(np.arange(3 * 20 * 30, dtype=np.float32).reshape(3, 20, 30) % 255.0)
    a = Resize((16, 16))(sample)
    b = Resize((16, 16))(sample)
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_resize_rejects_non_tensor_sample():
    with pytest.raises(DataError):
        Resize((10, 10))(np.zeros((3, 10, 10)))


def test_resize_rejects_wrong_ndim():
    with pytest.raises(DataError):
        Resize((10, 10))(Tensor(np.zeros((10, 10), dtype=np.float32)))


def test_resize_rejects_unsupported_channel_count():
    with pytest.raises(DataError):
        Resize((10, 10))(_image_tensor(4, 20, 30))


def test_resize_rejects_malformed_size_tuple():
    with pytest.raises(DataError):
        Resize((10, 10, 10))


def test_resize_rejects_non_tuple_size():
    with pytest.raises(DataError):
        Resize(10)


def test_resize_rejects_zero_dimension():
    with pytest.raises(DataError):
        Resize((0, 10))


def test_resize_rejects_negative_dimension():
    with pytest.raises(DataError):
        Resize((10, -5))


def test_resize_rejects_non_integer_dimension():
    with pytest.raises(DataError):
        Resize((10.5, 10))


def test_resize_repr():
    assert repr(Resize((64, 64))) == "Resize(size=(64, 64))"


def test_resize_composes_with_lambda():
    sample = _image_tensor(3, 20, 30, fill=255.0)
    pipeline = Compose([Resize((8, 8)), Lambda(lambda t: t * (1.0 / 255.0))])
    result = pipeline(sample)
    assert result.shape == (3, 8, 8)
    np.testing.assert_allclose(result.numpy(), 1.0, atol=1.0 / 255.0)


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
