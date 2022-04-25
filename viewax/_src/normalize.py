from __future__ import annotations
import jax.numpy as jnp
import chex
from .utils import flatten, unflatten


def normalize(
    img: chex.Array,
    mean: chex.Array | float | tuple[float, ...] = 0.0,
    std: chex.Array | float | tuple[float, ...] = 1.0,
):
    """Normalize an image array.

    This function processes img as follows:
        img <- img / 255
        img <- (img - mean) / std

    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        mean: Mean value to normalize.
        std: Std value to normalize.

    Returns:
        Normalized images.
    """
    img, original_shape = flatten(img)

    mean = jnp.array(mean).reshape(1, 1, 1, -1)
    std = jnp.array(std).reshape(1, 1, 1, -1)

    img = jnp.float32(img) / 255.0
    img = (img - mean) / std

    return unflatten(img, original_shape)
