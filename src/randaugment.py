"""RandAugment implemented by viewax."""
from __future__ import annotations
from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import chex

from . import _src as F


def default_augment_space(n_bins: int):
    return {
        "ShearX": (jnp.linspace(0, 0.3, n_bins), True),
        "ShearY": (jnp.linspace(0, 0.3, n_bins), True),
        "TranslateX": (jnp.linspace(0, 150.0 / 331.0, n_bins), True),
        "TranslateY": (jnp.linspace(0, 150.0 / 331.0, n_bins), True),
        "Rotate": (jnp.linspace(0, 30, n_bins), True),
        "Brightness": (jnp.linspace(0, 0.9, n_bins), True),
        "Color": (jnp.linspace(0, 0.9, n_bins), True),
        "Contrast": (jnp.linspace(0, 0.9, n_bins), True),
        "Sharpness": (jnp.linspace(0, 0.9, n_bins), True),
        "Posterize": (8 - jnp.round(jnp.arange(n_bins) / (n_bins - 1) / 4), False),
        "Solarize": (jnp.linspace(255.0, 0.0, n_bins), False),
        "AutoContrast": (jnp.zeros(n_bins), False),
        "Equalize": (jnp.zeros(n_bins), False),
        "Invert": (jnp.zeros(n_bins), False),
        "Identity": (jnp.zeros(n_bins), False),
    }


def create_randaugment(
    n_layers: int,
    n_bins: int,
    augment_space: dict[str, tuple[chex.Array], bool] | None = None,
    order: int = 0,
    mode: str = "constant",
    cval: int = 128,
) -> Callable[[chex.PRNGKey, chex.Array], chex.Array]:
    """Create a function that transforms images by randaugment.

    Args:
        n_layers (int): Number of operations to transform images.
        n_bins (int): Number of bins.
        augment_space (dict): Augmentation space.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond its boundaries.
                    "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant". Default is 128.

    Returns:
        A function that transforms images by randaugment.

    Example:
        >>> rng = jax.random.PRNGKey(0)
        >>> apply_ra = create_randaugment(...)
        >>> img = apply_ra(rng, img)
    """

    if augment_space is None:
        augment_space = default_augment_space(n_bins)

    @jax.jit
    def shear_x(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.shear(img, (0, v), order=order, mode=mode, cval=cval)

    @jax.jit
    def shear_y(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.shear(img, (v, 0), order=order, mode=mode, cval=cval)

    @jax.jit
    def translate_x(img: chex.Array, idx: int, magnitudes: chex.Array):
        width = img.shape[1]
        v = width * magnitudes[idx]
        return F.translate(img, (0, v), order=order, mode=mode, cval=cval)

    @jax.jit
    def translate_y(img: chex.Array, idx: int, magnitudes: chex.Array):
        height = img.shape[0]
        v = height * magnitudes[idx]
        return F.translate(img, (v, 0), order=order, mode=mode, cval=cval)

    @jax.jit
    def rotate(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.rotate(img, v, order=order, mode=mode, cval=cval)

    @jax.jit
    def brightness(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.brightness(img, v)

    @jax.jit
    def color(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.color(img, v)

    @jax.jit
    def contrast(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.contrast(img, v)

    @jax.jit
    def sharpness(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.sharpness(img, v)

    @jax.jit
    def posterize(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = jnp.int32(magnitudes[idx])
        return F.posterize(img, v)

    @jax.jit
    def solarize(img: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.solarize(img, v)

    @jax.jit
    def autocontrast(img: chex.Array, idx: int, magnitudes: chex.Array):
        return F.autocontrast(img)

    @jax.jit
    def equalize(img: chex.Array, idx: int, magnitudes: chex.Array):
        return F.equalize(img)

    @jax.jit
    def invert(img: chex.Array, idx: int, magnitudes: chex.Array):
        return F.invert(img)

    @jax.jit
    def identity(img: chex.Array, idx: int, magnitudes: chex.Array):
        return img

    operations = {
        "ShearX": shear_x,
        "ShearY": shear_y,
        "TranslateX": translate_x,
        "TranslateY": translate_y,
        "Rotate": rotate,
        "Brightness": brightness,
        "Color": color,
        "Contrast": contrast,
        "Sharpness": sharpness,
        "Posterize": posterize,
        "Solarize": solarize,
        "AutoContrast": autocontrast,
        "Equalize": equalize,
        "Invert": invert,
        "Identity": identity,
    }

    branches = []
    for key, (magnitudes, signed) in augment_space.items():
        op = operations[key]
        if signed:
            magnitudes = jnp.concatenate([magnitudes, -magnitudes])
        else:
            magnitudes = jnp.concatenate([magnitudes, magnitudes])
        branches.append(partial(op, magnitudes=magnitudes))

    @jax.jit
    def body(carry, x):
        op_index, magnitude_idx = x
        carry = jax.lax.switch(
            op_index,
            branches,
            carry,
            magnitude_idx,
        )
        return carry, jnp.zeros(())

    @jax.jit
    def fn(rng: chex.PRNGKey, img: chex.Array) -> chex.Array:
        rng_op, rng_mag = jax.random.split(rng, 2)
        op_idxs = jax.random.randint(rng_op, (n_layers,), 0, len(branches))
        mag_idxs = jax.random.randint(rng_mag, (n_layers,), 0, 2 * n_bins)
        img, _ = jax.lax.scan(body, img, xs=[op_idxs, mag_idxs])
        return img

    return fn
