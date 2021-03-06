"""Modify from: https://github.com/4rtemi5/imax/blob/master/imax/color_transforms.py"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import chex

from .utils import flatten, unflatten

__all__ = [
    "solarize",
    "color",
    "contrast",
    "brightness",
    "posterize",
    "autocontrast",
    "sharpness",
    "equalize",
    "invert",
    "rgb2gray",
]


def blend(img1: chex.Array, img2: chex.Array, factor: float) -> chex.Array:
    # factor * img1 + (1-factor) * img2
    img1 = jnp.float32(img1)
    img2 = jnp.float32(img2)
    return factor * (img1 - img2) + img2


@jax.jit
def rgb2gray(img: chex.Array) -> chex.Array:
    img = jnp.float32(img)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img = jnp.stack([v, v, v], axis=-1)
    return img


def solarize(img: chex.Array, threshold: int = 128, addition: int = 0) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        threshold (int): Threshold value to solarize.
        addition (int): Addition value. [-128, 128].

    Returns:
        Soralized images.
    """
    img = jnp.int32(img)
    img = jnp.clip(img + addition, 0, 255).astype(jnp.uint8)
    return jnp.where(img < threshold, img, 255 - img).clip(0, 255).astype(jnp.uint8)


def color(img: chex.Array, factor: float) -> chex.Array:
    return blend(img, rgb2gray(img), factor).clip(0, 255).astype(jnp.uint8)


def contrast(img: chex.Array, factor: float) -> chex.Array:
    degenerate = rgb2gray(img * 255)[..., 0]
    degenerate = jnp.mean(degenerate, axis=(-1, -2))
    degenerate = jnp.floor(degenerate + 0.5)[..., None, None, None]
    return blend(img, degenerate, factor).clip(0, 255).astype(jnp.uint8)


def brightness(img, factor):
    return blend(img, jnp.zeros_like(img), factor).clip(0, 255).astype(jnp.uint8)


def posterize(img: chex.Array, bits: int):
    shift = 8 - bits
    degenerate = jnp.left_shift(jnp.right_shift(img, shift), shift)
    return degenerate.clip(0, 255).astype(jnp.uint8)


def autocontrast(img: chex.Array):
    @jax.jit
    def scale_channel(carry, x):
        # carry is dummy variable to apply scale_channel with jax.lax.scan.

        low = jnp.min(x).astype("float32")
        high = jnp.max(x).astype("float32")

        def _scale_values(v: chex.Array) -> chex.Array:
            scale = 255.0 / (high - low)
            offset = -low * scale
            v = v.astype("float32") * scale + offset
            return jnp.clip(v, 0, 255).astype("uint8")

        x = jax.lax.cond(high > low, _scale_values, lambda v: v, x)
        return carry, x

    *_, height, width, channel = img.shape
    img, original_shape = flatten(img)

    img = img.transpose(0, 3, 1, 2).reshape(-1, height, width)
    _, img = jax.lax.scan(scale_channel, jnp.zeros(()), img)
    img = img.reshape(-1, channel, height, width).transpose(0, 2, 3, 1)
    img = unflatten(img, original_shape)
    return img.clip(0, 255).astype(jnp.uint8)


def sharpness(img: chex.Array, factor: float):
    img, original_shape = flatten(img)

    degenerate = img.astype("float32")
    # SMOOTH PIL Kernel.
    kernel = jnp.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype="float32") / 13.0
    kernel = jnp.reshape(kernel, (3, 3, 1, 1))
    kernel = jnp.tile(kernel, [1, 1, 1, 3])
    degenerate = jax.lax.conv_general_dilated(
        jnp.transpose(degenerate, [0, 3, 1, 2]),  # lhs = NCHW image tensor
        jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        "VALID",  # padding mode
        feature_group_count=3,
    )
    degenerate = jnp.clip(degenerate, 0.0, 255.0).astype(jnp.uint8)
    degenerate = jnp.transpose(degenerate, [0, 2, 3, 1])  # NCHW -> NHWC
    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = jnp.ones_like(degenerate)
    padded_mask = jnp.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
    padded_degenerate = jnp.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
    degenerate = jnp.where(jnp.equal(padded_mask, 1), padded_degenerate, img)

    degenerate = blend(img, degenerate, factor)
    degenerate = unflatten(degenerate, original_shape)
    return degenerate.clip(0, 255).astype(jnp.uint8)


def equalize(img: chex.Array) -> chex.Array:
    """
    Implements Equalize function from PIL using Jax ops.
    Args:
        image: image tensor
    Returns:
        Augmented image.
    """

    @jax.jit
    def build_lut(histo, step):
        # Compute the cumulative sum, shifting by step // 2
        # and then normalization by step.
        lut = (jnp.cumsum(histo) + (step // 2)) // step
        # Shift lut, prepending with 0.
        lut = jnp.concatenate([jnp.array([0]), lut[:-1]], 0)
        # Clip the counts to be in range.  This is done
        # in the C code for image.point.
        return jnp.clip(lut, 0, 255)

    @jax.jit
    def scale_channel(img):
        """
        Scale the data in the channel to implement equalize.
        Args:
            img: channel to scale.
        Returns:
            scaled channel
        """
        # im = im[:, :, c].astype('int32')
        img = img.astype("int32")
        # Compute the histogram of the image channel.
        histo = jnp.histogram(img, bins=255, range=(0, 255))[0]

        last_nonzero = jnp.argmax(histo[::-1] > 0)  # jnp.nonzero(histo)[0][-1]
        step = (jnp.sum(histo) - jnp.take(histo[::-1], last_nonzero)) // 255

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        return jax.lax.cond(
            step == 0, lambda x: x.astype("uint8"), lambda x: jnp.take(build_lut(histo, step), x).astype("uint8"), img
        )

    img, original_shape = flatten(img)
    _, height, width, channel = img.shape

    img = img.transpose(0, 3, 1, 2).reshape(-1, height, width)
    _, img = jax.lax.scan(
        lambda carry, x: (carry, scale_channel(x)),
        jnp.zeros(()),
        img,
    )

    img = img.reshape(-1, channel, height, width).transpose(0, 2, 3, 1)
    return unflatten(img, original_shape).clip(0, 255).astype(jnp.uint8)


def invert(img: chex.Array) -> chex.Array:
    return (255 - img).clip(0, 255).astype(jnp.uint8)
