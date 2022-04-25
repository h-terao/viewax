from __future__ import annotations

import jax.numpy as jnp
from jax.scipy import ndimage as ndi
import chex

from .utils import flatten, unflatten


__all__ = [
    "warp_affine",
    "rotate",
    "rot90",
    "translate",
    "shear",
    "hflip",
    "vflip",
]


def warp_affine(
    img: chex.Array, matrix: chex.Array, order: int = 0, mode: str = "constant", cval: int = 0
) -> chex.Array:
    """Warp an image according to a given coordinate transformations.

    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        matrix (array): An affine transformation matrix.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond its boundaries.
                    "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant". Default is 128.

    Notes:
        Transformed images.
    """
    img, original_shape = flatten(img)

    batch_size, height, width, channel = img.shape
    img = img.transpose(0, 3, 1, 2).reshape(-1, height, width)  # NHWC -> NCHW
    N = batch_size * channel

    img = jnp.float32(img)
    x_t, y_t = jnp.meshgrid(
        jnp.arange(0, width),
        jnp.arange(0, height),
    )
    pixel_coords = jnp.stack([x_t, y_t, jnp.ones_like(x_t)]).astype(jnp.float32)
    x_coords, y_coords, _ = jnp.einsum("ij,jkl->ikl", matrix, pixel_coords)

    coords_to_map = jnp.stack(
        [
            jnp.tile(jnp.arange(N).reshape(N, 1, 1), reps=(1, height, width)),
            jnp.tile(y_coords.reshape(1, height, width), reps=(N, 1, 1)),
            jnp.tile(x_coords.reshape(1, height, width), reps=(N, 1, 1)),
        ],
        axis=0,
    )

    img = ndi.map_coordinates(img, coords_to_map, order=order, mode=mode, cval=cval)
    img = img.reshape(-1, channel, height, width).transpose(0, 2, 3, 1)
    img = unflatten(img, original_shape)

    return jnp.clip(img, 0, 255).astype(jnp.uint8)


def rotate(
    img: chex.Array,
    angle: float,
    center: tuple[float, float] | None = None,
    order: int = 0,
    mode: str = "constant",
    cval: int = 0,
) -> chex.Array:
    """Rotate a image.

    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        angle (float): Rotation angle value in degrees.
        center (tuple[float, float], optional):
            Center of rotation (height, width). Origin is the upper left corner.
            If None, center of the image is used.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond its boundaries.
                    "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant". Default is 128.

    Returns:
        Transformed images.
    """
    if center is None:
        *_, height, width, _ = img.shape
        center = ((height - 1) / 2, (width - 1) / 2)

    center_y, center_x = center

    angle *= jnp.pi / 180
    matrix = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), center_x - center_x * jnp.cos(angle) + center_y * jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle), center_y - center_x * jnp.sin(angle) - center_y * jnp.cos(angle)],
            [0, 0, 1],
        ]
    )
    return warp_affine(img, matrix, order, mode, cval)


def rot90(img: chex.Array, n: int = 1) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        n (int): Number of rotate.

    Returns:
        Transformed images.
    """
    img, original_shape = flatten(img)
    img = jnp.rot90(img, n, axes=(1, 2))
    return unflatten(img, original_shape)


def translate(
    img: chex.Array,
    translation: tuple[int, int] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    cval: int = 0,
) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        translation: Vertical and horizontal number of pixels to translate.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond its boundaries.
                    "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant". Default is 128.

    Returns:
        Transformed images.
    """
    shift_y, shift_x = translation
    matrix = jnp.array(
        [
            [1, 0, -shift_x],  # y-axis
            [0, 1, -shift_y],  # x-axis
            [0, 0, 1],
        ]
    )
    return warp_affine(img, matrix, order, mode, cval)


def shear(
    img: chex.Array,
    angles: tuple[float, float] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    cval: int = 0,
) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        angles (tuple of float): Vertical and horizontal angles to shear.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond its boundaries.
                    "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant". Default is 128.

    Returns:
        Transformed images.
    """
    angle_y, angle_x = angles
    angle_x = angle_x * jnp.pi / 180
    angle_y = angle_y * jnp.pi / 180

    matrix = jnp.array(
        [
            [1, jnp.tan(angle_x), 0],
            [jnp.tan(angle_y), 1, 0],
            [0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    return warp_affine(img, matrix, order, mode, cval)


def hflip(img: chex.Array) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).

    Returns:
        Horizontally flipped images.
    """
    return img[..., :, ::-1, :]


def vflip(img: chex.Array) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).

    Returns:
        Vertically flipped images.
    """
    return img[..., ::-1, :, :]
