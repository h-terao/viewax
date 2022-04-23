from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy import ndimage as ndi
import chex


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
    img: chex.Array, matrix: chex.Array, order: int = 0, mode: str = "constant", fill: int = 0
) -> chex.Array:
    """Warp an image according to a given coordinate transformations.

    Args:
        img (array): An image array (HWC).
        matrix (array): An affine transformation matrix.
        order (int): Interpolation. 0 means NEAREST, and 1 is LINEAR.
        mode (str): How to fill area out of images.
        fill (int): Color to fill area out of images. Only used when mode=constant.

    Notes:
        Transformation matrix
    """
    img = jnp.float32(img)
    height, width, _ = img.shape

    x_t, y_t = jnp.meshgrid(
        jnp.arange(0, width),
        jnp.arange(0, height),
    )
    I = jnp.ones_like(x_t)
    pixel_coords = jnp.stack([x_t, y_t, I]).astype(jnp.float32)

    x_coords, y_coords, _ = jnp.einsum("ij,jkl->ikl", matrix, pixel_coords)
    coords = jnp.stack([y_coords, x_coords], axis=0)

    @jax.jit
    def body(carry: chex.Array, x: chex.Array) -> tuple[chex.Array, chex.Array]:
        y = ndi.map_coordinates(x, coords, order=order, mode=mode, cval=fill)
        return carry, y

    _, img = jax.lax.scan(f=body, init=jnp.zeros(()), xs=img.transpose(2, 0, 1))
    img = img.transpose(1, 2, 0)  # CHW -> HWC

    return jnp.clip(img, 0, 255).astype(jnp.uint8)


def rotate(
    img: chex.Array,
    angle: float,
    center: tuple[float, float] | None = None,
    order: int = 0,
    mode: str = "constant",
    fill: int = 0,
) -> chex.Array:
    """Rotate a image.

    Args:
        img (chex.Array): Image to rotate.
        angle (float): Rotation angle value in degrees.
        center (tuple[float, float], optional):
            Center of rotation (height, width). Origin is the upper left corner.
            If None, center of the image is used.
    """
    if center is None:
        height, width, _ = img.shape
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
    return warp_affine(img, matrix, order, mode, fill)


def rot90(img: chex.Array, n: int = 1) -> chex.Array:
    return jnp.rot90(img, n)


def translate(
    img: chex.Array,
    translation: tuple[int, int] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    fill: int = 0,
) -> chex.Array:
    """
    Args:
        translation: Vertical and horizontal translations.
    """
    shift_y, shift_x = translation
    matrix = jnp.array(
        [
            [1, 0, -shift_x],  # y-axis
            [0, 1, -shift_y],  # x-axis
            [0, 0, 1],
        ]
    )
    return warp_affine(img, matrix, order, mode, fill)


def shear(
    img: chex.Array,
    angles: tuple[float, float] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    fill: int = 0,
) -> chex.Array:
    """
    Args:
        angles (tuple of float): Angle to shear. Vertical and horizontal.
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
    return warp_affine(img, matrix, order, mode, fill)


def hflip(img: chex.Array) -> chex.Array:
    return img[:, ::-1]


def vflip(img: chex.Array) -> chex.Array:
    return img[::-1]


if __name__ == "__main__":
    from PIL import Image
    import numpy

    img = Image.open("icon11.png")
    # img = jnp.zeros([8, 10, 3])  # H, W, C
    img = jnp.array(img, dtype=jnp.uint8)

    img = rotate(img, 45, center=(0, 0))
    # img = translate(img, x=100, y=200)
    # img = translate(img, y=100)
    # img = shear_x(img, 20)
    # img = shear_y(img, 20)

    img = numpy.array(img)
    Image.fromarray(img).save("transformed_icon.png")
