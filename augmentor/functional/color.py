# Modify from: https://github.com/4rtemi5/imax/blob/master/imax/color_transforms.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import chex

__all__ = [
    "solarize",
    "solarize_add",
    "color",
    "contrast",
    "brightness",
    "posterize",
    "autocontrast",
    "sharpness",
    "equalize",
    "invert",
]


@jax.jit
def blend(x: chex.Array, y: chex.Array, factor: float) -> chex.Array:
    """compute (1-factor)*x + factor*y."""
    dtype = x.dtype

    x = x.astype("int32")
    y = y.astype("int32")

    scaled = factor * (y - x)
    return jnp.clip(x + scaled, 0.0, 255.0).astype(dtype)


@jax.jit
def rgb_to_grayscale(img: chex.Array) -> chex.Array:
    output = jnp.dot(img[..., :3], jnp.array([0.2989, 0.5870, 0.1140]).astype("uint8"))
    output = jnp.stack((output,) * 3, axis=-1)
    return output


def solarize(img: chex.Array, threshold: int = 128) -> chex.Array:
    degenerate = jnp.where(img < threshold, img, 255 - img)
    return degenerate


def solarize_add(img: chex.Array, addition: int = 0, threshold: int = 128) -> chex.Array:
    """
    Args:
        addition (int): Addition value. It is expected in [-128, 128].
    """
    added_img = img.astype("int32") + addition
    added_img = jnp.clip(added_img, 0, 255).astype("uint8")
    degenerate = jnp.where(img < threshold, added_img, img)
    return degenerate


# factor: (0, 2)
def color(img: chex.Array, factor: float) -> chex.Array:
    degenerate = rgb_to_grayscale(img)
    degenerate = blend(degenerate, img, factor)
    return degenerate


def contrast(img: chex.Array, factor: float) -> chex.Array:
    degenerate = rgb_to_grayscale(img)[:, :, 0]  # get the first channel.
    degenerate = degenerate.astype("int32")

    hist, _ = jnp.histogram(degenerate, bins=256, range=(0, 255))
    mean = jnp.sum(hist.astype("float32")) / 256.0

    degenerate = jnp.full_like(degenerate, fill_value=mean, dtype="float32")
    degenerate = jnp.clip(degenerate, 0.0, 255.0)
    degenerate = jnp.stack((degenerate,) * 3).astype(img.dtype)
    degenerate = blend(degenerate, img, factor)
    return degenerate


def brightness(img: chex.Array, factor: float) -> chex.Array:
    degenerate = jnp.zeros_like(img)
    degenerate = blend(degenerate, img, factor).astype(img.dtype)
    return degenerate


def posterize(img: chex.Array, bits: int) -> chex.Array:
    """
    Args:
        bits (int): Bits to shift. [0, 8]
    """
    shift = 8 - bits.astype("int32")
    degenerate = jnp.left_shift(jnp.right_shift(img, shift), shift)
    return degenerate.astype("uint8")


def autocontrast(img: chex.Array) -> chex.Array:
    @jax.jit
    def scale_channel(x: chex.Array):
        low = jnp.min(x).astype("float32")
        high = jnp.max(x).astype("float32")

        def _scale_values(v: chex.Array) -> chex.Array:
            scale = 255.0 / (high - low)
            offset = -low * scale
            v = v.astype("float32") * scale + offset
            return jnp.clip(v, 0, 255).astype("uint8")

        return jax.lax.cond(high > low, _scale_values, lambda v: v, x)

    red = scale_channel(img[:, :, 0])
    blue = scale_channel(img[:, :, 1])
    green = scale_channel(img[:, :, 2])

    return jnp.stack([red, blue, green], axis=2)


def sharpness(img: chex.Array, factor: float) -> chex.Array:
    """
    Implements Sharpness function from PIL using Jax ops.
    Args:
        image: image tensor
        factor: float factor
    Returns:
        Augmented image.
    """
    orig_img = img
    img = img.astype("float32")
    # Make image 4D for conv operation.
    img = jnp.expand_dims(img, 0)
    # SMOOTH PIL Kernel.
    kernel = jnp.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype="float32") / 13.0
    kernel = jnp.reshape(kernel, (3, 3, 1, 1))
    kernel = jnp.tile(kernel, [1, 1, 1, 3])
    degenerate = jax.lax.conv_general_dilated(
        jnp.transpose(img, [0, 3, 1, 2]),  # lhs = NCHW image tensor
        jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        "VALID",  # padding mode
        feature_group_count=3,
    )
    degenerate = jnp.clip(degenerate, 0.0, 255.0)
    degenerate = jnp.squeeze(degenerate.astype("uint8"), 0)
    degenerate = jnp.transpose(degenerate, [1, 2, 0])
    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = jnp.ones_like(degenerate)
    padded_mask = jnp.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = jnp.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = jnp.where(jnp.equal(padded_mask, 1), padded_degenerate, orig_img)

    # Blend the final result.
    return blend(result, orig_img, factor)


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

    scaled_channel_1 = scale_channel(img[:, :, 0])
    scaled_channel_2 = scale_channel(img[:, :, 1])
    scaled_channel_3 = scale_channel(img[:, :, 2])
    degenerate = jnp.stack([scaled_channel_1, scaled_channel_2, scaled_channel_3], 2)

    return degenerate


def invert(img: chex.Array) -> chex.Array:
    return 255 - img
