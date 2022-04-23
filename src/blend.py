"""Blend based augmentation."""
from __future__ import annotations
import math

import jax
import jax.numpy as jnp
import chex

__all__ = ["blend_image", "create_cut_mask", "create_cow_mask"]


def blend_image(img1: chex.Array, img2: chex.Array, mask: chex.Array) -> chex.Array:

    img1 = img1.astype(jnp.float32)
    img2 = img2.astype(jnp.float32)

    out = mask * img1 + (1 - mask) * img2
    return jnp.clip(out, 0, 255).astype(jnp.uint8)


def create_cut_mask(rng: chex.PRNGKey, size: tuple[int, int], mask_size: tuple[int, int]) -> chex.Array:
    """Generate a mask for CutOut or CutMix.

    Args:
        size (tuple of int): Height and width of the image.

    Returns:
        Float array. One means to remain original image, and zero means to hide by noize.
    """
    height, width = size
    mask_size_half = (mask_size[0] // 2, mask_size[1] // 2)

    x_ratio, y_ratio = jax.random.uniform(rng, (2,))
    start_indices = [jnp.int32(y_ratio * height), jnp.int32(x_ratio * width)]

    mask = jax.lax.dynamic_update_slice(
        jnp.ones((height + mask_size[0], width + mask_size[1]), dtype=jnp.float32),
        update=jnp.zeros((mask_size[0], mask_size[1]), dtype=jnp.float32),
        start_indices=start_indices,
    )

    mask = jax.lax.dynamic_slice(mask, mask_size_half, (height, width))
    return mask.reshape(height, width, 1)  # add channel dim.


_ROOT_2 = math.sqrt(2.0)
_ROOT_2_PI = math.sqrt(2.0 * math.pi)


def gaussian_kernels(sigmas, max_sigma):
    """Make Gaussian kernels for Gaussian blur.
    Args:
        sigmas: kernel sigmas as a [N] jax.numpy array
        max_sigma: sigma upper limit as a float (this is used to determine
          the size of kernel required to fit all kernels)
    Returns:
        a (N, kernel_width) jax.numpy array
    """
    sigmas = sigmas[:, None]
    size = round(max_sigma * 3) * 2 + 1
    x = jnp.arange(-size, size + 1)[None, :].astype(jnp.float32)
    y = jnp.exp(-0.5 * x**2 / sigmas**2)
    return y / (sigmas * _ROOT_2_PI)


def cow_masks(n_masks, mask_size, log_sigma_range, max_sigma, prop_range, rng_key):
    """Generate Cow Mask.
    Args:
        n_masks: number of masks to generate as an int
        mask_size: image size as a `(height, width)` tuple
        log_sigma_range: the range of the sigma (smoothing kernel)
            parameter in log-space`(log(sigma_min), log(sigma_max))`
        max_sigma: smoothing sigma upper limit
        prop_range: range from which to draw the proportion `p` that
          controls the proportion of pixel in a mask that are 1 vs 0
        rng_key: a `jax.random.PRNGKey`
    Returns:
        Cow Masks as a [v, height, width, 1] jax.numpy array
    """
    rng_k1, rng_k2 = jax.random.split(rng_key)
    rng_k2, rng_k3 = jax.random.split(rng_k2)

    # Draw the per-mask proportion p
    p = jax.random.uniform(rng_k1, (n_masks,), minval=prop_range[0], maxval=prop_range[1], dtype=jnp.float32)
    # Compute threshold factors
    threshold_factors = jax.scipy.special.erfinv(2 * p - 1) * _ROOT_2

    sigmas = jnp.exp(jax.random.uniform(rng_k2, (n_masks,), minval=log_sigma_range[0], maxval=log_sigma_range[1]))

    # Create initial noise with the batch and channel axes swapped so we can use
    # tf.nn.depthwise_conv2d to convolve it with the Gaussian kernels
    noise = jax.random.normal(rng_k3, (1,) + mask_size + (n_masks,))

    # Generate a kernel for each sigma
    kernels = gaussian_kernels(sigmas, max_sigma)
    # kernels: [batch, width] -> [width, batch]
    kernels = kernels.transpose((1, 0))
    # kernels in y and x
    krn_y = kernels[:, None, None, :]
    krn_x = kernels[None, :, None, :]

    # Apply kernels in y and x separately
    smooth_noise = jax.lax.conv_general_dilated(
        noise, krn_y, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=n_masks
    )
    smooth_noise = jax.lax.conv_general_dilated(
        smooth_noise, krn_x, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=n_masks
    )

    # [1, height, width, batch] -> [batch, height, width, 1]
    smooth_noise = smooth_noise.transpose((3, 1, 2, 0))

    # Compute mean and std-dev
    noise_mu = smooth_noise.mean(axis=(1, 2, 3), keepdims=True)
    noise_sigma = smooth_noise.std(axis=(1, 2, 3), keepdims=True)
    # Compute thresholds
    thresholds = threshold_factors[:, None, None, None] * noise_sigma + noise_mu
    # Apply threshold
    masks = (smooth_noise <= thresholds).astype(jnp.float32)
    return masks


def create_cow_mask(
    rng: chex.PRNGKey,
    size: tuple[int, int],
    sigma_range: tuple[float] = (4.0, 8.0),
    prop_range: tuple[float] = (0.0, 1.0),
):
    log_sigma_range = jnp.log(sigma_range[0]), jnp.log(sigma_range[1])
    max_sigma = sigma_range[1]
    return cow_masks(1, size, log_sigma_range, max_sigma, prop_range, rng)[0]
