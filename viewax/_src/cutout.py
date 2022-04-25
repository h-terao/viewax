from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import chex

from .utils import flatten, unflatten

__all__ = ["cutout", "cutmix"]


def _get_mask(rng: chex.PRNGKey, img: chex.Array, mask_size: tuple[int, int]):
    mask_size_half = (mask_size[0] // 2, mask_size[1] // 2)

    img, original_shape = flatten(img)
    batch_size, height, width, channel = img.shape

    mask = jnp.ones([height, width])
    mask = jnp.pad(mask, [mask_size_half[0], mask_size_half[1]], mode="reflect")

    start_indices = jnp.int32(jnp.array([height, width]) * jax.random.uniform(rng, (2,)))

    mask = jax.lax.dynamic_update_slice(
        mask,
        update=jnp.zeros(mask_size),
        start_indices=start_indices,
    )

    mask = jax.lax.dynamic_slice(mask, mask_size_half, (height, width))
    mask = jnp.tile(mask.reshape(1, height, width, 1), (batch_size, 1, 1, channel))

    return unflatten(mask, original_shape)


@partial(jax.jit, static_argnums=(2,))
def cutout(rng, img, mask_size: tuple[int, int], cval: int = 128):
    mask = _get_mask(rng, img, mask_size)
    return jnp.where(mask, img, jnp.full_like(img, cval))


@partial(jax.jit, static_argnums=(3,))
def cutmix(rng, img1, img2, mask_size: tuple[int, int]):
    mask = _get_mask(rng, img1, mask_size).astype(jnp.bool)
    img = jnp.where(mask, img1, img2)
    return img, jnp.float32(mask).mean()
