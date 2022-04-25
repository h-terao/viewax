from __future__ import annotations

import jax
import jax.numpy as jnp
import chex


def mixup(rng: chex.PRNGKey, img1: chex.Array, img2: chex.Array, alpha: float) -> tuple[chex.Array, float]:
    """
    Note:
        Output image is not uint8 array.
    """
    p = jax.random.beta(rng, alpha, alpha, shape=())
    img1 = jnp.float32(img1)
    img2 = jnp.float32(img2)

    return p * img1 + (1 - p) * img2
