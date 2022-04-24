from __future__ import annotations
import jax
import jax.numpy as jnp
import chex


def mixup(alpha: float):
    """Create mixup masks.

    Note:
        This function returns max(v, 1-v), where v~Beta(alpha, alpha).
        It means that img1 is always weighted heavier than img2 in blend function.
    """

    def fn(rng: chex.PRNGKey, size: tuple[int, int, int]) -> chex.Array:
        n = size[0]
        masks = jax.random.beta(rng, alpha, alpha, (n, 1, 1, 1))
        return jnp.maximum(masks, 1 - masks)

    return fn
