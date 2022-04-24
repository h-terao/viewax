from __future__ import annotations
import jax
import jax.numpy as jnp
import chex


def cutout(mask_size: tuple[int, int]) -> chex.Array:
    """Generate a mask for CutOut and CutMix.

    Args:
        size (tuple of int): Height and width of the image.

    Returns:
        Float array. One means to remain original image, and zero means to hide by noize.
    """

    def create_mask(rng: chex.PRNGKey, height: int, width: int):
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

    @jax.jit
    def fn(rng: chex.PRNGKey, size: tuple[int, int, int]) -> chex.Array:
        batch_size, height, width = size
        _, masks = jax.lax.scan(
            lambda carry, rng: (carry, create_mask(rng, height, width)),
            jnp.zeros(()),
            jax.random.split(rng, batch_size),
        )
        return masks

    return fn
