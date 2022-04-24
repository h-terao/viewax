import chex


def flatten(img: chex.Array) -> tuple[chex.Array, tuple[int, ...]]:
    *_, height, width, channel = img.shape
    return img.reshape(-1, height, width, channel), img.shape


def unflatten(img: chex.Array, original_shape: tuple[int, ...]) -> chex.Array:
    batch_dims, _, _, _ = original_shape
    *_, height, width, channel = img.shape
    return img.reshape(*batch_dims, height, width, channel)
