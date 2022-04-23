<div align="center">

# Augmentor

</div>

Augmentor is an image augmentation library for JAX inspired by [imax](https://github.com/4rtemi5/imax).

## Requirements
- Python >= 3.7
- jax
- chex

## Note
- Input image is expected to be an uint8 RGB array that has a shape of (height, width, channel).
- Most of functions are not JIT-ed, so you should compile the augmentation part in your code.
- Use vmap or pmap to transform multiple images at the same time.

## Example

```python
import jax
import jax.numpy as jnp
from PIL import Image

import augmentor as A
import augmentor.functional as AF

rng = jax.random.PRNGKey(0)
image = jnp.array(Image.open(...))

# Cropping.
image = A.random_crop(rng, image, (32, 32))

# Color augmentation.
image = AF.autocontrast(image)

# Geometrical augmentation.
image = AF.rotate(image, 30)

# CutOut.
h, w, _ = image.shape
mask = A.blend.create_cut_mask(rng, (h, w), (h//2, w//2))
image = A.blend.blend_image(image, jnp.full_like(image, fill_value.image.mean()), mask)

# Normalize and feed it to DNNs.
image = jnp.float32(image) / 255.0
```