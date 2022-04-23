<div align="center">

# Viewax

</div>

Viewax is an image augmentation library for JAX. This project is now in progress and some functions are not tested enough.

## Installation
```bash
git clone git@github.com:h-terao/viewax.git
cd viewax
pip install -e .
```


## Note
- Input image is expected to be an uint8 RGB array that has a shape of (height, width, channel).
- Most of functions are not JIT-ed, so you should compile the augmentation part in your code.
- Use vmap or pmap to transform multiple images at the same time.

## Example

```python
import jax
import jax.numpy as jnp
from PIL import Image

import viewax
import viewax.functional as VF
import viewax.blend as vblend

rng = jax.random.PRNGKey(0)
image = jnp.array(Image.open(...))

# Cropping.
image = viewax.random_crop(rng, image, (32, 32))

# Color augmentation.
image = VF.autocontrast(image)

# Geometrical augmentation.
image = VF.rotate(image, 30)

# CutOut.
h, w, _ = image.shape
mask = vblend.create_cut_mask(rng, (h, w), (h//2, w//2))
image = vblend.blend_image(image, jnp.full_like(image, fill_value=image.mean()), mask)

# Normalize and feed it to DNNs.
image = jnp.float32(image) / 255.0
```

## Future Work
- Support NHWC format.
