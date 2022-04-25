from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import viewax


def save(img, filename):
    out_dir = Path("transformed")
    out_dir.mkdir(exist_ok=True)
    img = np.array(img)
    Image.fromarray(img).save(out_dir / filename)


rng = jax.random.PRNGKey(1)
img = jnp.array(Image.open("lena.jpg"))

save(jax.jit(viewax.solarize)(img), "solarize.jpg")
save(jax.jit(viewax.color)(img, 1.5), "color.jpg")
save(jax.jit(viewax.contrast)(img, 1.5), "contrast.jpg")
save(jax.jit(viewax.brightness)(img, 1.5), "brighrness.jpg")
save(jax.jit(viewax.posterize)(img, 4), "posterize.jpg")
save(jax.jit(viewax.autocontrast)(img), "autocontrast.jpg")
save(jax.jit(viewax.sharpness)(img, 1.5), "sharpness.jpg")
save(jax.jit(viewax.equalize)(img), "equalize.jpg")
save(jax.jit(viewax.invert)(img), "invert.jpg")
save(jax.jit(viewax.rotate, static_argnames="mode")(img, 30, mode="reflect"), "rotate.jpg")
save(jax.jit(viewax.rot90, static_argnames="n")(img, 2), "rot90.jpg")
save(jax.jit(viewax.translate, static_argnames="cval")(img, (32, 64), cval=128), "translate.jpg")
save(jax.jit(viewax.shear, static_argnames="cval")(img, (10, 30), cval=128), "shear.jpg")
save(jax.jit(viewax.hflip)(img), "hflip.jpg")
save(jax.jit(viewax.vflip)(img), "vflip.jpg")
save(jax.jit(viewax.random_crop, static_argnames="size")(rng, img, (64, 64)), "random_crop.jpg")
save(jax.jit(viewax.center_crop, static_argnames="size")(img, (64, 64)), "center_crop.jpg")

save(viewax.cutout(rng, img, mask_size=(32, 32), cval=128), "cutout.jpg")
