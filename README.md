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
- Input image is expected to be an uint8 RGB array that has a shape of (..., height, width, channel).
    - Applying augmentation to images with different parameters is not supported. To achieve such augmentation, use vmap or pmap.
- Most of functions are not JIT-ed, so you should compile the augmentation part in your code.

## Example

See example/test_all.py
