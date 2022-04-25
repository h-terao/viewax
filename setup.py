from setuptools import setup, find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="viewax",
    version="0.0.1",
    description="An image augmentation library for JAX.",
    packages=find_packages(),
    install_requires=_requires_from_file("requirements.txt"),
)
