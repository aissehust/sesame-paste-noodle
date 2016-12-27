from setuptools import find_packages
from setuptools import setup

install_requires = [
    'numpy',
    'theano',
    'pyyaml',
    'h5py',
]

setup(
    name="TheFramework",
    version="0.0.1",
    description="A nn lib",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
)
