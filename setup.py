from setuptools import setup, find_packages

setup(
    name='vIQA',
    version='0.1.0',
    description='a Python package for volumetric image quality assessment',
    license='BSD-3-Clause License',
    author='Lukas Behammer',
    author_email='lukas.behammer@fh-wels.at',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'numpy',
        'scikit-image',
        'scipy',
        'torch',
        'piq',
        'pytest',
        'setuptools',
    ],
)
