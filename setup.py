from setuptools import setup, find_packages

setup(
    name='wobblyml',
    version='0.0.1dev1',
    description='A sample Python package',
    author='Huy Nguyen',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)