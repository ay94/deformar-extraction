from setuptools import setup, find_packages

setup(
    name='experiment_utils',
    version='0.1.0',
    packages=find_packages(include=['experiment_utils', 'experiment_utils.*'])
)