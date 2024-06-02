from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="experiment_utils",
    version="0.1.0",
    packages=find_packages(include=["experiment_utils", "experiment_utils.*"]),
    author="Ahmed Younes",
    author_email="ahmed.younes.sam@gmail.com",
    description="A utility package for experiment management.",
    install_requires=required
)
