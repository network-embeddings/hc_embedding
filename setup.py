#!/usr/bin/env python


from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = [x.strip() for  x in f]

setup(
    name="hc_embedding",
    version="0.1",
    packages=find_packages(),
    scripts=['hc_embedding/hce.py'],
    install_requires=requirements,
    author="Multiple",
    author_email="",
    description="Hyperbolic coalescent embeddings for plotting networkx graphs.",
    license="MIT",
    url="https://github.com/network-embeddings/hc_embedding",
)
