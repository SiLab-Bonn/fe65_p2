#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages
from platform import system

import numpy as np
import os

f = open('VERSION', 'r')
version = f.readline().strip()
f.close()

author = 'Tomasz Hemperek'
author_email = 'hemeprek@physik.uni-bonn.de'

# requirements for core functionality
install_requires = ['basil-daq==2.4.1', 'bitarray>=0.8.1', 'matplotlib', 'numpy', 'progressbar-latest>=2.4', 'tables', 'pyyaml', 'scipy']

setup(
    name='fe65p2',
    version=version,
    description='DAQ for FE65P2 prototype',
    url='https://github.com/SiLab-Bonn/fe65p2',
    license='',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),  
    include_package_data=True,  
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'fe65p2': ['*.yaml', '*.bit']},
    platforms='any'
)
