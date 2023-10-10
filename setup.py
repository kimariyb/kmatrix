# -*- coding: utf-8 -*-
"""
setup.py

This file is used to configure the setuptools package

@author:
Kimariyb, Hsiun Ryan (kimariyb@163.com)

@address:
XiaMen University, School of electronic science and engineering

@license:
Licensed under the MIT License.
For details, see the LICENSE file.

@data:
2023-10-09
"""

import setuptools

import kmatrix as kmx

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=kmx.__appname__,
    version=kmx.__version__,
    author=kmx.__author__,
    description=kmx.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=kmx.__website__,
    license='MIT',
    keywords='matrix',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        'Development Status :: 1 - Alpha',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.13',
)