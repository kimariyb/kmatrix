import setuptools

import matrix as ma

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=ma.__appname__,
    version=ma.__version__,
    author=ma.__author__,
    description=ma.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=ma.__website__,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.13',
)