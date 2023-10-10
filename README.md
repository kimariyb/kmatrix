# kmatrix

kmatrix is a third-party Python library that encapsulates a large number of operations on matrices and vectors. 

## Install

```shell
pip install kmatrix
```

## Usage

```python
# import package of kmatrix.
>>> from kmatrix import *

# Initial the matrix A.
>>> A = Matrix([
        [1, 2, 0],
        [1, 1, 1],
        [1, 0, 1]
    ])

# Inverse the matrix A to get the matrix B.
>>> B = A.inverse()

# Let matrix A be multiplied by matrix B, and print the result.
>>> print(Matrix.multiply(A, B))
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0

# Let matrix A be added by matrix B, and print the result.
>>> print(Matrix.add(A, B))
    2.0 0.0 2.0
    1.0 2.0 0.0
    0.0 2.0 0.0
```

## Independencies

- **Numpy**, `version: 1.24.4`
- **Pandas**, `version: 2.0.3`
- **Setuptools**, `version: 68.0.0`

## License

Licensed under the `MIT License`, Version 2.0 (the "License");


