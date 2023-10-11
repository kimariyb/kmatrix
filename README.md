# kmatrix

kmatrix is a third-party Python library that encapsulates a large number of operations on matrices and vectors. 

## Install

You can use the following commands to install the kmatrix.

```shell
pip install kmatrix
```

## Usage

You can use the following commands to run the kmatrix.

```python
# import package of kmatrix.
>>> import kmatrix as kmx

# Initial the matrix A.
>>> A = kmx.matrix([
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

You can also use a method to read the excel file.

```python
# import package of kmatrix.
>>> import kmatrix as kmx

# read the excel file, and print the result.
>>> print(kmx.read_excel('array.xlsx'))
    4  2  -5
    6  4  -9
    5  3  -7
```

## Document

> Document

## Independencies

- **Numpy**, `version: 1.24.4`
- **Pandas**, `version: 2.0.3`
- **Setuptools**, `version: 68.0.0`
- **Openpyxl**, `version: 3.1.2`

## License

Licensed under the `MIT License`, Version 2.0 (the "License");


