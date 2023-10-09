# kmatrix

kmatrix is a third-party Python library that encapsulates a large number of operations on matrices and vectors. 

## Installation

```shell
pip install kmatrix
```

## Quickly Start

```python
>>> from kmatrix import *

>>> A = Matrix([
        [1, 2, 0],
        [1, 1, 1],
        [1, 0, 1]
    ])

>>> print(A.inverse())

     1.0 -2.0  2.0
    -0.0  1.0 -1.0
    -1.0  2.0 -1.0
```

## Independencies

- **Numpy**, `version: 1.24.4`
- **Pandas**, `version: 2.0.3`
- **Setuptools**, `version: 68.0.0`

## License

Licensed under the `MIT License`, Version 2.0 (the "License");


