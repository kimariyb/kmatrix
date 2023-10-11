# -*- coding: utf-8 -*-
"""
line.py

This file is some of the methods for linear spaces 
and linear transformations and other advanced applications in matrix theory.

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

from .core import *

def matrix(array: list) -> Matrix:
    """Creates a Matrix object from a nested list.

    Args:
        array (list): A nested list representing the matrix.

    Returns:
        Matrix: A Matrix object representing the input matrix.
    """
    return Matrix(array)

def vector(array: list) -> Vector:
    """Creates a Vector object from a list.

    Args:
        array (list): A list representing the vector.

    Returns:
        Vector: A Vector object representing the input vector.
    """
    return Vector(array)

