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


def get_elementary_matrix(matrix1: Matrix, matrix2: Matrix, trans_type: str) -> Matrix:
    """Returns the elementary matrix obtained from elementary transformations.

    Args:
        matrix1 (Matrix): The initial matrix.
        matrix2 (Matrix): The transformed matrix.
        trans_type (str): The type of transformation, must be either 'col' or 'row'.

    Returns:
        Matrix: The elementary matrix obtained from the elementary transformations.
    """
    if trans_type == 'col':
        # Perform column transformations
        return matrix1.inverse() * matrix2
    elif trans_type == 'row':
        # Perform row transformations
        return matrix2 * matrix1.inverse()
    else:
        raise ValueError("trans_type must be 'col' or 'row'.")
