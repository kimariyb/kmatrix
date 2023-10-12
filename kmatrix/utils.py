# -*- coding: utf-8 -*-
"""
utils.py

This file is part of the kmatrix package.

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

import pandas as pd


def read_excel(excel_file: str) -> Matrix:
    """Retrieve matrix data from an XLSX file and return a Matrix object.

    Args:
        excel_file (str): Path to the XLSX file.

    Returns:
        Matrix: A Matrix object containing the matrix data read from the XLSX file.
    """
    # Read the XLSX file using pandas
    df = pd.read_excel(excel_file, header=None)
    # Convert DataFrame to a matrix format
    matrix_data = df.values.tolist()
    # Create a Matrix object and return it
    return Matrix(matrix_data)
