# -*- coding: utf-8 -*-
"""
core.py

This file is the base class for executing matrix operations

@author:
Kimariyb, Hsiun Ryan (kimariyb@163.com)

@address:
XiaMen University

@license:
Licensed under the MIT License.
For details, see the LICENSE file.

@data:
2023-10-08
"""

import pandas as pd
import numpy as np

class Matrix:
    """Represents a matrix and provides various matrix operations.

    Attributes:
        data (list[list[float]]): The data of the matrix, a 2D list of floats.
        rows (int): The number of rows in the matrix.
        cols (int): The number of columns in the matrix.
    """
    def __init__(self, data: list):
        """Initializes an empty matrix."""
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
    
    def __str__(self):
        """Returns a string representation of the matrix."""
        matrix_str = ""
        for row in self.data:
            row_str = " ".join(str(element) for element in row)
            matrix_str += row_str + "\n"
        return matrix_str
    
    def __eq__(self, other_matrix):
        """Compares two matrices for equality.

        Args:
            other_matrix (Matrix): The matrix to compare with.

        Returns:
            bool: True if the matrices are equal, False otherwise.
        """
        if not isinstance(other_matrix, Matrix):
            return False

        return self.data == other_matrix.data
    
    def add(self, other_matrix) -> any:
        """Adds another matrix to the current matrix.

        Args:
            other_matrix (Matrix): The matrix to be added to the current matrix.

        Returns:
            Matrix: A new matrix representing the sum of the current matrix and the other matrix.

        Raises:
            ValueError: If the dimensions of the matrices do not match.
        """
        if not self.dimensions_match():
            raise ValueError("Matrix dimensions must match for addition.")

        result_data = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                element = self.data[i][j] + other_matrix.data[i][j]
                row.append(element)
            result_data.append(row)

        result_matrix = Matrix(result_data)
        return result_matrix
    
    def multiply(self, other_matrix) -> any:
        """Multiplies the current matrix with another matrix.

        Args:
            other_matrix (Matrix): The matrix to multiply with.

        Returns:
            Matrix: A new matrix representing the result of the multiplication.

        Raises:
            ValueError: If the number of columns in the current matrix is not equal to the number of rows in the other matrix.
        """
        if self.cols != other_matrix.rows:
            raise ValueError("Number of columns in the current matrix must be equal to the number of rows in the other matrix.")

        result_data = []
        for i in range(self.rows):
            row = []
            for j in range(other_matrix.cols):
                element = 0
                for k in range(self.cols):
                    element += self.data[i][k] * other_matrix.data[k][j]
                row.append(element)
            result_data.append(row)

        result_matrix = Matrix(result_data)
        return result_matrix
    
    def transpose(self) -> any:
        """Transposes the current matrix.

        Returns:
            Matrix: A new matrix representing the transposed form of the current matrix.
        """
        transposed_data = []
        for j in range(self.cols):
            transposed_row = [self.data[i][j] for i in range(self.rows)]
            transposed_data.append(transposed_row)
        
        transposed_matrix = Matrix(transposed_data)
        return transposed_matrix
    
    def dimensions_match(self, other_matrix) -> bool:
        """Checks if the dimensions of the current matrix match with another matrix.

        Args:
            other_matrix (Matrix): The matrix to compare dimensions with.

        Returns:
            bool: True if the dimensions match, False otherwise.
        """
        return self.rows == other_matrix.rows and self.cols == other_matrix.cols

    def is_square(self) -> bool:
        """Checks if the current matrix is a square matrix.

        Returns:
            bool: True if the matrix is square, False otherwise.
        """
        return self.rows == self.cols

    def is_symmetric(self) -> bool:
        """Checks if the current matrix is a symmetric matrix.

        Returns:
            bool: True if the matrix is symmetric, False otherwise.
        """
        if not self.is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != self.data[j][i]:
                    return False

        return True

    def is_diagonal(self) -> bool:
        """Checks if the current matrix is a diagonal matrix.

        Returns:
            bool: True if the matrix is diagonal, False otherwise.
        """
        if not self.is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.data[i][j] != 0:
                    return False

        return True

    def is_identity(self) -> bool:
        """Checks if the current matrix is an identity matrix.

        Returns:
            bool: True if the matrix is an identity matrix, False otherwise.
        """
        if not self.is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if i == j:
                    if self.data[i][j] != 1:
                        return False
                else:
                    if self.data[i][j] != 0:
                        return False

        return True  

    def determinant(self) -> any:
        """Calculates the determinant of the current matrix.

        Returns:
            float: The determinant of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Matrix must be square to calculate the determinant.")

        if self.rows == 2 and self.cols == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

        determinant = 0
        for j in range(self.cols):
            cofactor = self.cofactor(0, j)
            determinant += self.data[0][j] * cofactor

        return determinant

    def cofactor(self, row, col) -> any:
        """Calculates the cofactor of the element at the specified row and column.

        Args:
            row (int): The row index.
            col (int): The column index.

        Returns:
            float: The cofactor of the element.
        """
        submatrix_data = [
            [self.data[i][j] for j in range(self.cols) if j != col]
            for i in range(self.rows) if i != row
        ]
        submatrix = Matrix(submatrix_data)
        cofactor = (-1) ** (row + col) * submatrix.determinant()
        return cofactor

    def inverse(self) -> any:
        """Calculates the inverse of the current matrix.

        Returns:
            Matrix: The inverse of the matrix.

        Raises:
            ValueError: If the matrix is not square or its determinant is zero.
        """
        if not self.is_square():
            raise ValueError("Matrix must be square to have an inverse.")

        determinant = self.determinant()
        if determinant == 0:
            raise ValueError("Matrix is not invertible (determinant is zero).")

        numpy_data = np.array(self.data)
        inverse_data = np.linalg.inv(numpy_data).tolist()
        return Matrix(inverse_data)

    def eigenvalues(self) -> any:
        """Calculates the eigenvalues of the current matrix.

        Returns:
            list[float]: A list of eigenvalues.
        """
        numpy_data = np.array(self.data)
        eigenvalues, _ = np.linalg.eig(numpy_data)
        return eigenvalues.tolist()


    def eigenvectors(self) -> any:
        """Calculates the eigenvectors of the current matrix.

        Returns:
            list[Vector]: A list of eigenvectors.
        """
        numpy_data = np.array(self.data)
        _, eigenvectors = np.linalg.eig(numpy_data)
        return eigenvectors.tolist()    
    
def init_matrix(array_2d: list) -> Matrix:
    """_summary_

    Args:
        array_2d (list): a list of arrays of matrices.

    Returns:
        Matrix: A Matrix object containing the matrix data read a array list.
    """
    return Matrix(array_2d)

def read_xlsx(xlsx_file: str) -> Matrix:
    """Retrieve matrix data from an XLSX file and return a Matrix object.

    Args:
        xlsx_file (str): Path to the XLSX file.

    Returns:
        Matrix: A Matrix object containing the matrix data read from the XLSX file.
    """
    # Read the XLSX file using pandas
    df = pd.read_excel(xlsx_file, header=None)
    
    # Convert DataFrame to a matrix format
    matrix_data = df.values.tolist()
    
    # Create a Matrix object and return it
    return Matrix(matrix_data)
