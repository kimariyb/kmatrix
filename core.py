# -*- coding: utf-8 -*-
"""
core.py

This file is the base class for executing matrix and vector operations

@author:
Kimariyb, Hsiun Ryan (kimariyb@163.com)

@address:
XiaMen University, School of electronic science and engineering

@license:
Licensed under the MIT License.
For details, see the LICENSE file.

@data:
2023-10-08
"""

from typing_extensions import override
import pandas as pd
import numpy as np

class Matrix:
    """Represents a matrix and provides various matrix operations.

    Attributes:
        data (list[list[float]]): The data of the matrix, a 2D list of floats.
        rows (int): The number of rows in the matrix.
        cols (int): The number of columns in the matrix.
        rank (int): The rank of the matrix.
        is_square (bool): Indicates whether the matrix is square or not.
        is_inverse (bool): Indicates whether the matrix has an inverse or not.
        is_symmetric (bool): Indicates whether the matrix is symmetric or not.
        is_diagonal (bool): Indicates whether the matrix is diagonal or not.
        is_identity (bool): Indicates whether the matrix is an identity matrix or not.
    """
    def __init__(self, data: list):
        """Initializes a matrix with the given data.

        Args:
            data (list): The data of the matrix, a 2D list of floats.
        """
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        self.rank = self._rank()
        self.is_square = self._is_square()
        self.is_inverse = self._is_inverse()
        self.is_symmetric = self._is_symmetric()
        self.is_diagonal = self._is_diagonal()
        self.is_identity = self._is_identity()
        self.positive_definite = self._positive_definite()
    
    def __str__(self):
        """Returns a string representation of the matrix."""
        matrix_str = ""
        max_element_width = 0

        # Find the maximum width of any element in the matrix
        for row in self.data:
            for element in row:
                max_element_width = max(max_element_width, len(str(element)))

        # Build the matrix string with aligned elements
        for row in self.data:
            row_str = " ".join(f"{element:>{max_element_width}}" for element in row)
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

    def _is_square(self) -> bool:
        """Checks if the current matrix is a square matrix.

        Returns:
            bool: True if the matrix is square, False otherwise.
        """
        return self.rows == self.cols

    def _is_symmetric(self) -> bool:
        """Checks if the current matrix is a symmetric matrix.

        Returns:
            bool: True if the matrix is symmetric, False otherwise.
        """
        if not self._is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != self.data[j][i]:
                    return False

        return True

    def _is_diagonal(self) -> bool:
        """Checks if the current matrix is a diagonal matrix.

        Returns:
            bool: True if the matrix is diagonal, False otherwise.
        """
        if not self._is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.data[i][j] != 0:
                    return False

        return True

    def _is_identity(self) -> bool:
        """Checks if the current matrix is an identity matrix.

        Returns:
            bool: True if the matrix is an identity matrix, False otherwise.
        """
        if not self._is_square():
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
    
    def _is_inverse(self) -> bool:
        """
        Checks if the current matrix is invertible.

        Returns:
            bool: True if the matrix is invertible, False otherwise.

        Raises:
            ValueError: If the matrix is not square.
        """
        if not self._is_square():
            return False

        return self.determinant() != 0
    
    def _rank(self) -> int:
        """Calculates the rank of the matrix.

        Returns:
            int: The rank of the matrix.

        Note:
            The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix.
        """
        numpy_data = np.array(self.data)
        return np.linalg.matrix_rank(numpy_data)
    
    def _positive_definite(self) -> bool:
        """Checks if the matrix is positive definite.

        Returns:
            bool: True if the matrix is positive definite, False otherwise.
        """
        if self._is_square() and self._is_symmetric():
            matrix_np = np.array(self.data)
            eigenvalues = np.linalg.eigvals(matrix_np)
            return np.all(eigenvalues > 0)
        else:
            return False
        
    def _cofactor(self, row: int, col: int) -> float:
        """Calculates and returns the cofactor of the specified element in the matrix.

        Args:
            row (int): The row index of the element.
            col (int): The column index of the element.

        Returns:
            float: The cofactor of the element.
        """
        submatrix_data = []
        for i in range(self.rows):
            if i == row:
                continue
            row_data = []
            for j in range(self.cols):
                if j == col:
                    continue
                row_data.append(self.data[i][j])
            submatrix_data.append(row_data)
        submatrix = Matrix(submatrix_data)
        return (-1) ** (row + col) * submatrix.determinant()
        
    def determinant(self) -> float:
        """Calculates the determinant of the current matrix.

        Returns:
            float: The determinant of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
        if not self._is_square():
            raise ValueError("Matrix must be square to calculate the determinant.")
        
        numpy_data = np.array(self.data)
        
        return np.round(np.linalg.det(numpy_data), decimals=4)
    
    def multiply(self, other_matrix) -> 'Matrix':
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
    
    def scalar_multiply(self, scalar: float) -> 'Matrix':
        """Multiplies the matrix by a scalar value.

        Args:
            scalar (float): The scalar value to multiply the matrix by.

        Returns:
            Matrix: A new Matrix object representing the result of the scalar multiplication.
        """
        result = [[scalar * element for element in row] for row in self.data]
        return Matrix(result)
        
    def transpose(self) -> 'Matrix':
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

    def inverse(self) -> 'Matrix':
        """Calculates the inverse of the current matrix.

        Returns:
            Matrix: The inverse of the matrix.

        Raises:
            ValueError: If the matrix is not square or its determinant is zero.
        """
        if not self._is_inverse():
            raise ValueError("Matrix is not invertible (determinant is zero).")

        numpy_data = np.array(self.data)
        inverse_data = np.linalg.inv(numpy_data).tolist()
        return Matrix(np.round(inverse_data, decimals=4))

    def eigenvalues(self) -> any:
        """Calculates the eigenvalues of the current matrix.

        Returns:
            list[float]: A list of eigenvalues.
        """
        numpy_data = np.array(self.data)
        eigenvalues, _ = np.linalg.eig(numpy_data)
        rounded_eigenvalues = np.round(eigenvalues, decimals=4)
        return rounded_eigenvalues.tolist()

    def eigenvectors(self) -> 'Matrix':
        """Calculates the eigenvectors of the current matrix.

        Returns:
            list[Vector]: A list of eigenvectors.
        """
        numpy_data = np.array(self.data)
        _, eigenvectors = np.linalg.eig(numpy_data)
        rounded_eigenvectors = np.round(eigenvectors, decimals=4)
        return rounded_eigenvectors.tolist()    
    
    def adjoint_matrix(self) -> 'Matrix':
        """Calculates and returns the adjoint matrix of the matrix.

        Returns:
            Matrix: The adjoint matrix.
        """
        if self._is_square():
            adjoint_data = [[0.0] * self.cols for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(self.cols):
                    cofactor = self._cofactor(i, j)
                    adjoint_data[j][i] = cofactor
            return Matrix(adjoint_data)
        else:
            raise ValueError("The matrix is not square.")
    
    @staticmethod
    def add(matrix1, matrix2) -> 'Matrix':
        """Adds two matrices.

        Args:
            matrix1 (Matrix): The first matrix.
            matrix2 (Matrix): The second matrix.

        Returns:
            Matrix: A new matrix representing the sum of the two matrices.

        Raises:
            ValueError: If the dimensions of the matrices do not match.
        """
        if not Matrix.dimensions_match(matrix1, matrix2):
            raise ValueError("Matrix dimensions must match for addition.")

        result_data = []
        for i in range(matrix1.rows):
            row = []
            for j in range(matrix1.cols):
                element = matrix1.data[i][j] + matrix2.data[i][j]
                row.append(element)
            result_data.append(row)

        return Matrix(result_data)

    @staticmethod
    def subtract(matrix1, matrix2) -> 'Matrix':
        """Subtracts one matrix from another.

        Args:
            matrix1 (Matrix): The matrix to subtract from.
            matrix2 (Matrix): The matrix to subtract.

        Returns:
            Matrix: A new Matrix object representing the result of the subtraction.

        Raises:
            ValueError: If the dimensions of the matrices do not match.
        """
        if not Matrix.dimensions_match(matrix1, matrix2):
            raise ValueError("Matrix dimensions must match for subtraction.")

        result = [[matrix1.data[i][j] - matrix2.data[i][j] for j in range(matrix1.cols)] 
                for i in range(matrix1.rows)]
        return Matrix(result)
    
    @staticmethod
    def dimensions_match(matrix1, matrix2) -> bool:
        """Checks if the dimensions of two matrices match for addition or subtraction.

        Args:
            matrix1 (Matrix): The first matrix.
            matrix2 (Matrix): The second matrix.

        Returns:
            bool: True if the dimensions match, False otherwise.
        """
        return matrix1.rows == matrix2.rows and matrix1.cols == matrix2.cols
    
    @staticmethod
    def create_identity(dimension: int) -> 'Matrix':
        """Creates an identity matrix of the specified dimension.

        Args:
            dimension (int): The dimension of the identity matrix.

        Returns:
            Matrix: A new identity matrix with the specified dimension.
        """
        data = [[1 if i == j else 0 for j in range(dimension)] for i in range(dimension)]
        return Matrix(data)
    
    @staticmethod
    def create_zeros(dimension: int) -> 'Matrix':
        """Creates a zero matrix of the specified dimension.

        Args:
            dimension (int): The dimension of the matrix.

        Returns:
            Matrix: The zero matrix.
        """
        data = [[0.0] * dimension for _ in range(dimension)]
        return Matrix(data)

    @staticmethod
    def read_xlsx(xlsx_file: str) -> 'Matrix':
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

class Vector(Matrix):
    """A class representing a vector."""
    def __init__(self, data: list):
        """Initializes a vector with the given data.

        Args:
            data (list): The data of the vector, a 1D list of floats.
        """
        super().__init__(data)
        self.magnitude = self._magnitude()
        
    def _magnitude(self):
        """Calculates the magnitude (length) of the vector.
        
        Returns:
            float: The magnitude of the vector.
        """
        vector = np.array(self.data)
        magnitude = np.linalg.norm(vector)
        return np.round(magnitude, decimals=4)
    
    def dot_product(self, other_vector):
        """Calculates the dot product of the vector with another vector.

        Args:
            other_vector (Vector): The other vector to calculate the dot product with.

        Returns:
            float: The dot product value.
        """
        vector1 = np.array(self.data)
        vector2 = np.array(other_vector.data)
        dot_product_value = np.dot(vector1, vector2)
        return dot_product_value

    def cross_product(self, other_vector):
        """Calculates the cross product of the vector with another vector.

        Args:
            other_vector (Vector): The other vector to calculate the cross product with.

        Returns:
            Vector: The cross product vector.
        """
        vector1 = np.array(self.data)
        vector2 = np.array(other_vector.data)
        cross_product_vector = np.cross(vector1, vector2)
        return Vector(cross_product_vector.tolist())
    
    @override
    @staticmethod
    def dimensions_match(vector1, vector2) -> bool:
        """Checks if the dimensions of two vectors match for addition or subtraction.

        Args:
            vector1 (Vector): The first vector.
            vector2 (Vector): The second vector.

        Returns:
            bool: True if the dimensions match, False otherwise.
        """
        return len(vector1.data) == len(vector2.data)
    
    @override
    @staticmethod
    def add(vector1, vector2) -> 'Vector':
        ...
        
    @override
    @staticmethod
    def subtract(matrix1, matrix2) -> 'Vector':
        ...
        
    @override
    @staticmethod
    def create_identity(dimension: int) -> 'Vector':
        ...
    
    @override
    @staticmethod
    def create_zeros(dimension: int) -> 'Vector':
        ...