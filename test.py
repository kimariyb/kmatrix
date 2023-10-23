# -*- coding: utf-8 -*-
"""
test.py

This file is part of the test module.

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
import kmatrix as kmx

# 这里有一个矩阵 A
A = kmx.matrix([
    [1, 0],
    [0, 1]
])

B = kmx.matrix([
    [0, 1],
    [1, 0]
])
print(kmx.Vector.create_identity(3, 'row'))
print(kmx.kronecker_product(A, B))