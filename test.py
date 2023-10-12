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

A = kmx.matrix([
    [1, 2, 3],
    [3, 2, 1],
    [0, 0, 7]
])

B = kmx.matrix([
    [1, 2, 3],
    [3, 2, 1],
    [2, 4, 13]
])

print(A.swap_row(0, 1))

