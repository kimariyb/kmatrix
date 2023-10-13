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
    [1, 2, 3],
    [3, 2, 1],
    [0, 0, 7]
])
# 对矩阵 A 做一次列变换
# 交换了第二列和第三列
# [1, 3, 2]
# [3, 1, 2]
# [0, 7, 0]
B = A.swap_col(1, 2)
