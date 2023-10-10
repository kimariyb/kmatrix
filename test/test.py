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

import kmatrix as kms

# This is the test case for the matrix and the vector.
test_vector = kms.vector([
    [1],
    [1, 2],
])
    
if __name__ == '__main__':
    print(test_vector)