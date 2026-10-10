#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:48:46 2026

@author: nayandusoruth
"""

import sympy

v, o = sympy.symbols('\nu \sigma')
lambdaMatrix = sympy.matrices.Matrix([[v,o],[1-v,1-o]])
conjugateMatrix = lambdaMatrix.H
outputMatrix = sympy.simplify(lambdaMatrix * conjugateMatrix)
print(sympy.latex(lambdaMatrix), "\cdot", sympy.latex(conjugateMatrix),"=", sympy.latex(outputMatrix))
