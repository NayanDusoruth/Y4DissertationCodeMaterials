#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:48:46 2026

@author: nayandusoruth
"""

# =============================================================================
# library imports
# =============================================================================
import sympy

# =============================================================================
# basic expressions
# https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html
# https://docs.sympy.org/latest/tutorials/intro-tutorial/printing.html
# =============================================================================

# define variables
x = sympy.Symbol('x')
y, z = sympy.symbols('y z')

# define expression
expr = x + y
print("basic expression: ", expr)

# evaluate expression
args = {x:1, y:2}
out = expr.evalf(subs=args)
print("expression ", expr, " w/args ", args, " evaluates to: ", out)

# substitution - substitute x->z  - Note can also sub in numeric values
expr2 = expr.subs(x,z)
print("expression substitution: ", expr2)

# convert string to sympy expression - may be useful for lambda function conversion
a,b,c = sympy.symbols('a b c')
expressionStr = "a+b**2"
expression = sympy.sympify(expressionStr)
print("String to sympy: ",type(expression), ":", expression)

# convert sympy expression to lambda expression
lambdaExpr = sympy.lambdify((x,y),expr, "numpy") # need to give it variables, expr, and optionally library with which to evaluate (I.E, use numpy.sqrt or math.sqrt)
outLambda = lambdaExpr(args[x],args[y])
print("lambdified expression: ",type(lambdaExpr), ": ", lambdaExpr, ": ", outLambda)

# print to latex expression
latexString = sympy.latex(expression, mode='plain')
print("latex string: ",latexString)

# unicode pretty print
prettyString = sympy.pretty(expression)
print("pretty expression: ", prettyString)

# =============================================================================
# simplification 
# https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html
# =============================================================================

# basic symplify usage - Note that what's most "simplified" is ambigious, so this general simplification can fail
expression2 = sympy.sin(x)**2 + sympy.cos(x)**2
expression2Simplified = sympy.simplify(expression2)
print("simplified expression: ",expression2Simplified)

# expand simplifier
expression3 = (x+y)**2
expression3Expanded = sympy.expand(expression3)
print("expanded expression: ", expression3Expanded)

# factor simplifier
expression3Factored = sympy.factor(expression3Expanded)
print("factored expression", expression3Factored)

# collect simplifier
expression4 = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
expression4Collected = sympy.collect(expression4, x) # need to tell it what you want collected (x)
print("collected expression: ", expression4Collected)

# cancel simplifier
expression5 = (x**2 + 2*x + 1)/(x**2 + x)
expression5Cancelled = sympy.cancel(expression5)
print("cancelled expression: ", expression5Cancelled)

# ... other simplifying methods in docs - not relevant here - includes trig simplification, power, exp, log simplification...


# =============================================================================
# calculus
# https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html
# =============================================================================

# single value differentiation
expression6 = sympy.cos(x)
expression6Derived = sympy.diff(expression6, x) # need to give it variable of differentiation
print("differentiated expression: ", expression6Derived)

# multivariable differentiation
expression7 = x**2*y**2
expression7Derived = sympy.diff(expression7, x, y) # need to give it variable of differentiation
print("differentiated expression: ", expression7Derived)

# indefinite integral
indefExpression6Integrated = sympy.integrate(expression6)
print("indefinite integral: ", indefExpression6Integrated) # note no constant of integration

# definite integral
defExpression6Integrated = sympy.integrate(expression6, (x, 0, sympy.pi/2)) # give tuple w/variable of integration and bounds
print("definite integral: ", defExpression6Integrated)

# can also integrate over multiple variables
# Note can also apply evalf() to integral for numerical integration

# limits
expression8 = sympy.sin(x) / x
x0 = 0
expression8Limit = sympy.limit(expression8, x, x0) # need to give variable and point approached - optional arg includes direction
print("limit: ",expression8Limit)

# series expansion
expression6Expanded = sympy.series(expression6, x, x0=0, n=6)
print("series expansion: ", expression6Expanded)

# docs go into finite differences and other topics not immediately relevant

# =============================================================================
# solvers
# https://docs.sympy.org/latest/tutorials/intro-tutorial/solvers.html
# =============================================================================

# basic equations
EqExpr1 = 2*x
Equation1 = sympy.Eq(y,EqExpr1) # equiv to y=EqExpr1
print("Basic equation: ", sympy.pretty(Equation1))

# algebraic solver
Equation1Sol = sympy.solve(Equation1, variable=None, domain=sympy.S.Complexes)
print("Algebraic solver solution: ",Equation1Sol, " (", type(Equation1Sol), ")")

# differential equation solver
f, g = sympy.symbols('f g', cls=sympy.Function) # undefined functions
diffeq = sympy.Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sympy.sin(x)) # note that function.diff(x) is an unevaluated derivative
print("diff eq:")
print(sympy.pretty(diffeq))
print()
print("diff sol:")

diffSol = sympy.dsolve(diffeq, f(x))
print(sympy.pretty(diffSol))

# =============================================================================
# Sympy can also do matrix maths, but not immediately relevant, see docs
# https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html
# =============================================================================
