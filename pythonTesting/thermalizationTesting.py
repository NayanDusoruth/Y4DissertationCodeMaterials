#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:54:24 2026

@author: nayandusoruth
"""

# =============================================================================
# library imports
# =============================================================================
import sympy
import numpy as np
from NayanGeneralUtils.plotting import plotter
import time 
from scipy.integrate import quad

# =============================================================================
# scipy testing
# =============================================================================
# ----------------------
# setup
# ----------------------

prefactor = 0.005
bandwidth = 1
systemEnergy = 0
mu = 1
beta = 1
spectralFunction = lambda omega: prefactor * np.sqrt(1.-(omega/bandwidth)**2)

fermiFunction = lambda x:1/(1+np.exp(beta*(x-mu)))

# ----------------------
# \Lambda(\omega) integral 
# ----------------------

def computeLambda(spectralFunction, omegaVal, bandwidth, eta=0.0001):
    integrand = lambda omega: spectralFunction(omega) / (omegaVal - omega)
    integralBelow = quad(integrand, -bandwidth, omegaVal-eta, points=[-bandwidth, omegaVal],limit=1000, complex_func=False)
    integralAbove = quad(integrand, omegaVal+eta, bandwidth, points=[bandwidth, omegaVal],limit=1000, complex_func=False)
    integral = integralBelow[0] + integralAbove[0]
    return integral

computeLambda(spectralFunction, 0.5, bandwidth)

# ----------------------
# \A(\omega) function 
# ----------------------

def computeA(spectralFunction, omegaVal, systemEnergy, bandwidth):
    spectralValue = spectralFunction(omegaVal)
    lambdaVal = computeLambda(spectralFunction, omegaVal, bandwidth)
    value = (2*np.pi*spectralValue) / ((omegaVal - systemEnergy - lambdaVal)**2 + (np.pi)**2*spectralValue**2)
    return value

# ----------------------
# compute occupancy
# ----------------------

def computeOccupancy(spectralFunction, systemEnergy, bandwidth):
    integrand = computeA
# ----------------------
# plotting A(\omega)
# ----------------------
nPoints = 100
fudge = 0.1
xVals = np.linspace(-bandwidth+fudge, bandwidth-fudge, nPoints)
yVals = []
for x in xVals:
    print("evaluation for ", x)
    y = computeA(spectralFunction, x, systemEnergy, bandwidth)
    print("evaluation complete: ", y)
    yVals = yVals + [y]
#yVals = [Aexpression.evalf(subs={omega:x}) for x in xVals]
print("check8")
plotter = plotter()
plotter.scatter(xVals, yVals)
plotter.display()
plotter.saveFig("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/pythonTesting/thermalizationPlots", "nPoints_"+str(nPoints))



# =============================================================================
# sympy testing
# =============================================================================
"""
# ----------------------
# setup
# ----------------------

prefactor = 0.005
bandwidth = 1
systemEnergy = 0

omega = sympy.Symbol(r"\omega",real=True)


# ----------------------
# spectral function
# ----------------------
spectralExpression = prefactor * sympy.sqrt(1 - omega/bandwidth)

print(sympy.pretty(spectralExpression))


# ----------------------
# \Lambda(\omega) integral 
# ----------------------
 # done like this cause delimiters

print(sympy.integrate(omega, (omega, -bandwidth, 0.5)))
#omegaVal = 1

#print(sympy.pretty(integrandFunction))

#etaVal = 0.001

def computeLambda(spectralExpression, omegaVal, bandwidth):
    print("omegaVal",omegaVal)
    omegaDash = sympy.Symbol(r'\omega^{dash}', real=True)
    eta = sympy.Symbol(r'\eta')
    spectralExpressionSubstituted = spectralExpression.subs(omega, omegaDash)
    integrandFunction = spectralExpressionSubstituted / (omegaVal - omegaDash)
    print(sympy.pretty(integrandFunction))
    
    indefLambdaExpr = sympy.integrate(integrandFunction)
    #indefLambdaExprUpper = sympy.limit(indefLambdaExpr,omegaDash, omegaVal, dir="-")
    #print(sympy.pretty(indefLambdaExpr))
    #print("->", sympy.pretty(indefLambdaExprUpper))
    defInt = sympy.integrate(integrandFunction, (omegaDash, -bandwidth, omegaVal-0.001))
    print(sympy.pretty(defInt))
    
    lambdaExpressionLower = sympy.limit(sympy.integrate(integrandFunction, (omegaDash, -bandwidth, omegaVal-eta)), eta,0,dir="+")
    return lambdaExpressionLower
    #lambdaExpressionUpper = sympy.limit(sympy.integrate(integrandFunction, (omegaDash, omegaVal+eta, bandwidth)), eta,0,dir="+")
    #lambdaExpression = sympy.simplify(lambdaExpressionLower + lambdaExpressionUpper)
    #return lambdaExpression

testCase = computeLambda(spectralExpression, sympy.Rational(1,2), bandwidth)
#print(testCase)

"""
"""
print("test: ", sympy.latex(lambdaExpressionLowerSubbed))
print("check0")
lambdaExpressionLower = sympy.integrate(integrandFunction, (omegaDash, -bandwidth, omega-eta))
print("check1")
lambdaExpressionUpper = sympy.integrate(spectralExpressionSubstituted / (omega - omegaDash), (omegaDash, omega+eta, bandwidth))
print("check2")
lambdaExpression = sympy.simplify(lambdaExpressionLower + lambdaExpressionUpper)
print("check3")
print(sympy.latex(lambdaExpression))

lambdaExpression = lambdaExpression.subs(eta, etaVal)
print("check4")




# ----------------------
# A(\omega) function
# ----------------------

Aexpression = (2 * sympy.pi * spectralExpression) / ((omega - systemEnergy - lambdaExpression)**2+(sympy.pi)**2*spectralExpression**2)
print("check5")
print(sympy.latex(Aexpression))
print("check6")
startTime = time.time()
testEval = Aexpression.evalf(subs={omega:0.5})
print("test eval: ", type(testEval), ": ", testEval)
endTime = time.time()

print("check7 - time elapsed since check6: ", endTime-startTime)
# ----------------------
# plotting A(\omega)
# ----------------------
nPoints = 10
fudge = 0.1
xVals = np.linspace(-bandwidth+fudge, bandwidth-fudge, nPoints)
yVals = []
for x in xVals:
    print("evaluation for ", x)
    y = Aexpression.evalf(subs={omega:x})
    print("evaluation complete: ", y)
    yVals = yVals + [y]
#yVals = [Aexpression.evalf(subs={omega:x}) for x in xVals]
print("check8")
plotter = plotter()
plotter.scatter(xVals, yVals)
plotter.display()
plotter.saveFig("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/pythonTesting/thermalizationPlots", "nPoints_"+str(nPoints))
"""