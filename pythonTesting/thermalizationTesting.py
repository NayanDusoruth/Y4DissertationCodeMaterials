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
import scipy
# =============================================================================
# scipy testing
# =============================================================================
# ----------------------
# setup
# ----------------------
"""
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

def computeLambda(omegaVal, spectralFunction, bandwidth, eta=0.0001):
    integrand = lambda omega: spectralFunction(omega) / (omegaVal - omega)
    integralBelow = quad(integrand, -bandwidth, omegaVal-eta, points=[-bandwidth, omegaVal],limit=1000, complex_func=False)
    integralAbove = quad(integrand, omegaVal+eta, bandwidth, points=[bandwidth, omegaVal],limit=1000, complex_func=False)
    integral = integralBelow[0] + integralAbove[0]
    return integral

#computeLambda(0.5, spectralFunction, bandwidth)

# ----------------------
# \A(\omega) function 
# ----------------------

def computeA(omegaVal, spectralFunction, systemEnergy, bandwidth):
    spectralValue = spectralFunction(omegaVal)
    lambdaVal = computeLambda(omegaVal, spectralFunction, bandwidth)
    value = (2*np.pi*spectralValue) / ((omegaVal - systemEnergy - lambdaVal)**2 + (np.pi)**2*spectralValue**2)
    return value

# ----------------------
# compute occupancy
# ----------------------

def computeOccupancy(spectralFunction, systemEnergy, bandwidth, eta=0.0001):
    integrand = lambda omega: computeA(omega, spectralFunction, systemEnergy, bandwidth) * fermiFunction(omega)
    integral = quad(integrand, -bandwidth+eta, bandwidth-eta)
    return integral / (2*np.pi)
# ----------------------
# plotting A(\omega)
# ----------------------
nPoints = 100
fudge = 0.1
xVals = np.linspace(-bandwidth+fudge, bandwidth-fudge, nPoints)
yVals = []
for x in xVals:
    print("evaluation for ", x)
    y = computeA(x, spectralFunction, systemEnergy, bandwidth)
    print("evaluation complete: ", y)
    yVals = yVals + [y]
#yVals = [Aexpression.evalf(subs={omega:x}) for x in xVals]
print("check8")
plotter = plotter()
plotter.scatter(xVals, yVals)
plotter.display()
plotter.saveFig("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/pythonTesting/thermalizationPlots", "nPoints_"+str(nPoints))

print("check9")
print("Occupancy:",computeOccupancy(spectralFunction, systemEnergy, bandwidth))
"""


class analyticalThermalization():
    
    def __init__(self, spectralFunction, spectralPrefactor, bandwidth, systemEnergy, beta, mu):
        self.spectralPrefactor = spectralPrefactor
        self.bandwidth = bandwidth
        self.systemEnergy = systemEnergy
        self.beta = beta
        self.mu = mu
        self.eta = 0.0001
        self.spectralFunction = spectralFunction
        

    def computeLambda(self, omegaVal):
        integrand = lambda omega: self.spectralFunction(omega, self.spectralPrefactor, self.bandwidth) / (omegaVal - omega)
        integralBelow = scipy.integrate.quad(integrand, -bandwidth, omegaVal-self.eta, points=[-bandwidth, omegaVal],limit=1000, complex_func=False)
        integralAbove = scipy.integrate.quad(integrand, omegaVal+self.eta, bandwidth, points=[bandwidth, omegaVal],limit=1000, complex_func=False)
        integral = integralBelow[0] + integralAbove[0]
        return integral
    
    def computeA(self, omegaVal):
        spectralValue = self.spectralFunction(omegaVal, self.spectralPrefactor, self.bandwidth)
        lambdaVal = self.computeLambda(omegaVal)
        value = (2*np.pi*spectralValue) / ((omegaVal - self.systemEnergy - lambdaVal)**2 + (np.pi)**2*spectralValue**2)
        return value
    
    def occupancyIntegrand(self, omega):
        return self.computeA(omega) / (1 + np.exp(self.beta*(omega-self.mu)))
    
    def computeOccupancy(self):
        return scipy.integrate.quad(self.occupancyIntegrand, -self.bandwidth+self.eta, self.bandwidth-self.eta)[0] / (2*np.pi)
    
    def plotA(self,):
        pass
spectralFunction = lambda omega, prefactor, bandwidth: prefactor * np.sqrt(1.-(omega/bandwidth)**2)
prefactor = 0.005
n=300
bandwidth = 0.2
energyScale = 0
beta = 1
mu = 0    
    
thermalization = analyticalThermalization(spectralFunction, prefactor, bandwidth, energyScale, beta, mu)
#thermalization.plotA()
print("occupancy: ", thermalization.computeOccupancy())
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