#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:22:34 2026

@author: nayandusoruth
"""
# =============================================================================
# library imports
# =============================================================================
from simulationUtility import *
from simulationHandlers import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import NayanGeneralUtils.plotting as plotting
import pandas as pd

    
# =============================================================================
# standard parameters
# =============================================================================

dataDirectory = "/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/Data"


standardSpectralFunction = lambda omega, prefactor, bandwidth: prefactor * np.sqrt(1.-(omega/bandwidth)**2)

standardParams = {"dt": 1,
"tMax" : 50,
"spectralFunction" : standardSpectralFunction,
"prefactor" : 0.005,
"n":200,
"bandwidth" : 1,
"energyScale" : 0,
"beta" : 1,
"mu" : 1,
"initialSystemOccupation":0,
"animateFps":25, 
"animateDuration":10, 
"animateFormat":"mp4",
"initialOccupations":np.linspace(0,1,10)
}

variationCount = 5


# =============================================================================
# Initial simulation variations
# =============================================================================

"""
# vary prefactor
prefactorArgs = generateParamsVariation(standardParams, "prefactor", np.linspace(0.005, 0.01, variationCount))
simulateVariations("prefactor", prefactorArgs, dataDirectory, produceAnimatedPlot=False)
    
# vary bandwidth
bandwidthArgs = generateParamsVariation(standardParams, "bandwidth", np.linspace(0.5, 1.5, variationCount))
simulateVariations("bandwidth", bandwidthArgs, dataDirectory, produceAnimatedPlot=False)
    
# vary n
nArgs = generateParamsVariation(standardParams, "n", np.arange(100, 500, 100))
simulateVariations("n", nArgs, dataDirectory, produceAnimatedPlot=False)

# vary energy scale
energyScaleArgs = generateParamsVariation(standardParams, "energyScale", np.linspace(0, 0.5, variationCount))
simulateVariations("energyScale", energyScaleArgs, dataDirectory, produceAnimatedPlot=False)
    
# vary beta
betaArgs = generateParamsVariation(standardParams, "beta", np.linspace(0, 2, variationCount))
simulateVariations("beta", betaArgs, dataDirectory, produceAnimatedPlot=False)

# vary mu
muArgs = generateParamsVariation(standardParams, "mu", np.linspace(0, 2, variationCount))
simulateVariations("mu", muArgs, dataDirectory, produceAnimatedPlot=False)
"""
    
    
# =============================================================================
# begining at thermalization
# =============================================================================
"""
analyticalSystemOccupancy = computeThermalizedOccupancy(standardParams)
thermalArgs = standardParams.copy()
thermalArgs["initialSystemOccupation"] = analyticalSystemOccupancy
fullSim("fromThermalizedoccupancy", dataDirectory, thermalArgs, produceAnimatedPlot=False)
"""


# =============================================================================
# animated thermalization plot
# =============================================================================
"""
fullSim("standardSim", dataDirectory, standardParams, produceAnimatedPlot=True)
"""

# =============================================================================
# plotting two evolutions - from 0, 1,
# =============================================================================
"""
#thermalization = thermalizationWrapped(standardParams)
#analyticalSystemOccupancy = thermalization.computeOccupancy()
plotSystemMultipleOccupancy(standardParams, [0, 1], "fromThree", dataDirectory, occupancyNames=["0", "1"])
"""

# =============================================================================
# standard dual sim plot
# =============================================================================
"""
fullDynamicMap("standardDualSim", dataDirectory, standardParams, saveSim=False)
"""

# =============================================================================
# TIny model for presentation illustrative purposes
# =============================================================================
"""
tinyArgs = standardParams.copy()
tinyArgs["n"] = 11
fullSim("tinyBath", dataDirectory, tinyArgs, produceAnimatedPlot=False)

"""
# =============================================================================
# animate
# =============================================================================
# vary prefactor
"""
testThermalizationDensities = fullSim("fromThermalizedoccupancy", dataDirectory, copyDict(standardParams, {"tMax":1000, "dt":500}), produceAnimatedPlot=False, saveSim=False).density.real
numericThermalization = testThermalizationDensities[-1]
animateVariationCount = 25*5
analyticalSystemOccupancy = computeThermalizedOccupancy(standardParams)
thermalArgs = copyDict(standardParams, {"tMax":50, "dt":0.5,"initialSystemOccupation":0.8, "bandwidth":2, "beta":10, "mu":0.1})
prefactorArgs = generateParamsVariation(thermalArgs, "prefactor", np.linspace(0.005, 0.2, animateVariationCount))
sims = simulateVariations("prefactor", prefactorArgs, dataDirectory, saveSims=False, produceAnimatedPlot=False)
animateSystemVariations(sims, "Coupling strength", np.linspace(0.005, 0.01, animateVariationCount), dataDirectory, "testMemoryAnimate", figParams=figStandardParams,fps=25, vidFormat="mp4")

"""


# =============================================================================
# testing non markovian "hump"
# =============================================================================
"""
humpVariations = copyDict(standardParams, {"tMax":50, "dt":0.1, "prefactor":0.2,"initialSystemOccupation":0.8, "bandwidth":2, "beta":7, "mu":0.1, "n":300, "initialOccupations":np.linspace(0.4, 0.9, 20)})
dualSimHumps = fullDynamicMap("humpDualSimulation", dataDirectory, humpVariations, saveSim=True)
print("dial sim numeric thermalization: ", dualSimHumps.numericSystemOccupancy)
"""
# =============================================================================
# varying max deviation over mu
# =============================================================================
"""
humpVariations = copyDict(standardParams, {"tMax":50, "dt":0.1, "prefactor":0.2,"initialSystemOccupation":0.8, "bandwidth":2, "beta":10, "mu":0.1, "n":300, "initialOccupations":np.linspace(0.4, 0.9, 20)})
argsVaried = np.linspace(-0.5, 0.5, 20)
ArgsVar = generateParamsVariation(humpVariations, "mu", argsVaried)
dualSims = simulateDynamicsVariations("maxDeviationInitialExamination", ArgsVar, dataDirectory, saveSims=False)

maxDevs = [dualSim.maximumDeviationFromThermalization()[0] for dualSim in dualSims]
plot = plotting.plotter(xLabel=r"$\mu$", yLabel=r"$\delta p$")
plot.scatter(argsVaried, maxDevs)
"""

# =============================================================================
# saving/loading utility tests
# =============================================================================
"""
fullSim("standardSim", dataDirectory, standardParams, produceAnimatedPlot=False, saveSim=True)
sim = loadFullSim("standardSim", dataDirectory)
print("saved/loaded sim data: ", sim.spectralPrefactor, " ", len(sim.correlations))

fullDynamicMap("standardDualSim", dataDirectory, copyDict(standardParams,{"initialOccupations":[0.1, 0.2]}), saveSim=True)
dualSim = loadFullDynamicMap("standardDualSim", dataDirectory)
print("saved/loaded dualSim: ", dualSim.initialOccupations, " ", len(dualSim.dynamicalMaps))
"""