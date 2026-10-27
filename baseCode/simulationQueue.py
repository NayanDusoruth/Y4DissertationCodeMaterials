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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# =============================================================================
# Utilities
# =============================================================================
# runs full simulation and saves it to directory
def fullSim(name, directory, args, saveSim=True, produceAnimatedPlot=True):
    # extract args from args dict - packaged like this to make handling "parameter sets" easier
    dt = args["dt"]
    tMax = args["tMax"]
    spectralFunction  = args["spectralFunction"]
    prefactor  = args["prefactor"]
    bandwidth  = args["bandwidth"]
    n  = args["n"]
    energyScale = args["energyScale"]
    beta = args["beta"]
    mu = args["mu"]
    initialSystemOccupation = args["initialSystemOccupation"]
    animateFps=args["animateFps"]
    animateDuration=args["animateDuration"]
    animateFormat=args["animateFormat"]
    
    # run simulation
    sim = simulation(name, spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu, initialSystemOccupation=initialSystemOccupation, animateFps=animateFps, animateDuration=animateDuration, animateFormat=animateFormat)
    sim.setupSim()
    sim.simulationCompute(dt, tMax)
    
    # save
    if(saveSim):
        sim.saveAll(directory, produceAnimatedPlot=produceAnimatedPlot)
    
    return sim

def fullDynamicMap(name, directory, args, saveSim):
    # extract args from args dict - packaged like this to make handling "parameter sets" easier
    dt = args["dt"]
    tMax = args["tMax"]
    spectralFunction  = args["spectralFunction"]
    prefactor  = args["prefactor"]
    bandwidth  = args["bandwidth"]
    n  = args["n"]
    energyScale = args["energyScale"]
    beta = args["beta"]
    mu = args["mu"]
    
    dualSim = dualSimulation(name, spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu)
    dualSim.simulateCompute(dt, tMax)
    dualSim.computeDynamicMap()
    
    if(saveSim):
        dualSim.saveAll(directory)
    
    return dualSim
def thermalizationWrapped(args):
    return analyticalThermalization(args["spectralFunction"], args["prefactor"], args["bandwidth"], args["energyScale"], args["beta"], args["mu"])
   
def computeThermalizedOccupancy(args):
    thermalization = thermalizationWrapped(args)
    return thermalization.computeOccupancy()

def generateParamsVariation(args, key, variations):
    argsList = []
    for variation in variations:
        newArgs = args.copy()
        newArgs[key] = variation
        argsList = argsList + [newArgs]
    return argsList

def simulateVariations(baseName, argsList, directory, saveSims = True, produceAnimatedPlot=False):
    if(saveSims):
        createFolder(directory, baseName)
        
    sims = []
    for i in range(0, len(argsList), 1):
        sim = fullSim(baseName+str(i), directory+"/"+baseName, argsList[i], saveSim=saveSims,produceAnimatedPlot=produceAnimatedPlot)
        sims = sims + [sim]
    return sims
       

def animateSystemVariationsHelper(frame, simulations=[], labels=[], fig=None, ax=None, figParams=figStandardParams, axLims=[0.55, 0.85]):
    ax.clear()
    fig = simulations[frame].plotSystemOccupation(fig=fig, ax=ax, legend=False, linestyle="-",plotThermalization=False, axLims=[0,1], densityLabel=labels[frame], thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=figStandardParams, plotColor="k", analyticalThermalizationColor="r", display=False)
    thermalizationPoint = simulations[frame].density.real[-1]
    ax.plot(simulations[frame].times, np.full(len(simulations[frame].times), thermalizationPoint), color="r", linewidth=figParams["markerWidth"], label=r"$\lim_{t\to\infty}(n_{occ}(t))$")
    ax.set_ylim(axLims)
    ax.legend(loc=figParams["legendLoc"],fontsize=figParams["legendFont"])
    return fig.get_axes()[0]

def animateSystemVariations(simulations, variedArg, variations, directory, name, figParams=figStandardParams,fps=25, vidFormat="gif"):
    # set up fig/ax
    fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
    ax = fig.add_subplot()
    pltTimesNewRoman()
    
    # frame stuff
    frames = len(simulations)
    
    # create labels
    labels = [str(variedArg) + ": "+'%0.3f' %variation for variation in variations]
    animatedPlot = animation.FuncAnimation(fig, functools.partial(animateSystemVariationsHelper,simulations=simulations, labels=labels, fig=fig, ax=ax, figParams=figParams), frames=frames, interval=(1/fps)/1000)
    
    gifWriter = animation.PillowWriter(fps=fps)
    FFwriter =  animation.FFMpegWriter(fps=fps)#animation.writers['ffmpeg']
    # create path and gif name
    path = directory + "/" + name
    
    # save
    if(vidFormat == "gif"):
        animatedPlot.save(path+".gif", gifWriter)
    elif(vidFormat == "mp4"):
        animatedPlot.save(path+".mp4", FFwriter)

# produces plot of three initial occupancies
def plotSystemMultipleOccupancy(args, occupancies, plotName, directory, occupancyNames=None, figParams=figStandardParams):
    argsList = generateParamsVariation(args, "initialSystemOccupation", occupancies)
    sims = simulateVariations("occupancy", argsList, directory, saveSims = False, produceAnimatedPlot=False)
    
    fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
    ax = fig.add_subplot()
    pltTimesNewRoman()
    
    thermalizationPlotted = False
    
    if(occupancyNames is None):
        occupancyNames = [str(x) for x in occupancies]
    
    colors = [plt.cm.coolwarm(x/len(sims)) for x in range(1, len(sims), 1)]
    
    for sim, occupancy, color in zip(sims, occupancyNames, colors):
        if(not thermalizationPlotted):
            fig = sim.plotSystemOccupation(fig=fig, ax=ax, legend=False, plotThermalization=True, axLims=[0,1], densityLabel=r"$n_{occ}(0)="+occupancy+"$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=figStandardParams, plotColor=color, analyticalThermalizationColor="r", display=False)
            thermalizationPlotted = True
        else:
            fig = sim.plotSystemOccupation(fig=fig, ax=ax, legend=False, plotThermalization=False, axLims=[0,1], densityLabel=r"$n_{occ}(0)="+occupancy+"$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=figStandardParams, plotColor=color, analyticalThermalizationColor="r", display=False)
    ax.legend(loc=figParams["legendLocMid"],fontsize=figParams["legendFont"])
    plt.show()
    saveFigure(directory, plotName, fig)
    
# =============================================================================
# standard parameters
# =============================================================================

dataDirectory = "/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/Data"


standardSpectralFunction = lambda omega, prefactor, bandwidth: prefactor * np.sqrt(1.-(omega/bandwidth)**2)

standardParams = {"dt": 1,
"tMax" : 500,
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
"animateFormat":"mp4"}

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

fullDynamicMap("standardDualSim", dataDirectory, standardParams, saveSim=True)


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