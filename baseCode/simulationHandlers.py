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
import NayanGeneralUtils.plotting as plotting
import pandas as pd
# =============================================================================
# Simulation utilities
# =============================================================================
# runs full simulation and saves it to directory - </function verified/>
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

# loads simulation object from "saveAll" folder directory/name  - </function verified/>
def loadFullSim(name, directory):
    sim = simulation(name, None, None, None, None, None, None, None)
    sim.load(directory + "/" + name, name)
    return sim

# runs full dynamic map and saves it to directory - </function verified/>
def fullDynamicMap(name, directory, args, saveSim=True):
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
    initialOccs = args["initialOccupations"]
    
    dualSim = dualSimulation(name, spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu, initialOccupations=initialOccs)
    dualSim.simulateCompute(dt, tMax)
    dualSim.computeDynamicMap()
    
    if(saveSim):
        dualSim.saveAll(directory)
    
    return dualSim

# loads dynamic map from "saveAll" folder directory/name - </function verified/>
def loadFullDynamicMap(name, directory):
    dualSim = dualSimulation(name, None, None, None, None, None, None, None)
    dualSim.load(directory, name)
    return dualSim

# =============================================================================
# Data simulation utilities
# =============================================================================

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

def simulateDynamicsVariations(baseName, argsList, directory, saveSims = True):
    if(saveSims):
        createFolder(directory, baseName)
        
    sims = []
    for i in range(0, len(argsList), 1):
        sim = fullDynamicMap(baseName+str(i), directory+"/"+baseName, argsList[i], saveSim=saveSims)
        sims = sims + [sim]
    return sims

# =============================================================================
# dataset class
# =============================================================================

"""
dt = args["dt"]
tMax = args["tMax"]
spectralFunction  = args["spectralFunction"]
prefactor  = args["prefactor"]
bandwidth  = args["bandwidth"]
n  = args["n"]
energyScale = args["energyScale"]
beta = args["beta"]
mu = args["mu"]
initialOccs = args["initialOccupations"]"""

class dualDataset():
    # ----------------------------------------------
    # Constructor methods
    # ----------------------------------------------
    def __init__(self, name, variations, loadData=None, fromLoad=False):
        self.variations = variations
        if(not fromLoad):
            argsList = dualDataset.constructParameterSpace(variations)
            # TODO: construct dataframe, add argsList as 1 column, unpack args in argslist across rows for each parameter variable
            
            #self.parameterData = 
        else:
            self.parameterData = loadData
    
    
    def constructParameterSpace(variations, args={}): 
        """
        Parameters
        ----------
        variations : Dictionary of form {key:[keyVariations]}
            list of parameters to vary, list of keys: ["dt", "tMax", "spectralFunction", "prefactor", "bandwidth", "n", "energyScale", "beta", "mu", "initialOccupations"].

        Returns
        -------
        list of dictionaries - each dictionary represents a parameter coordinate

        """
        keys = list(variations.keys())
        values = list(variations.values())
        
        # base case
        if(len(keys)==1):
           return generateParamsVariation(args, keys[0], values[0])
       
        # inductive case
        else:
            argsList = []
            currentParam = keys[0]
            currentVars = values[0]
            reducedVariations = dict(zip(keys[1:], values[1:]))
            
            # explore variations of current parameter and append to list
            for variation in currentVars:
                newArgs = copyDict(args, {currentParam:variation})
                argsList = argsList + [dualDataset.constructParameterSpace(reducedVariations, args=newArgs)]
                
            return argsList
            
    
    # ----------------------------------------------
    # Saving methods
    # ----------------------------------------------
    def save(self, directory):
        # create folder
        
        # save infoDict w/variations dictionary stored
        
        # save parameter space as csv - instead of dualSim column, have list of relative directory names
        
        # save dualSim folders according to directory names
        pass
    
    def load(name, directory):
        # load infoDict
        
        # read parameter space csv
        
        # read dualSims list
        
        # assign to dualDataset object
        
        # return
        
        pass

    # ----------------------------------------------
    # Simulation methods
    # ----------------------------------------------
    
    # ----------------------------------------------
    # Accessor methods
    # ----------------------------------------------

# =============================================================================
# Analytical utilities
# =============================================================================

def thermalizationWrapped(args):
    return analyticalThermalization(args["spectralFunction"], args["prefactor"], args["bandwidth"], args["energyScale"], args["beta"], args["mu"])
   
def computeThermalizedOccupancy(args):
    thermalization = thermalizationWrapped(args)
    return thermalization.computeOccupancy()



# =============================================================================
# plotting utilities
# =============================================================================
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
# testing
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

