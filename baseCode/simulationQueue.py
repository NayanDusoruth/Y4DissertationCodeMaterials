#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:22:34 2026

@author: nayandusoruth
"""
# =============================================================================
# library imports
# =============================================================================
from simulationUtility import simulation, createFolder
import numpy as np

# =============================================================================
# Utilities
# =============================================================================
def fullSim(name, directory, dt, tMax, spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu):
    sim = simulation(name, spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu)
    sim.setupSim()
    sim.simulationCompute(dt, tMax)
    sim.saveAll(directory)
    

# =============================================================================
# Queue
# =============================================================================

dataDirectory = "/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/Data"
standardDt = 0.1
standardTmax = 1000
standardSpectralFunction = lambda omega, prefactor, bandwidth: prefactor * np.sqrt(1.-(omega/bandwidth)**2)
standardPrefactor = 0.005
standardN=300
standardBandwidth = 1
standardEnergyScale = 0
standardBeta = 1
standardMu = 1

variationCount = 5

# vary prefactor
"""prefactorSubDir = "prefactor"
createFolder(dataDirectory, prefactorSubDir)
i = 0
for prefactor in np.linspace(0.005, 0.01, variationCount):
    fullSim("prefactor_"+str(i), dataDirectory+"/"+prefactorSubDir, standardDt, standardTmax, standardSpectralFunction, prefactor, standardBandwidth, standardN, standardEnergyScale, standardBeta, standardMu)
    i = i+1
    
# vary bandwidth
bandwidthSubDir = "bandwidth"
createFolder(dataDirectory, bandwidthSubDir)
i = 0
for bandwidth in np.linspace(0.5, 1.5, variationCount):
    fullSim("bandwidth_"+str(i), dataDirectory+"/"+bandwidthSubDir, standardDt, standardTmax, standardSpectralFunction, standardPrefactor, bandwidth, standardN, standardEnergyScale, standardBeta, standardMu)
    i = i+1
    
# vary n
nSubDir = "n"
createFolder(dataDirectory, nSubDir)
i = 0
for n in np.arange(100, 500, 100):
    fullSim("n_"+str(i), dataDirectory+"/"+nSubDir, standardDt, standardTmax, standardSpectralFunction, standardPrefactor, standardBandwidth, n, standardEnergyScale, standardBeta, standardMu)
    i = i+1"""
#fullSim("n_4", dataDirectory+"/n", standardDt, standardTmax, standardSpectralFunction, standardPrefactor, standardBandwidth, 500, standardEnergyScale, standardBeta, standardMu)
# vary energyScale

"""
energyScaleSubDir = "energyScaleSubDir"
createFolder(dataDirectory, energyScaleSubDir)
i = 0
for energyScale in np.linspace(0, 0.5, variationCount):
    fullSim("energyScale_"+str(i), dataDirectory+"/"+energyScaleSubDir, standardDt, standardTmax, standardSpectralFunction, standardPrefactor, standardBandwidth, standardN, energyScale, standardBeta, standardMu)
    i = i+1
    
# vary beta
betaSubDir = "beta"
createFolder(dataDirectory, betaSubDir)
i = 0
for beta in np.linspace(0, 2, variationCount):
    fullSim("beta_"+str(i), dataDirectory+"/"+betaSubDir, standardDt, standardTmax, standardSpectralFunction, standardPrefactor, standardBandwidth, standardN, standardEnergyScale, beta, standardMu)
    i = i+1"""
#fullSim("beta_6", dataDirectory+"/beta", standardDt, standardTmax, standardSpectralFunction, standardPrefactor, standardBandwidth, standardN, standardEnergyScale, 10, standardMu)
"""# vary mu
muSubDir = "mu"
createFolder(dataDirectory, muSubDir)
i = 0
for mu in np.linspace(0, 2, variationCount):
    fullSim("mu_"+str(i), dataDirectory+"/"+muSubDir, standardDt, standardTmax, standardSpectralFunction, standardPrefactor, standardBandwidth, standardN, standardEnergyScale, standardBeta, mu)
    i = i+1"""