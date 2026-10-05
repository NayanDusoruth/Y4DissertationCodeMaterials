#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:54:59 2026

@author: nayandusoruth
"""
# =============================================================================
# Libraries and initial setup
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
from scipy.linalg import eigh
import os
import json
import torch
import inspect
import pickle
# check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU acceleration.")
else:
    device = torch.device("cpu")
    print("MPS device not found. Using CPU.")

# =============================================================================
# Code utilities
# =============================================================================

# matplot lib utility function - sets font to Times New Roman
def pltTimesNewRoman():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.it'] = 'serif:italic'
    plt.rcParams['mathtext.bf'] = 'serif:bold'
    plt.rcParams['mathtext.fontset'] = "custom"

# dictionary for standard fig values
figStandardParams = {"markerWidth":0.6, 
                     "markerArea":12, 
                     "gridColor":'k', 
                     "fontsize":8, 
                     "gridStyle":':',
                     "gridWidth":0.6, 
                     "errCapsize":2, 
                     "histWidth":0.5, 
                     "width":8.75, 
                     "heightOverWidth":3/4, 
                     "dpi":600,
                     "nFuncPoints":1000,
                     "labelpad":-0.4}


def createFolder(directory, folderName):
    """File handling function - creates new folder at 'directory/folderName'"""
    newPath = str(directory + "/"+folderName)
    if (not os.path.exists(newPath)):
        os.makedirs(newPath)
    return newPath

# utility function - saves python dictionary as a json file  - </verified/>
def saveDictAsJson(directory, fileName, dictionary, indent=3):
    """File handling function - saves dictionary to directory with filename as Json with indent=indent"""
    
    # go to desired directory
    curDir = os.getcwd() # get the current directory
    os.chdir(directory)
    
    # save file as json
    with open(fileName+".json", "w") as file:
        json.dump(dictionary, file, indent=indent)
        
    # go back to original directory
    os.chdir(curDir)

# utility function - loads python dictionary from json file - </verified/>
def loadDictFromJson(directory, fileName):
    """File handling function - loads dictionary from directory/fileName.json"""
    # go to desired directory
    curDir = os.getcwd() # get the current directory
    os.chdir(directory)
    
    # access json file
    with open(directory + "/" + fileName + '.json') as json_file:
        dictionary = json.load(json_file) 
        
    # go back to original directory
    os.chdir(curDir)
    
    return dictionary

# utility function - saves fig to directory with filename - </verified/>
def saveFigure(directory, fileName, fig):
    """File handling function - saves fig to directory with filename"""
    # go to desired directory
    curDir = os.getcwd() # get the current directory
    os.chdir(directory)

    # save figure to folder
    fig.savefig(fileName, dpi=300, bbox_inches='tight')
    
    # go back to original directory
    os.chdir(curDir)

# converts lambda function to string 
def lambdaAsString(lambdaFunc):
    #print("func: ",lambdaFunc)
    #print(type(lambdaFunc))
    #print(lambdaFunc(1,1,1))
    funcString = str(inspect.getsourcelines(lambdaFunc)[0])
    funcString = funcString.strip("['\\n']").split(" = ")[1]
    print(funcString)
    return funcString


# =============================================================================
# Analytical thermalization object
# Contains the maths for the analytical thermalized system occupation
# =============================================================================
class analyticalThermalization():
    
    def __init__(self, spectralFunction, spectralPrefactor, bandwidth, systemEnergy, beta, mu, eta=0.0001):
        self.spectralPrefactor = spectralPrefactor
        self.bandwidth = bandwidth
        self.systemEnergy = systemEnergy
        self.beta = beta
        self.mu = mu
        self.eta = eta
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
        return float(scipy.integrate.quad(self.occupancyIntegrand, -self.bandwidth+self.eta, self.bandwidth-self.eta)[0] / (2*np.pi))
    
    def plotA(self,figParams=figStandardParams, display=True):
        # fig setup
        inch = 2.54
        fig = plt.figure(figsize=(figParams["width"]/inch, figParams["heightOverWidth"] * figParams["width"]/inch), dpi=figParams["dpi"])
        ax = fig.add_subplot()
        pltTimesNewRoman()
        
                
        ax.set_xlabel("$\omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("$A(\omega)$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        # produce data for function plot
        xData = np.linspace(-self.bandwidth,self.bandwidth,figParams["nFuncPoints"])
        yData = [self.computeA(x) for x in xData]
        
        # plot function data
        ax.plot(xData, yData, linewidth=figParams["markerWidth"])

        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
        
        


# =============================================================================
# simulation class object
# =============================================================================
class simulation():
    
    # ----------------------------------------------
    # Constructor methods
    # ----------------------------------------------
    def __init__(self, name, spectralFunction, spectralPrefactor, bandwidth, nBathSites, epsilon, beta, mu, initialSystemOccupation=0,usePytorch=True, saveCorrelations=False):
        # simulation configs
        self.name = name
        self.usePytorch = usePytorch
        self.saveCorrelations = saveCorrelations
        
        # assign initial variables
        self.initialSystemOccupation = initialSystemOccupation
        self.spectralFunction = spectralFunction # assume (omega, prefactor, bandwidth) args
        self.spectralFunctionStr = lambdaAsString(spectralFunction)
        self.spectralPrefactor = spectralPrefactor
        self.bandwidth = bandwidth
        self.nBathSites = nBathSites
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.fermi = lambda x:1/(1+np.exp(self.beta*(x-self.mu)))
        
        # intialise density array as None - there are reasons for this
        self.density = None
        
        # setup analytical thermalization object
        self.analyticalThermalization = analyticalThermalization(spectralFunction,  spectralPrefactor, bandwidth, epsilon, beta, mu)
        self.analyticalSystemOccupancy = self.analyticalThermalization.computeOccupancy()
        print("occupancy: ", self.analyticalSystemOccupancy, " ", type(self.analyticalSystemOccupancy))
    
    # ----------------------------------------------
    # Saving and loading methods
    # ----------------------------------------------
    # Saves data to a pickle object w/extension .simData
    def save(self, directory):
        
        if(self.saveCorrelations):
            savingDict ={"name":self.name,
                         "saveCorrelations":True,
                         "spectralFunction":self.spectralFunctionStr,
                         "spectralPrefactor":self.spectralPrefactor,
                         "bandwidth":self.bandwidth,
                         "nBathSites":self.nBathSites,
                         "epsilon":self.epsilon,
                         "beta":self.beta,
                         "mu":self.mu,
                         "timeSteps":len(list(self.correlations.keys())),
                         "tMax":list(self.correlations.keys())[-1],
                         "correlations":self.correlations
                         }
            
            pickle.dump(savingDict, open(directory+'/'+self.name+'.simData', 'wb'))
        else:
            """print(list(self.correlations.values())[0])
            print("----")
            print(list(self.correlations.values())[-1].numpy)
            print("-----")"""
            savingDict ={"name":self.name,
                         "saveCorrelations":False,
                         "spectralFunction":self.spectralFunctionStr,
                         "spectralPrefactor":self.spectralPrefactor,
                         "bandwidth":self.bandwidth,
                         "nBathSites":self.nBathSites,
                         "epsilon":self.epsilon,
                         "beta":self.beta,
                         "mu":self.mu,
                         "timeSteps":len(list(self.correlations.keys())),
                         "tMax":list(self.correlations.keys())[-1],
                         "times":self.times,
                         "density":self.density,
                         "bathDensity":self.bathDensity,
                         "lastCorrelation":list(self.correlations.values())[-1].numpy()
                         }
            
            pickle.dump(savingDict, open(directory+'/'+self.name+'.simData', 'wb'))
    
    # Loads data from a pickle object w/extension .simData - BE CAREFUL opening files - the lambda function needs to be evaluated which presents a security risk, only open safe files
    def load(self, directory, fileName):
        loadData = pickle.load(open(directory+'/'+fileName+'.simData', 'rb'))
        
        self.name = loadData["name"]
        self.spectralFunctionStr = loadData["spectralFunction"]
        self.spectralFunction = eval(loadData["spectralFunction"])
        self.spectralPrefactor=loadData["spectralPrefactor"]
        self.bandwidth = loadData["bandwidth"]
        self.nBathSites = loadData["nBathSites"]
        self.epsilon = loadData["epsilon"]
        self.beta = loadData["beta"]
        self.mu = loadData["mu"]
        
        self.evaluateSpectralFunction()
        self.initaliseHamiltonian()
        
        self.analyticalThermalization = analyticalThermalization(self.pectralFunction, self.spectralPrefactor, self.bandwidth, self.epsilon, self.beta, self.mu)
        self.analyticalSystemOccupancy = self.analyticalThermalization.computeOccupancy()
        
        if(loadData["saveCorrelations"]):
            self.correlations = loadData["correlations"]
        else:
            self.times = loadData["times"]
            self.density = loadData["density"]
            self.bathDensity = loadData["bathDensity"]
            
            self.computeInitialCorrelation()
            for i in range(1, len(self.times)-1,1):
                time = self.times[i]
                self.correlations[time] = None
                
            self.correlations[self.times[-1]] = loadData["lastCorrelation"]
            print("len:",len(self.correlations))
        
    # creates a folder and saves .simData and all plots... for human readability
    def saveAll(self, directory):
        createFolder(directory, self.name)
        workingDirectory = directory + "/" + self.name
        
        # save simulation data
        self.save(workingDirectory)
        
        # save general info to a json file
        #print(lambdaAsString(self.spectralFunction))
        print()
        infoDict ={"name":self.name,
                     "saveCorrelations":True,
                     "spectralFunction":self.spectralFunctionStr,
                     "spectralPrefactor":self.spectralPrefactor,
                     "bandwidth":self.bandwidth,
                     "nBathSites":int(self.nBathSites),
                     "epsilon":self.epsilon,
                     "beta":self.beta,
                     "mu":self.mu,
                     "timeSteps":len(list(self.correlations.keys())),
                     "tMax":float(list(self.correlations.keys())[-1]),
                     "analyticalSystemOccupation":self.analyticalSystemOccupancy
                     }
        saveDictAsJson(workingDirectory, self.name+"_simulationInfo", infoDict)
        
        # save plots
        saveFigure(workingDirectory, self.name+"_spectralFunction",self.plotSpectralFunction(display=False))
        saveFigure(workingDirectory, self.name+"_InitialCorrelationMatrix",self.plotCorrelationMatrix(0,display=False))
        saveFigure(workingDirectory, self.name+"_systemOccupation",self.plotSystemOccupation(display=False))
        saveFigure(workingDirectory, self.name+"_bathOccupation",self.plotBathOccupation(display=False))
        saveFigure(workingDirectory, self.name+"_AomegaPlot",self.analyticalThermalization.plotA(display=False))
        
        
    # ----------------------------------------------
    # simulation methods
    # ----------------------------------------------
    # Evaluate the spectral function over entire bandwidth and save result - </method verified/>
    def evaluateSpectralFunction(self):
        self.omegas = np.linspace(-self.bandwidth, self.bandwidth, self.nBathSites)
        self.spectralValues = self.spectralFunction(self.omegas, self.spectralPrefactor, self.bandwidth)
    
    # Initialize hamiltonian matrix given spectral function values - </method verified/>
    def initaliseHamiltonian(self):
        # code copied from provided code
        self.hamiltonian = np.zeros((self.nBathSites+1,self.nBathSites+1),dtype=np.complex128)
        # first column
        self.hamiltonian[1:,0] = np.sqrt(self.spectralValues)/np.sqrt(self.nBathSites)
        # first row
        self.hamiltonian[0,1:] = np.sqrt(self.spectralValues)/np.sqrt(self.nBathSites)
        # diagonal
        np.fill_diagonal(self.hamiltonian, [self.epsilon]+list(self.omegas))
        
        
    # Compute initial correlation matrix - </method verified/>
    def computeInitialCorrelation(self):
        # code copied from provided code
        CorrelationSiteBasis_0 = np.zeros_like(self.hamiltonian)
        np.fill_diagonal(CorrelationSiteBasis_0,[self.initialSystemOccupation]+[self.fermi(o) for o in self.omegas])
        
        self.correlations = {0:CorrelationSiteBasis_0}
    
    # 
    def setupSim(self):
        self.evaluateSpectralFunction()
        self.initaliseHamiltonian()
        self.computeInitialCorrelation()
    
    # Compute and return unitary function
    def unitary(self, t):
        return np.matrix(scipy.linalg.expm(-1j*self.hamiltonian*t))
    
    # Main simulation method 1 - Computes correlation matrix states using iterative implementation (Computationally intensive step)
    def iterativeCompute(self, dt, tMax):
        lastTime = list(self.correlations.keys())[-1] # the last time in the data - allows for simulation to be appended
        # code copied from provided code
        times = np.arange(lastTime,lastTime+tMax, dt)
        Udt = self.unitary(dt)
        UdtT = Udt.H
        key = lastTime
        for t in times[1:]:
            self.correlations[t] = Udt@self.correlations[key]@UdtT
            key = t
    
    # Main simulation method 1 - Computes correlation matrix states using pytorch implementation (Computationally intensive step)
    def pytorchCompute(self, dt, tMax):
        lastTime = list(self.correlations.keys())[-1] # the last time in the data - allows for simulation to be appended
        
        # code copied from provided code
        times = np.arange(lastTime,lastTime+tMax, dt)
        Udt = self.unitary(dt)
        UdtT = torch.from_numpy(Udt.H)
        Udt = torch.from_numpy(self.unitary(dt))
        key = lastTime
        for t in times[1:]:
            self.correlations[t] = Udt@self.correlations[key]@UdtT
            key = t
    
    def simulationCompute(self, dt, tMax):
        if(self.usePytorch):
            self.pytorchCompute(dt, tMax)
        else:
            self.iterativeCompute(dt, tMax)
        
        self.computeDensity()
    
    # Compute the system density operators for the entire simulation
    def computeDensity(self):
        # code copied from provided code
        self.times = list(self.correlations.keys())
        
        if(self.density is None): # if there is no density list yet (so nothing to append to)
            self.density = np.array([ val[0,0] for key, val in self.correlations.items()])
            self.bathDensity = np.array([ np.diagonal(val[1:,1:]) for key, val in self.correlations.items()])
        else: # if the densities already exist, append new densities rather than overwriting them
            correlations = list(self.correlations.items())
            for i in range(len(self.density),len(self.times),1):
                key = correlations[i][0]
                val = correlations[i][1]
                self.density = np.append(self.density,np.array(val[0,0]))
                self.bathDensity = np.vstack([self.bathDensity,np.diagonal(val[1:,1:])])

                
    
    # ----------------------------------------------
    # Analytical methods
    # ----------------------------------------------
    # compute the correlation matrix  in the energy basis
    def computeCorrelationEnergyBasis(self, correlationSiteBasis):        
        # code copied from provided code
        # Step 1: Diagonalize H
        e_vals, U = eigh(self.hamiltonian)  # H = U @ diag(e_vals) @ U.conj().T
        # Step 2: Transform C0 into the energy eigenbasis
        return U.conj().T @ correlationSiteBasis @ U
        
    

    # ----------------------------------------------
    # Plotting and accessor methods
    # ----------------------------------------------
    
    
    # Plot the spectral function  - </method verified/>
    def plotSpectralFunction(self, marker = 'x', scatterColour="r",functionColour="k", figParams=figStandardParams, display=True):
        # fig setup
        inch = 2.54
        fig = plt.figure(figsize=(figParams["width"]/inch, figParams["heightOverWidth"] * figParams["width"]/inch), dpi=figParams["dpi"])
        ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_xlabel("$\omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("$J(\omega)$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        # produce data for function plot
        xData = np.linspace(self.omegas.min(),self.omegas.max(),figParams["nFuncPoints"])
        yData = self.spectralFunction(xData, self.spectralPrefactor, self.bandwidth)
        
        # plot function data
        ax.plot(xData, yData, color=functionColour, linewidth=figParams["markerWidth"])
        
        # plot sampled data
        ax.scatter(self.omegas, self.spectralValues, marker = marker, color=scatterColour, s=figParams["markerArea"], linewidth=figParams["markerWidth"])
        
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    # plot correlation matrix in both bases  - </method verified/>
    def plotCorrelationMatrix(self, t, figParams=figStandardParams, display=True):
        # fig setup
        inch = 2.54
        fig, ax = plt.subplots(1,2,figsize=(figParams["width"]/inch, figParams["heightOverWidth"] * figParams["width"]/inch), dpi=figParams["dpi"])
        pltTimesNewRoman()
        
        ax[0].tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="out", length=2, rotation=45)
        ax[1].tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="out", length=2, rotation=45)
        
        # get correlation matrices
        correlationSiteBasis = self.correlations.get(t)
        if(correlationSiteBasis is None):
            return None
        
        correlationEnergyBasis = self.computeCorrelationEnergyBasis(correlationSiteBasis)
        
        # code copied from provided code
        im0 = ax[0].matshow(np.log10(np.abs(correlationSiteBasis) + 1e-6), cmap="inferno")
        im1 = ax[1].matshow(np.log10(np.abs(correlationEnergyBasis) + 1e-6), cmap="inferno")
        
        ax[0].set_title("C in the original basis", fontsize=figParams["fontsize"])
        ax[1].set_title("C in the energy basis",fontsize=figParams["fontsize"])
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
        fig.colorbar(im0, cax=cbar_ax, location="bottom")
        
        
        fig.tight_layout(pad=0.5)
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    def plotSystemOccupation(self, figParams=figStandardParams, plotColor="k", analyticalThermalizationColor="r", display=True):
        inch = 2.54
        fig = plt.figure(figsize=(figParams["width"]/inch, figParams["heightOverWidth"] * figParams["width"]/inch), dpi=figParams["dpi"])
        ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_xlabel("Time",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("System occupation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        # get and plot occupation data over time
        times  = list(self.correlations.keys())
        
        ax.plot(times, self.density.real, color=plotColor, linewidth=figParams["markerWidth"])
        
        # plot analytical occupancy
        ax.plot(times, np.full(len(times), self.analyticalSystemOccupancy), color=analyticalThermalizationColor, linewidth=figParams["markerWidth"])
        
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    def plotBathOccupation(self, bandwidthMultiplier=1.2,figParams=figStandardParams, display=True):
        inch = 2.54
        fig = plt.figure(figsize=(figParams["width"]/inch, figParams["heightOverWidth"] * figParams["width"]/inch), dpi=figParams["dpi"])
        ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_xlabel("$\Omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("Occupation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
                 
        # plot bath occupation
        lastTime = list(self.correlations.keys())[-1]
        for i in range(0, len(self.times),1):
            bathDensity = self.bathDensity[i]
            time = self.times[i]
            ax.plot(self.omegas,bathDensity, color= plt.cm.coolwarm(time/lastTime))
            #print("colour: ", plt.cm.coolwarm(time/lastTime), " = ", time/lastTime)
            
        # plot fermi dirac distribution
        fermiOmegas = np.linspace(-self.bandwidth*bandwidthMultiplier, self.bandwidth*bandwidthMultiplier, self.nBathSites)
        fermiVals = self.fermi(fermiOmegas)
        ax.plot(fermiOmegas, fermiVals, color="k", linewidth=figParams["markerWidth"], linestyle=":")
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    
# =============================================================================
# Testing
# =============================================================================
spectralFunction = lambda omega, prefactor, bandwidth: prefactor * np.sqrt(1.-(omega/bandwidth)**2)
prefactor = 0.005
n=300
bandwidth = 0.2
energyScale = 0
beta = 2
mu = 1

dt = 1
tMax = 100

sim = simulation("testSim", spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu)
sim.setupSim()

#thermalization = analyticalThermalization(spectralFunction, prefactor, bandwidth, energyScale, beta, mu)
#thermalization.plotA()
#print("occupancy: ", thermalization.computeOccupancy())
#sim.plotSpectralFunction()
sim.simulationCompute(dt, tMax)
#sim.saveAll("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/baseCode")
#sim.save("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/baseCode")
#sim.load("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/baseCode", "testSim")
#print("Hamiltonian: ")
#print(sim.hamiltonian)
"""print("diagnostics: ")
print(sim.correlations[0])
print("-------")
print(sim.correlations[list(sim.correlations.keys())[-1]])"""
#sim.plotCorrelationMatrix(0)

#sim.simulationCompute(dt, tMax)
#print("HERE")
#sim.computeDensity()
#sim.plotSystemOccupation()
#sim.plotBathOccupation()

sim.saveAll("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/baseCode")