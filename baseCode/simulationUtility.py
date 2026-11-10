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
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = "/usr/local/bin/ffmpeg"
import scipy
from scipy.linalg import eigh
import os
import json
import torch
import inspect
import pickle
import functools

import NayanGeneralUtils.plotting as standardPlotting
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
figStandardParams = {"width":8.75, 
                     "heightOverWidth":3/4, 
                     "dpi":600,
    
                    "markerWidth":0.6, 
                     "markerArea":12, 
                     "plotColor":"k",
                     
                     "fontsize":8, 
                     "legendFont":8,
                     "axisTickFont":7.5,
                     
                     "gridColor":'k', 
                     "gridStyle":':',
                     "gridWidth":0.6, 
                     
                     "errCapsize":2, 
                     "histWidth":0.5, 
                     
                     "nFuncPoints":1000,
                     "labelpad":-0.4,
                     "inch":2.54,
                     "legendLoc":"upper right",
                     "legendLocMid":"center right",
                     "legendLocLow":'lower right',
                     
                     "fps":25}

# utility function - creates edited copy of dictionary
def copyDict(dictionary, newVals):
    newDict = dictionary.copy()
    for key, val in newVals.items():
        newDict[key] = val
    return newDict

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
    if(lambdaFunc is None): # null checker
        return None
    
    funcString = str(inspect.getsourcelines(lambdaFunc)[0])
    funcString = funcString.strip("['\\n']").split(" = ")[1]
    return funcString


def isUnitary(matrix, printProduct=False):
    transpose = matrix.transpose().conjugate()
    product = np.matmul(matrix, transpose)
    matrixSize = len(matrix)
    if(printProduct):
        print(product)
    error = np.linalg.norm(np.eye(matrixSize) - product)
    return (error < np.finfo(matrix.dtype).eps * 10.0 *matrixSize)


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
        integralBelow = scipy.integrate.quad(integrand, -self.bandwidth, omegaVal-self.eta, points=[-self.bandwidth, omegaVal],limit=1000, complex_func=False)
        integralAbove = scipy.integrate.quad(integrand, omegaVal+self.eta, self.bandwidth, points=[self.bandwidth, omegaVal],limit=1000, complex_func=False)
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
    
    def plotA(self,fig=None, ax=None, figParams=figStandardParams, display=True):
        # fig setup
        if(fig is None or ax is None): # allow passing of fig/ax object in for subfigs and such
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
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
    def __init__(self, name, spectralFunction, spectralPrefactor, bandwidth, nBathSites, epsilon, beta, mu, animateFps=25, animateDuration=10, animateFormat="gif",initialSystemOccupation=0,usePytorch=True, saveCorrelations=False):
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
        
        # animation details
        self.animateFps = animateFps
        self.animateDuration = animateDuration
        self.animateFormat = animateFormat
        
        # setup analytical thermalization object
        if(not (spectralFunction is None or spectralPrefactor is None or bandwidth is None or epsilon is None or beta is None or mu is None)):
            self.analyticalThermalization = analyticalThermalization(spectralFunction,  spectralPrefactor, bandwidth, epsilon, beta, mu)
            self.analyticalSystemOccupancy = self.analyticalThermalization.computeOccupancy()
        else:
            self.analyticalThermalization = None
            self.analyticalSystemOccupancy = None
    
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
        
        self.analyticalThermalization = analyticalThermalization(self.spectralFunction, self.spectralPrefactor, self.bandwidth, self.epsilon, self.beta, self.mu)
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
    def saveAll(self, directory, produceAnimatedPlot=False):
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
        saveFigure(workingDirectory, self.name+"_AomegaSpectralPlotCombined",self.plotSpectralAndAfunction(display=False))
        saveFigure(workingDirectory, self.name+"_spectralFunctionFlipped",self.plotSpectralFunctionFlippedAxies(display=False))
        
        
        if(produceAnimatedPlot):
            self.animateSystemAndBathEvolution(workingDirectory, duration=self.animateDuration, fps=self.animateFps, vidFormat=self.animateFormat)
        
        
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
    
    # TODO: Implement
    # Simulate the system until the system occupation is within eta of the analytical thermalization - can probably be implemented w/binary search over time domain
    def simulateToThermalization(self, eta):
        pass
    
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
    def plotSpectralFunction(self, fig=None, ax=None, legend=False, spectralLabel="Spectral function", bathSiteLabel="Bath sites", marker = 'x', scatterColour="r",functionColour="k", figParams=figStandardParams, display=True):
        # fig setup
        if(fig is None or ax is None): # allow passing of fig/ax object in for subfigs and such
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_xlabel("$\omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("$J(\omega)$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        # produce data for function plot
        xData = np.linspace(self.omegas.min(),self.omegas.max(),figParams["nFuncPoints"])
        yData = self.spectralFunction(xData, self.spectralPrefactor, self.bandwidth)
        
        # plot function data
        ax.plot(xData, yData, color=functionColour, linewidth=figParams["markerWidth"], label=spectralLabel)
        
        # plot sampled data
        ax.scatter(self.omegas, self.spectralValues, marker = marker, color=scatterColour, s=figParams["markerArea"], linewidth=figParams["markerWidth"],label=bathSiteLabel)
        
        if(legend):
            ax.legend(loc=figParams["legendLoc"],fontsize=figParams["legendFont"])
        
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    # Plot the spectral function  - </method verified/>
    def plotSpectralFunctionFlippedAxies(self, fig=None, ax=None, legend=False, spectralLabel="Spectral function", bathSiteLabel="Bath sites", marker = 'o', scatterColour="r",functionColour="k", figParams=figStandardParams, display=True):
        # fig setup
        print("fig", fig)
        if(fig is None or ax is None): # allow passing of fig/ax object in for subfigs and such
            fig = plt.figure(figsize=((2/3)*figParams["width"]/figParams["inch"], figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_ylabel("$\omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_xlabel("$J(\omega)$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        # produce data for function plot
        yData = np.linspace(self.omegas.min(),self.omegas.max(),figParams["nFuncPoints"])
        xData = self.spectralFunction(yData, self.spectralPrefactor, self.bandwidth)
        
        # plot function data
        ax.plot(xData, yData, color=functionColour, linewidth=figParams["markerWidth"], label=spectralLabel)
        
        # plot sampled data
        ax.scatter(self.spectralValues, self.omegas, marker = marker, color=scatterColour, s=figParams["markerArea"], linewidth=figParams["markerWidth"],label=bathSiteLabel)
        #ax.set_box_aspect(1)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        if(legend):
            ax.legend(loc=figParams["legendLoc"],fontsize=figParams["legendFont"])
        
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    # TODO: needs work - grid axes are fucked - may want to set y axis lims to help as well as set only one to have grids
    def plotSpectralAndAfunction(self, fig=None, ax=None, legend=True, spectralLabel="Spectral function", bathSiteLabel="Bath sites", Alabel=r"$A(\omega)$", marker = 'o', scatterColour="r",spectralFunctionColour="k", figParams=figStandardParams, display=True):
        if(fig is None or ax is None): # allow passing of fig/ax object in for subfigs and such
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        fig = self.plotSpectralFunction(fig=fig, ax=ax, legend=False, spectralLabel=spectralLabel, bathSiteLabel=bathSiteLabel, marker = marker, scatterColour=scatterColour,functionColour=spectralFunctionColour, figParams=figParams, display=False)
        
        ax2 = ax.twinx()
        self.analyticalThermalization.plotA(fig, ax2, figParams=figParams)
        
        if(legend):
            ax.legend(loc=figParams["legendLoc"],fontsize=figParams["legendFont"])
        
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
        
    
    # plot correlation matrix in both bases  - </method verified/>
    def plotCorrelationMatrix(self, t,fig=None, ax=None, figParams=figStandardParams, display=True):
        # fig setup
        if(fig is None or ax is None):
            fig, ax = plt.subplots(1,2,figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            pltTimesNewRoman()
        
        ax[0].tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="out", length=2, rotation=45)
        ax[1].tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="out", length=2, rotation=45)
        
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
    
    def plotSystemOccupation(self, fig=None, ax=None, legend=False, linestyle="-",plotThermalization=True, axLims=[0,1], densityLabel=r"$n_{occ}(t)$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=figStandardParams, analyticalThermalizationColor="r", display=True):
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_xlabel("Time",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("System occupation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        
            
        ax.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        # get and plot occupation data over time
        times  = list(self.correlations.keys())
        
        ax.plot(times, self.density.real, color=figParams["plotColor"], linewidth=figParams["markerWidth"], label=densityLabel, linestyle=linestyle)
        
        if(axLims is not None):
            ax.set_ylim(axLims)
            ax.set_xlim([np.min(times), np.max(times)])
        
        # plot analytical occupancy
        if(plotThermalization):
            ax.plot(times, np.full(len(times), self.analyticalSystemOccupancy), color=analyticalThermalizationColor, linewidth=figParams["markerWidth"], label=thermalizationLabel)
        if(legend):
            ax.legend(loc=figParams["legendLocMid"],fontsize=figParams["legendFont"])
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    # helper method
    def plotBathOccupationPart(self, ax, i, color, figParams):
        bathDensity = self.bathDensity[i]
        ax.plot(self.omegas,bathDensity, color= color, linewidth=figParams["markerWidth"])
        return ax
    
    def plotBathOccupation(self, fig=None, ax=None, bandwidthMultiplier=1.2,axLims=[0,1],figParams=figStandardParams, display=True):
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        ax.set_xlabel("$\Omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("Occupation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
                 
        # plot bath occupation
        lastTime = list(self.correlations.keys())[-1]
        for i in range(0, len(self.times),1):
            bathDensity = self.bathDensity[i]
            time = self.times[i]
            color = plt.cm.coolwarm(time/lastTime)
            ax = self.plotBathOccupationPart(ax, i, color, figParams)
            #print("colour: ", plt.cm.coolwarm(time/lastTime), " = ", time/lastTime)
            
        # plot fermi dirac distribution
        fermiOmegas = np.linspace(-self.bandwidth*bandwidthMultiplier, self.bandwidth*bandwidthMultiplier, self.nBathSites)
        fermiVals = self.fermi(fermiOmegas)
        ax.plot(fermiOmegas, fermiVals, color="k", linewidth=figParams["markerWidth"], linestyle=":")
        
        if(axLims is not None):
            ax.set_ylim(axLims)
            ax.set_xlim([np.min(self.omegas)*bandwidthMultiplier, np.max(self.omegas)*bandwidthMultiplier])
        
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    def animateSystemAndBathHelper(self, frame, fig=None, axSys=None, axBath=None, Dt=1, startTime=0, endTime=0, bandwidthMultiplier=1.2, plotColor="k",densityLabel=r"$n_{occ}(t)$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", analyticalThermalizationColor="r", plotThermalization=True, figParams=figStandardParams):
        axSys.clear()
        axBath.clear()
        
        # get appropriate index
        index = frame * Dt
        
        # set axSys axis params
        axSys.set_title("System occupation")
        axSys.set_xlabel("Time",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        axSys.set_ylabel("Occupation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        axSys.set_xlim([startTime, endTime])
        axSys.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        axSys.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
    
        # set axBath axis params
        axBath.set_title("Bath occupation")
        axBath.set_xlabel(r"$\omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        axBath.set_ylim([0,1])
        axBath.set_xlim([-self.bandwidth*bandwidthMultiplier, self.bandwidth*bandwidthMultiplier])
        axBath.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="inout", length=2)
        axBath.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        bathLine = self.plotBathOccupationPart(axBath, index, plotColor, figParams)
        fermiOmegas = np.linspace(-self.bandwidth*bandwidthMultiplier, self.bandwidth*bandwidthMultiplier, self.nBathSites)
        fermiVals = self.fermi(fermiOmegas)
        fermiLine = axBath.plot(fermiOmegas, fermiVals, color=analyticalThermalizationColor, linewidth=figParams["markerWidth"], linestyle=":")
        
        systemLine = axSys.plot(self.times[0:index], self.density[0:index].real, color=plotColor, linewidth=figParams["markerWidth"], label=densityLabel)
        
        fig.subplots_adjust(wspace=0)
        #fig.tight_layout()
        
        if(plotThermalization):
            thermalLine = axSys.plot(self.times, np.full(len(self.times), self.analyticalSystemOccupancy), color=analyticalThermalizationColor, linewidth=figParams["markerWidth"], label=thermalizationLabel)
            
            return bathLine, fermiLine, systemLine, thermalLine
        
        return bathLine, fermiLine, systemLine
    
    def animateSystemAndBathEvolution(self, saveLoc,duration=10, fps=25, vidFormat="gif",axLims=[0,1],figParams=figStandardParams):
        fig, axs = plt.subplots(1,2, sharey=True, figsize=(14/figParams["inch"], 7/figParams["inch"]), dpi=figParams["dpi"])
        #fig.tight_layout()
        axSys = axs[0]
        axBath = axs[1]
        startTime = np.min(self.times)
        endTime = np.max(self.times)
        frames = int(duration*fps)
        Dt = int(len(self.times) / frames) # number of timesteps per frame
        # create animation object
        animatedPlot = animation.FuncAnimation(fig, functools.partial(self.animateSystemAndBathHelper,fig=fig, Dt=Dt,startTime=startTime, endTime=endTime, axSys=axSys, axBath=axBath, figParams=figParams), frames=frames, interval=(1/fps)/1000)
        
        gifWriter = animation.PillowWriter(fps=fps)
        FFwriter =  animation.FFMpegWriter(fps=fps)#animation.writers['ffmpeg']
        # create path and gif name
        path = saveLoc + "/" + self.name + "_systemBathEvo"
        
        # save
        if(vidFormat == "gif"):
            animatedPlot.save(path+".gif", gifWriter)
        elif(vidFormat == "mp4"):
            animatedPlot.save(path+".mp4", FFwriter)
# =============================================================================
# dynamic map dual setup
# =============================================================================
class dualSimulation():

    # ----------------------------------------------
    # Constructor methods
    # ----------------------------------------------
    def __init__(self, name, spectralFunction, spectralPrefactor, bandwidth, nBathSites, epsilon, beta, mu, initialOccupations=np.linspace(0,1, 10), usePytorch=True, saveCorrelations=False):
        self.name = name
        self.occupiedSimulation = simulation(name+"_occupied", spectralFunction, spectralPrefactor, bandwidth, nBathSites, epsilon, beta, mu, initialSystemOccupation=1)
        self.unoccupiedSimulation = simulation(name+"_unoccupied", spectralFunction, spectralPrefactor, bandwidth, nBathSites, epsilon, beta, mu, initialSystemOccupation=0)
        
        if(not(spectralFunction is None or spectralPrefactor is None or bandwidth is None or nBathSites is None or epsilon is None or beta is None or mu is None)):
            self.occupiedSimulation.setupSim()
            self.unoccupiedSimulation.setupSim()
        
        self.analyticalSystemOccupancy = self.occupiedSimulation.analyticalSystemOccupancy
        self.numericSystemOccupancy = None
        
        self.initialOccupations = initialOccupations
        self.dynamicalMaps = {}
    # ----------------------------------------------
    # Saving and loading methods
    # ----------------------------------------------
    def save(self, directory):
        createFolder(directory, self.name)
        workingDirectory = directory + "/" + self.name
        
        self.occupiedSimulation.save(workingDirectory)
        self.unoccupiedSimulation.save(workingDirectory)
        
        # save dynamic map data
        pickle.dump(self.dynamicalMaps, open(workingDirectory+'/'+self.name+'.dynData', 'wb'))
    
    def load(self, directory, folderName):
        workingDirectory = directory + "/" +folderName
        if(os.path.isdir(workingDirectory)):
            self.name = folderName
            self.occupiedSimulation.load(workingDirectory, self.name+"_occupied")
            self.occupiedSimulation.setupSim()
            self.unoccupiedSimulation.load(workingDirectory, self.name+"_unoccupied")
            self.unoccupiedSimulation.setupSim()
            
            # load dynamic map data
            self.dynamicalMaps = pickle.load(open(workingDirectory +"/"+self.name+'.dynData', 'rb'))
            
            # extract additional data
            infoDict = loadDictFromJson(workingDirectory, self.name+"_simulationInfo")
            self.initialOccupations = infoDict["initialOccupations"]
            
            # recompute thermalization
            self.analyticalSystemOccupancy = self.occupiedSimulation.analyticalSystemOccupancy
            self.computeNumericThermalization(searchSteps=20)
            
            
    def saveAll(self, directory):
        self.save(directory)
        workingDirectory = directory + "/" + self.name
        
        infoDict ={"name":self.name,
                     "saveCorrelations":True,
                     "spectralFunction":self.occupiedSimulation.spectralFunctionStr,
                     "spectralPrefactor":self.occupiedSimulation.spectralPrefactor,
                     "bandwidth":self.occupiedSimulation.bandwidth,
                     "nBathSites":int(self.occupiedSimulation.nBathSites),
                     "epsilon":self.occupiedSimulation.epsilon,
                     "beta":self.occupiedSimulation.beta,
                     "mu":self.occupiedSimulation.mu,
                     "timeSteps":len(list(self.occupiedSimulation.correlations.keys())),
                     "tMax":float(list(self.occupiedSimulation.correlations.keys())[-1]),
                     "initialOccupations":self.initialOccupations,
                     }
        
        saveDictAsJson(workingDirectory, self.name+"_simulationInfo", infoDict)
        
        # save plots
        saveFigure(workingDirectory, self.name+"_dynamicalEvo",self.plotBothOccupationsFromMap(display=False))
        saveFigure(workingDirectory, self.name+"_compareEvo",self.plotBothOccupationsBoth(display=False))
        saveFigure(workingDirectory, self.name+"_rangeInitial",self.plotManyOccupations(self.initialOccupations, display=False))
        saveFigure(workingDirectory, self.name+"_maxDeviation",self.plotMaximumDeviationOverOccupation(display=False))
    
    # ----------------------------------------------
    # Analytical methods
    # ----------------------------------------------
    
    # Compuational method - computes dynamic map f
    def computeDynamicMap(self):
        times = list(self.occupiedSimulation.correlations.keys())
        occupiedCorrelations = self.occupiedSimulation.density
        unOccupiedCorrelations = self.unoccupiedSimulation.density
        #unitary = []
        for i in range(0, len(times)):
            time = times[i]
            C_occ = occupiedCorrelations[i]
            C_unocc = unOccupiedCorrelations[i]
            
            dynamicalMap = np.array([[1-C_occ,1-C_occ-C_unocc],[C_occ,C_occ+C_unocc]],dtype=np.complex128)
            dynamicalMapAlt = np.array([[(1-C_unocc),(1-C_occ)],[C_unocc,C_occ]],dtype=np.complex128)
            #unitary = unitary+ [isUnitary(dynamicalMapAlt, printProduct=False)]
            self.dynamicalMaps[time] = dynamicalMapAlt
            
        self.computeNumericThermalization(searchSteps=20)
        #print(unitary)
        #print(self.dynamicalMaps[times[0]])
        #print(self.dynamicalMaps[times[1]])
        #print(self.dynamicalMaps[times[2]])
    
    # computes the state evolution using the dynamical map given some initial occupation
    def computeStateEvolution(self, initialOccupation):
        times = list(self.dynamicalMaps.keys())
        states = {times[0]:np.array([1-initialOccupation, initialOccupation], dtype=np.complex128)}
        
        #lastTime = times[0]
        for i in range(1, len(times)):
            time = times[i]
            states[time] = np.matmul(self.dynamicalMaps[time], states[times[0]])
            
            #lastTime = times[i]
            
        return states
    
    # helper method for binarySearchEigenvector - searches for p value which best satisfies dynamicMap [1-p, p] = a[1-p, p]
    def binarySearchEigenvector(self, dynamicMap, lowerBound, upperBound, searchSteps):
        
        # select upper and lower test values
        pLow = lowerBound + (upperBound-lowerBound)/3
        pHigh =lowerBound + 2*(upperBound-lowerBound)/3
        
        # build corresponding vectors
        vecLow = np.array([1-pLow, pLow])
        vecHigh = np.array([1-pHigh, pHigh])
        
        # matrix multiply these w/dynamic map
        prodLow = np.matmul(dynamicMap, vecLow)
        prodHigh = np.matmul(dynamicMap, vecHigh)
        
        # compute difference between in/out vectors
        diffLow = np.linalg.norm(vecLow-prodLow)
        diffHigh = np.linalg.norm(vecHigh-prodHigh)
        
        # base case
        if(searchSteps==1):
            # select and return appropriate vector
            if(diffLow <= diffHigh):
                return vecLow, diffLow
            return vecHigh, diffHigh
        
        # recursive case
        else:
            if(diffLow <= diffHigh):
                newLowerBound = lowerBound
                newUpperBounds = lowerBound + (upperBound-lowerBound)/2
                return self.binarySearchEigenvector(dynamicMap, newLowerBound, newUpperBounds, searchSteps-1)
            
            else:
                newLowerBound = lowerBound + (upperBound-lowerBound)/2
                newUpperBounds = upperBound
                return self.binarySearchEigenvector(dynamicMap, newLowerBound, newUpperBounds, searchSteps-1)
    
    def computeNumericThermalization(self, searchSteps=10, minTime=1): # note minTime exists to remove early outliers from data
        times = list(self.dynamicalMaps.keys())
        
        diffAbs1s = []
        for i in range(1, len(times)):
            timePrev = times[i-1]
            time = times[i]
            diff1 = self.dynamicalMaps[time] - self.dynamicalMaps[timePrev]
            diffAbs1s = diffAbs1s + [np.linalg.norm(diff1)]
        
        minTimeIndex = np.argwhere(np.array(times)>minTime)[0][0]
        timeStable = times[np.argmin(diffAbs1s[minTimeIndex:])]
        stableMap = self.dynamicalMaps[timeStable]
        
        closestVector, err = self.binarySearchEigenvector(stableMap, 0, 1, searchSteps)
        #print(closestVector)
        #print(err)
        self.numericSystemOccupancy = closestVector[1]
        #return closestVector[1]
    
    # ----------------------------------------------
    # Non markovianity methods 0 - Maximum gap
    # ----------------------------------------------
    
    def maximumDeviation(self, initialOccupation=1):
        # get system occupation data
        times = list(self.dynamicalMaps.keys())
        states = self.computeStateEvolution(initialOccupation)
        systemOccupation = np.array([val[1].real for key, val in states.items()])
        
        # get relative maxima
        maxDeviation = None
        maxTime = None
        indexMax = np.argmax(systemOccupation)
        # identify non-thermalized maxima
        if(not(indexMax == len(systemOccupation)-1) and not(indexMax == 0) and systemOccupation[indexMax-1] <= systemOccupation[indexMax] and systemOccupation[indexMax+1] <= systemOccupation[indexMax]):
            maxDeviation = initialOccupation - systemOccupation[indexMax]
            
            maxTime = times[indexMax]
            
        # get relative minima
        minDeviation = None
        minTime = None
        indexMin = np.argmin(systemOccupation)
        # identify non-thermalized minima
        if(not(indexMin == len(systemOccupation)-1) and not(indexMin == 0) and systemOccupation[indexMin-1] > systemOccupation[indexMin] and systemOccupation[indexMin+1] > systemOccupation[indexMin]):
            minDeviation = initialOccupation - systemOccupation[indexMin]
            
            minTime= times[indexMin]
        
        # compare relative extrema
        if(maxDeviation is None and minDeviation is None):
            return None, None, None
        
        if(maxDeviation is None):
            return minDeviation, minTime, systemOccupation[indexMin]
        
        if(minDeviation is None):
            return maxDeviation, maxTime
        
        if(abs(maxDeviation) >= abs(minDeviation)):
            return maxDeviation, maxTime, systemOccupation[indexMax]
        
        return minDeviation, minTime, systemOccupation[indexMin]
        
    def maximumDeviationFromThermalization(self):
        return self.maximumDeviation(self.numericSystemOccupancy)
    
    # plots maximum deviation over initial occupancy
    def plotMaximumDeviationOverOccupation(self, fig=None, ax=None, figParams=figStandardParams, plotThermalization=True, display=True, numPoints=1000):
        
        occupancies = np.linspace(0,1, num=numPoints)
        maxDeviations = [self.maximumDeviation(initialOccupation=occupancy)[0] for occupancy in occupancies]
        
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        # set axis labels and related parameters
        ax.set_xlabel("Initial occupancy",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("Max deviation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        ax.plot(occupancies, maxDeviations, color=figParams["plotColor"], linewidth=figParams["markerWidth"])
        
        if(plotThermalization):
            ax.axvline(x=self.numericSystemOccupancy, color="r", linestyle="-", linewidth=figParams["markerWidth"])

        if(display):
            plt.show()
            
        return fig

    # ----------------------------------------------
    # Non markovianity methods 1 - Memory kernels
    # ----------------------------------------------
    
    # ----------------------------------------------
    # Non markovianity methods 2 - Information backflow
    # ----------------------------------------------
    
    # ----------------------------------------------
    # Non markovianity methods 3 - Failure of CP positivity
    # ----------------------------------------------
    
    # ----------------------------------------------
    # Simulation methods
    # ----------------------------------------------
    def simulateCompute(self, dt, tMax):
        self.occupiedSimulation.simulationCompute(dt, tMax)
        self.unoccupiedSimulation.simulationCompute(dt, tMax)
        
    # Simulate the system until the system occupation is within eta of the analytical thermalization - can probably be implemented w/binary search over time domain
    def simulateToThermalization(self, eta):
        pass
    # ----------------------------------------------
    # Plotting methods
    # ----------------------------------------------
    
    def plotOccupationFromMap(self, initialOccupation, legend=False, label=r"$n_{occ}(t)$", plotThermalization=False, thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", fig=None, ax=None, figParams=figStandardParams, analyticalThermalizationColor='r', display=True):
        # get system occupation data
        states = self.computeStateEvolution(initialOccupation)
        systemOccupation = np.array([val[1] for key, val in states.items()])
        
        # generate fig/ax objects if not provided
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], 1/2 * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        # set axis labels and related parameters
        ax.set_xlabel("Time",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        ax.set_ylabel("System occupation",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
        
        ax.tick_params(axis='both', which='major', labelsize=figParams["axisTickFont"], direction="in", length=2)
        
        ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
        
        # get and plot occupation data over time
        times = list(self.dynamicalMaps.keys())
        
        ax.plot(times, systemOccupation.real, color=figParams["plotColor"], linewidth=figParams["markerWidth"], label=label)
        

        if(plotThermalization):
            ax.plot(times, np.full(len(times), self.numericSystemOccupancy), color=analyticalThermalizationColor, linewidth=figParams["markerWidth"], label=thermalizationLabel)
        
        if(legend):
            ax.legend(loc=figParams["legendLocMid"],fontsize=figParams["legendFont"])
        # display plot if desired
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    def plotBothOccupationsFromMap(self, fig=None, ax=None, figParams=figStandardParams, display=True):
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        mainFigArgs = figParams.copy()

        
        fig = self.plotOccupationFromMap(1, legend=False,plotThermalization=False, label=r"dynamic $n_{occ}(t=0)=1$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", fig=fig, ax=ax, figParams=copyDict(mainFigArgs, {"plotColor":"k"}), analyticalThermalizationColor='r', display=False)
        fig = self.plotOccupationFromMap(0, legend=False,plotThermalization=True, label=r"dynamic $n_{occ}(t=0)=0$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", fig=fig, ax=ax, figParams=copyDict(mainFigArgs, {"plotColor":"g"}), analyticalThermalizationColor='r', display=False)
        
        ax.legend(loc=figParams["legendLocLow"], fontsize=figParams["legendFont"])
        
        if(display):
            plt.show()
            
        # return fig
        return fig
    
    def plotManyOccupations(self, initialOccupations, fig=None, ax=None, figParams=figStandardParams, display=True, plotThermalization=True):
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
    
        
        for i in range(0, len(initialOccupations)):
            color = plt.cm.seismic(2*i/len(initialOccupations))
            tempFigArgs = copyDict(figParams, {"plotColor":color})
            fig = self.plotOccupationFromMap(initialOccupations[i], legend=False,plotThermalization=False, label=r"$n_{occ}(t=0)="+str(initialOccupations[i])+r"$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", fig=fig, ax=ax, figParams=tempFigArgs, analyticalThermalizationColor='r', display=False)
            dev, time, val = self.maximumDeviation(initialOccupation=initialOccupations[i])
            ax.scatter([time], [val], marker="x", color=tempFigArgs["plotColor"], s=tempFigArgs["markerArea"], linewidth=tempFigArgs["markerWidth"])
        
        
        if(plotThermalization):
            times = list(self.dynamicalMaps.keys())
            ax.plot(times, np.full(len(times), self.numericSystemOccupancy), color="k", linewidth=figParams["markerWidth"], label="$\lim_{t\to\infty}(n_{occ}(t))$")
            
        if(display):
            plt.show()
            
        # return fig
        return fig
        
    # compares simulated vs dynamic plots
    def plotBothOccupationsBoth(self, fig=None, ax=None, figParams=figStandardParams, display=True):
        if(fig is None or ax is None):
            fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
            ax = fig.add_subplot()
        pltTimesNewRoman()
        
        # define all plotting dicts
        outlineParams = copyDict(figParams, {"markerWidth":1.2, "plotColor":"w"})
        simParams = copyDict(figParams, {"markerWidth":0.65*outlineParams["markerWidth"], "plotColor":"k"})
        mainFigArgs = copyDict(figParams, {"markerWidth":2.3*outlineParams["markerWidth"], "plotColor":"cornflowerblue"})
        
        # plot dynamic map line
        fig = self.plotOccupationFromMap(1, legend=False,plotThermalization=False, label=r"dynamic $n_{occ}$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", fig=fig, ax=ax, figParams=mainFigArgs, analyticalThermalizationColor='r', display=False)
        fig = self.plotOccupationFromMap(0, legend=False,plotThermalization=False, label=None, thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", fig=fig, ax=ax, figParams=mainFigArgs, analyticalThermalizationColor='r', display=False)
        
        # plot large white outline lines
        self.occupiedSimulation.plotSystemOccupation(fig=fig, ax=ax, legend=False, linestyle="-",plotThermalization=False, axLims=[0,1], densityLabel=None, thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=outlineParams, analyticalThermalizationColor="r", display=False)
        self.unoccupiedSimulation.plotSystemOccupation(fig=fig, ax=ax, legend=False, linestyle="-",plotThermalization=False, axLims=[0,1], densityLabel=None, thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=outlineParams, analyticalThermalizationColor="r", display=False)
        
        # plot smaller black dotted line
        self.occupiedSimulation.plotSystemOccupation(fig=fig, ax=ax, legend=False, linestyle="--",plotThermalization=False, axLims=[0,1], densityLabel=r"simulated $n_{occ}$", thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=simParams, analyticalThermalizationColor="r", display=False)
        self.unoccupiedSimulation.plotSystemOccupation(fig=fig, ax=ax, legend=False, linestyle="--",plotThermalization=False, axLims=[0,1], densityLabel=None, thermalizationLabel=r"$\lim_{t\to\infty}(n_{occ}(t))$", figParams=simParams, analyticalThermalizationColor="r", display=False)
        
        
        times = list(self.dynamicalMaps.keys())
        ax.plot(times, np.full(len(times), self.numericSystemOccupancy), color="r", linewidth=figParams["markerWidth"], label="$\lim_{t\to\infty}(n_{occ}(t))$")
        
        
        ax.legend(loc=figParams["legendLocLow"], fontsize=figParams["legendFont"])
        
        if(display):
            plt.show()
            
        # return fig
        return fig
    
# =============================================================================
# Testing
# =============================================================================
"""spectralFunction = lambda omega, prefactor, bandwidth: prefactor * np.sqrt(1.-(omega/bandwidth)**2)
prefactor = 0.2
n=300
bandwidth = 2
energyScale = 0
beta = 7
mu = 0.1

dt = 0.1
tMax = 50"""
"""
sim = simulation("testSim", spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu)
sim.setupSim()
sim.simulationCompute(dt, tMax)
print("sim complete")
sim.animateSystemAndBathEvolution("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials")
print("done")
#thermalization = analyticalThermalization(spectralFunction, prefactor, bandwidth, energyScale, beta, mu)
#thermalization.plotA()
sim.saveAll("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/baseCode")
"""


#print("occupancy: ", thermalization.computeOccupancy())
#sim.plotSpectralFunction()
#sim.simulationCompute(dt, tMax)
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

#sim.saveAll("/Users/nayandusoruth/Desktop/Y4physics/Dissertation/Y4DissertationCodeMaterials/baseCode")

"""
dualSimulation = dualSimulation("testDual",  spectralFunction, prefactor, bandwidth, n, energyScale, beta, mu)
dualSimulation.simulateCompute(dt, tMax)
dualSimulation.computeDynamicMap()
print("numeric thermalization: ", dualSimulation.numericSystemOccupancy)
print("maximumDeviation:", dualSimulation.maximumDeviation(initialOccupation=0.8))
dualSimulation.plotMaximumDeviationOverOccupation()
dualSimulation.plotOccupationFromMap(0.55, display=True, plotThermalization=True)
#dualSimulation.unoccupiedSimulation.plotSystemOccupation(display=True)
#dualSimulation.plotBothOccupationsFromMap(display=True)
dualSimulation.plotBothOccupationsBoth(display=True)"""
