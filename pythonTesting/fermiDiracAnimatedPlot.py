#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:10:27 2026

@author: nayandusoruth
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import functools

def pltTimesNewRoman():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.it'] = 'serif:italic'
    plt.rcParams['mathtext.bf'] = 'serif:bold'
    plt.rcParams['mathtext.fontset'] = "custom"

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




def fermiDirac(omega, mu, beta):
    return 1 / (np.exp(beta *(omega - mu))+1)

def animateSystemVariationsHelper(frame, betas=[], labels =[], mu=1, fig=None, ax=None, figParams=figStandardParams):
    ax.clear()

            
    ax.set_xlabel(r"$\omega$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])
    ax.set_ylabel(r"$f(\omega)$",fontsize=figParams["fontsize"], labelpad=figParams["labelpad"])

    ax.tick_params(axis='both', which='major', labelsize=figParams["fontsize"], direction="in", length=2)
    
    ax.grid(which="both", color=figParams["gridColor"], linestyle=figParams["gridStyle"], linewidth=figParams["gridWidth"])
    ax.set_xlim([0,2])
    ax.set_ylim([0,1])
    betaCurrent = betas[frame]
    xs = np.linspace(0, 2, 1000)
    ys = [fermiDirac(x, mu, betaCurrent) for x in xs]
    
    ax.plot(xs, ys, color="k", linewidth=figParams["markerWidth"], label=labels[frame])
    ax.axvline(x=mu, color="r", linestyle="-", linewidth=figParams["markerWidth"],label=r"$\mu=1$" )
    ax.legend(loc=figParams["legendLoc"],fontsize=figParams["legendFont"])
    return ax


def animateSystemVariations(betas, directory, name, figParams=figStandardParams,fps=25, vidFormat="gif"):
    # set up fig/ax
    fig = plt.figure(figsize=(figParams["width"]/figParams["inch"], figParams["heightOverWidth"] * figParams["width"]/figParams["inch"]), dpi=figParams["dpi"])
    ax = fig.add_subplot()
    pltTimesNewRoman()
    
    # frame stuff
    frames = len(betas)
    
    # create labels
    labels = [r"$\beta$" + ": "+'%0.3f' %beta for beta in betas]
    animatedPlot = animation.FuncAnimation(fig, functools.partial(animateSystemVariationsHelper,betas=betas, labels=labels, fig=fig, ax=ax, figParams=figParams), frames=frames, interval=(1/fps)/1000)
    
    gifWriter = animation.PillowWriter(fps=fps)
    FFwriter =  animation.FFMpegWriter(fps=fps)#animation.writers['ffmpeg']
    # create path and gif name
    path = directory + "/" + name
    
    # save
    if(vidFormat == "gif"):
        animatedPlot.save(path+".gif", gifWriter)
    elif(vidFormat == "mp4"):
        animatedPlot.save(path+".mp4", FFwriter)
        
        
betas = np.linspace(10, 1, 5*25)
animateSystemVariations(betas, "/Users/nayandusoruth/Desktop/Y4physics/Dissertation/presentation/Diagrams", "fermiDirac", figParams=figStandardParams,fps=25, vidFormat="mp4")