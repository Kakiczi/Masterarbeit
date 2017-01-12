############################## IMPORTS ####################################################
import matplotlib
import matplotlib.pyplot as plt
#import pylab
import healpy as H
import numpy as np
#import time as tm
#import sys
#import copy


def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def newfig(number,width):
    fig = plt.figure(number,figsize=figsize(width))
    return fig
    
######## PLOTTING
font = {'family' : 'serif',
		#'serif'  : 'cmr10', #'Bitstream Vera Serif', 
		#'weight' : '',
		'size'   : 50} #20
ec = {"markeredgewidth" : 0.1}
matplotlib.rcdefaults()
matplotlib.rc('font', **font)
#matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick.major', pad=25) #10
#matplotlib.rc('xtick.minor', pad=10)
matplotlib.rc('ytick.major', pad=25) #10
#matplotlib.rc('ytick.minor', pad=10)
matplotlib.rc('lines', **ec)
matplotlib.rc('lines', linewidth=2.0)
matplotlib.rc('figure.subplot', bottom=0.15) # Abstand von unterem Plot Ende bis zum Rand des Bildes - nuetzlich um Achsenbeschriftung nach oben zu schieben undgroesser zu machen
matplotlib.rc('figure.subplot', left=0.15)
matplotlib.rc('figure.subplot', top=1.-0.1)
matplotlib.rc('figure.subplot', right=1.-0.05)
#matplotlib.rc('figure.subplot', wspace = 0.12)
matplotlib.rc('figure.subplot', hspace = 0.4)
#matplotlib.rc('legend', fontsize = "medium")
matplotlib.rc('text', usetex=True)


matplotlib.rc('xtick', labelsize=30) #22 
matplotlib.rc('ytick', labelsize=30) #22


matplotlib.rc('figure', figsize=(16,12))
matplotlib.rc('axes', grid=True)
#matplotlib.rcParams['xtick.major.size'] = 2.0
#matplotlib.rcParams['xtick.minor.size'] = 2.0
legend_fs = 30.


GLOB_FIG_NUMBERS = []
FIG_TITLES       = []
color = ["blue", "lightgreen", "orange", "red", "magenta", "darkblue", "green", "brown", "darkred", "purple"]
matplotlib.rcParams['axes.color_cycle']=color
linestyles = ["-", "--", "-.", ":"]
markers = ["o", "s", "p", "d"]

######## HEALPY 
nside = np.power(2,9) 
npix = H.nside2npix(nside)

#globPixCosTheta = np.zeros(npix)
#globPixSinTheta = np.zeros(npix)
#globPixCosPhi = np.zeros(npix)
#globPixSinPhi = np.zeros(npix) 
allPixNeighbours = []


### number events: 57281 (only upgoing)
