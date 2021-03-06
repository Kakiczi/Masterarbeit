############################ IMPORTS ####################################################
import numpy as np
import healpy as H
from scipy import interpolate
from scipy.integrate import quad
from scipy.special import erfinv, erf, gamma, erfcinv
from scipy.stats import gamma as GM
from scipy.stats import t, percentileofscore
import time as tm
import types
import sys, os
import copy
import pickle
import ConfigParser
from numpy.lib.recfunctions import append_fields
import psutil
import math as m
import time
import matplotlib.pyplot as plt
from scipy import integrate
from calcEffArea_class import calcEffArea
from scipy.optimize import fsolve
from scipy.interpolate import RectBivariateSpline
import matplotlib.colors as colors
from SkymapPlot.coords import galactic
from scipy.interpolate import interp1d

import socket

print "\n"

if 'physik.rwth-aachen.de' in socket.gethostname()  or "Ubuntu" in socket.gethostname():
    try:
        from scipy.optimize import curve_fit, leastsq
    except:
        print "Failed to import curve_fit module."
        
from declarations2 import *

if socket.gethostname()=='theo-UL50VT':
        localPath="/home/theo/fusessh/"
else:
        localPath = "/net/scratch_icecube4/user/glauch/"

catalogPath = "/net/scratch_icecube4/user/kalaczynski/Catalogs/"

def plot_aeff_vs_dec_energy(mc, logE_range=(2,9), sinDec_range=(-1,1), bins=[25,10], title=None, savepath=None):
    logEEdges = np.linspace(logE_range[0],logE_range[1], bins[0], endpoint=True)
    deltaLogE = (logEEdges[-1]-logEEdges[0])/(len(logEEdges)-1)
    lowerLogEEdge = np.floor((np.log10(mc["trueE"]) - logEEdges[0])/deltaLogE)*deltaLogE + logEEdges[0]
    upperLogEEdge = lowerLogEEdge + deltaLogE
    EBinWidth = pow(10, upperLogEEdge) - pow(10, lowerLogEEdge)
    print "EBinWidth {} MC  length {}".format(len(EBinWidth), len(mc["trueE"]))

    sinDecEdges = np.linspace(sinDec_range[0],sinDec_range[1],bins[1], endpoint=True)
    deltaSinDec = (sinDecEdges[-1]-sinDecEdges[0])/(len(sinDecEdges)-1)
    solidAngle = deltaSinDec*2*np.pi
    w_aeff = mc["ow"]/EBinWidth/solidAngle
    H, logEEdges, sinDecEdges = np.histogram2d(np.log10(mc["trueE"]), np.sin(mc["trueDec"]), bins=[logEEdges, sinDecEdges], weights=w_aeff)
    H = np.ma.masked_array(H)
    H.mask = H == 0
    
    # to get E and angle instead:
    #EEdges = np.linspace(10**2,10**9, bins[0], endpoint=True)
    #DecEdges = np.linspace(-np.pi/2,np.pi/2,bins[1], endpoint=True)
    #H2, EEdges, DecEdges = np.histogram2d(mc["trueE"], mc["trueDec"], bins=[EEdges, DecEdges], weights=w_aeff)
    #H2 = np.ma.masked_array(H2)
    #H2.mask = H2 == 0
    
    center_logE_bins = np.array(logEEdges[:-1])+(logEEdges[1]-logEEdges[0])/2
    center_sinDec_bins = np.array(sinDecEdges[:-1]) + (sinDecEdges[1]-sinDecEdges[0])/2
    #H_clip0 = H+(H<=0.0)*np.min(H[np.nonzero(H)])

    X, Y = np.meshgrid(sinDecEdges, logEEdges)
    #plt.figure(figsize=(16,9))
    #print H
    #plt.pcolormesh(X,Y, H, norm=colors.LogNorm(vmin=H.min(), vmax=H.max()), cmap="Blues") 
    #plt.xlim(sinDec_range)
    #plt.colorbar().set_label(r"$\log_{10}(A_{eff}/ \mathrm{cm}^2)$")
    #plt.xlabel(r"$\sin(\delta)$")
    #plt.ylabel(r"$\log_{10}(E/\mathrm{GeV})$")
    if title!=None:
        plt.title(title)
    if savepath!=None:
        plt.savefig(savepath, dpi=100)

    spline_logE_sinDec = RectBivariateSpline(center_logE_bins, center_sinDec_bins, H, kx=3, ky=1)
    return H, logEEdges, sinDecEdges, spline_logE_sinDec

def getEffAreaVsDec(): ###		Getting the effective area as a function of the declination angle:
	datapath="/net/scratch_icecube4/user/glauch/DATA/"
	DETECTOR=LoadDETECTOR("1111")
	index=2.13 # assumed spectral index
	Aeff_res=plot_aeff_vs_dec_energy(simulationLoad(DETECTOR[1], datapath, index)[0], logE_range=(2,9), sinDec_range=(np.sin(np.radians(-6)),1), bins=[40,40], title=None, savepath=None)
	
	X,Y=np.meshgrid(Aeff_res[2],Aeff_res[1])
	H=Aeff_res[0]
	
	Aeff= np.sum(H, axis=1)*(X[0][1]-X[0][0])
	deltaE=np.array([10**Aeff_res[1][i+1]-10**Aeff_res[1][i] for i in range(len(Aeff_res[2])-1)])
	E=np.array([(10**Aeff_res[1][i+1]+10**Aeff_res[1][i])/2 for i in range(len(Aeff_res[2])-1)])
	Aeff_theta=[np.sum((E**(-index))*np.concatenate(H[:,i:i+1])*deltaE)/np.sum((E**(-index))*deltaE) for i in range(len(X)-1)]
	
	x=np.degrees(np.arcsin(np.linspace(np.sin(np.radians(-6)),1.,num=len(Aeff_theta))))
	Aeff_theta_interpolated = interp1d(x, Aeff_theta)
	
	return Aeff_theta_interpolated


def sample_in_galactic_plane(NSou, llimits=[28.43659357,217.42740643]):
	###sample events in the galatic plane. llimits is given in range [0,360]. return values in equatorial coordinates
    b=np.random.normal(0,1,NSou)
    l=np.random.uniform(llimits[0], llimits[1], NSou)
    resvec=[ga2equ([l[i],b[i]]) for i in range(NSou)]

    return resvec
    
def ga2equ(ga):
    """
    Convert Galactic to Equatorial coordinates (J2000.0)
    
    Input: [l,b] in decimal degrees
    Returns: [ra,dec] in decimal degrees
    
    Source: 
    - Book: "Practical astronomy with your calculator" (Peter Duffett-Smith)
    - Wikipedia "Galactic coordinates"
    
    Tests (examples given on the Wikipedia page):
    >>> ga2equ([0.0, 0.0]).round(3)
    array([ 266.405,  -28.936])
    >>> ga2equ([359.9443056, -0.0461944444]).round(3)
    array([ 266.417,  -29.008])
    """
    l = np.radians(ga[0])
    b = np.radians(ga[1])

    # North galactic pole (J2000) -- according to Wikipedia
    pole_ra = np.radians(192.859508)
    pole_dec = np.radians(27.128336)
    posangle = np.radians(122.932-90.0)
    
    # North galactic pole (B1950)
    #pole_ra = radians(192.25)
    #pole_dec = radians(27.4)
    #posangle = radians(123.0-90.0)
    
    ra = m.atan2( (m.cos(b)*m.cos(l-posangle)), (m.sin(b)*m.cos(pole_dec) - m.cos(b)*m.sin(pole_dec)*m.sin(l-posangle)) ) + pole_ra
    dec = m.asin( m.cos(b)*m.cos(pole_dec)*m.sin(l-posangle) + m.sin(b)*m.sin(pole_dec) )
    
    return np.array([np.degrees(ra), np.degrees(dec)])


def sample_from_circle_on_sphere(mu,angle,theta,phi): ## All Angles in degrees
    distance=np.sqrt(np.random.uniform(0, (np.pi/180.*angle)**2 , mu))
    #print np.degrees(distance)
    rotangle=np.random.uniform(0,2*np.pi,mu)
    rot_pos=[]
    for k,i in enumerate(distance):
        rat=np.radians(phi)
        rar=np.radians(phi)
        dect=np.radians(theta)
        decr=np.radians(theta+i/(2.*np.pi)*360)
    
        rot_pos.append(rotate_to_valid_angle(rat,dect,rar,decr,rotangle[k]))
    ra_ret=[x[0] for x in rot_pos]
    dec_ret=[x[1] for x in rot_pos]
    return ra_ret, dec_ret


def sample_from_gaussian_on_sphere(mu,angle,theta,phi): ## Ra, Dec Values from Gaussian on a Sphere --- Used for extended Sources
    distance=np.random.normal(0, np.pi/180.*angle , mu)
    rotangle=np.random.uniform(0,np.pi,mu)*np.sign(distance)+np.pi
    distance=np.fabs(distance)
    rot_pos=[]
    for k,i in enumerate(distance):
        rat=np.radians(phi)
        rar=np.radians(phi)
        dect=np.radians(theta)
        decr=np.radians(theta+i/(2.*np.pi)*360)
    
        rot_pos.append(rotate_to_valid_angle(rat,dect,rar,decr,rotangle[k]))
    ra_ret=[x[0] for x in rot_pos]
    dec_ret=[x[1] for x in rot_pos]
    return ra_ret, dec_ret
    
        
def plot_BG_dist_quantiles(popt, quantiles=[0.9987,0.999997]):        
	dist=init_sens_function(popt)
	norm=dist(np.inf)
	val=[]
	for i in quantiles:
		val.append(fsolve(lambda x: dist(x)/norm-i, 0.001))
	return val
	

def smooth_array(x,y):
	xval=[]
	yval=[]
	val_arr=remove_doubles_from_list(x)
	for i, val in enumerate(val_arr):
		yval.append(np.mean(y[np.where(x==val)[0]]))
		xval.append(val)
	return xval, yval

def dNdmu_discret(x, MU_SOURCES, N_SOURCES):
	width=MU_SOURCES[0]/1000
	ret=0.
	for j,i in enumerate(MU_SOURCES):
		if i-width/2<x and x<i+width/2:
			ret=N_SOURCES[j]/width
			break
	return ret

def calc_nneutrinos_conv_fac(Gamma):
	DETECTOR=LoadDETECTOR("1111")
	ndict= np.load(localPath+"MCode/data/NumberNeutrinos"+str(Gamma)+".npy")[DETECTOR]
	NUMBER_Astro_Events = [float(ndict[i][0]) for i in DETECTOR]
	nSig = [sum(NUMBER_Astro_Events)]
	fac=np.float64(nSig[0])/np.float64(NUMBER_Astro_Events[0])
	return fac
	
def eval_discret_model(MU_SOURCES, N_SOURCES, tenpercentquantile=True, spline=True):
	DETECTOR=LoadDETECTOR("1111")
	Gamma, Project=get_current_project()
	ndict= np.load(localPath+"MCode/data/NumberNeutrinos"+str(Gamma)+".npy")[DETECTOR]
	NUMBER_Astro_Events = [float(ndict[i][0]) for i in DETECTOR]
	nSig = [sum(NUMBER_Astro_Events)]
	fac=np.float64(nSig[0])/np.float64(NUMBER_Astro_Events[0])
	MUS=np.array(copy.copy(MU_SOURCES))*fac
	width=MUS[0]/1000
	fitfunc= "gaussexpconvolv"
	#print Gamma
	#print Project
	if spline==True:
		if tenpercentquantile==True:
		    func_dSigmadN=pickle.load(open("dsigma_dn_spline_10pquantile"+Project, "rb"))
		else:
		    func_dSigmadN=pickle.load(open("dsigma_dn_spline"+Project, "rb")) 	
	else:	
		if tenpercentquantile==True:
			dsigma_dn_fit_params = pickle.load(open("dsigma_dn_fit_results10pquantile", "rb"))
		else:
			dsigma_dn_fit_params = pickle.load(open("dsigma_dn_fit_results", "rb"))    
		func_dSigmadN=lambda x:dsigma_dn_fit_params[Project][0][0][1]*x**dsigma_dn_fit_params[Project][0][0][0]
	print func_dSigmadN(2.)
	y=0.
	for j,i in enumerate(MUS):
	    y+=quad(lambda x:func_dSigmadN(x)*dNdmu_discret(x, MUS, N_SOURCES), i-width, i+width)[0]
	if tenpercentquantile==True:
	    filepath=localPath+"MCode/"
	    dict=np.load(filepath+"unsmeared_significances_1Samples_delta_MCgen_useE_"+"1111"+"_E"+str(Gamma)+"_useDiff_zs_normL1000_"+fitfunc+".dict")
	    y=y+float(dict["Mu5.0"]["NSou0"]["effCl"][2])
	print "Significance:{}".format(y)
	return y



def calc_posmu_from_Aeff_ratio(theta, theta_source, N_SOURCES, MU_SOURCES, NSIG_EVENTS, samples,  DETECTOR, GAMMA, prePath):
	print "------------------ For N:{} and MU:{} calc the source strength distribution ..... -------------------".format(N_SOURCES, MU_SOURCES)
	Aeffratios=dict() 
	thetaEffArea=calcEffArea()
	GAMMA, _=get_best_fit_values()
	for l in DETECTOR:
		thetaEffArea.add_sample(l, samples[l][0])
	for l in range(len(theta_source)):	
		Aeffratios[l]=thetaEffArea.powerlaw_weights(theta_source[l]-(np.pi/2), gamma=GAMMA) 
	pkl_file = open(prePath+DETECTOR[0]+"/sim/gamma"+str(GAMMA)+"/acceptanceSpline.pkl", 'rb')
	acc_spline = pickle.load(pkl_file) 
	actualmuSim = []
	R = 0.
	M = 0.
	for t in theta:
		R += acc_spline(t)      # get the expected mean source strength before acceptance  checks = "real" source strength 
			##~ print "R="+str(R)
	if N_SOURCES>0:
		M = MU_SOURCES*N_SOURCES*1./R 		#		M = mu[i]*self.slope*N_sources[i]*1./R
	for t in theta:
		actualmuSim.append(acc_spline(t)*M)    ### calculate simulation values
	
	muSim=dict()
	muSim[DETECTOR[0]]=actualmuSim
	allyear_MU_SOURCES=[]
	
	for temp,year in enumerate(DETECTOR):
		tempar=[]
		allyear_MU_SOURCES.append(MU_SOURCES*NSIG_EVENTS[temp]/NSIG_EVENTS[0])
		for sp in range(len(theta_source)): 
			tempar.append(actualmuSim[sp]*((Aeffratios[sp][temp][0][0]*samples[year][2])/(Aeffratios[sp][0][0][0]*samples[DETECTOR[0]][2])))
		muSim[year]=tempar

#	for key in muSim:
#		print "Simulated Events for Sample "+key
#		print muSim[key]
	print "--"*20
	return muSim, allyear_MU_SOURCES


def init_sens_function(popt):
	if len(popt)==5:
		sigma=popt[0]
		mu  =popt[1]
		A   =popt[2]
		C    =popt[3]
		cut  =popt[4]
		func=lambda x: evalgaussexp(x,sigma,mu,A,C,cut)
		return func
	if len(popt)==4:
		A=popt[0]
		sigma  =popt[1]
		tau   =popt[2]
		x0    =popt[3]	
		vfunc=np.vectorize(gaussexpconvolv)
		func= lambda x:  integrate.quad(lambda y:vfunc(y,A,sigma,tau,x0) , -0.001, x, epsabs=1e-10, epsrel=1e-10, limit=5000)[0]
		return func


def evalgaussexp(x, sigma, mu,A,C,cut):
	if x>cut:
    		return np.sqrt(np.pi/2)*sigma*A*(1+erf((cut-mu)/np.sqrt(2*sigma**2)))+1/C*A*np.exp(-1/2*((cut-mu)/sigma)**2)*(1-np.exp(-C*(x-cut)))
	else:
		return  np.sqrt(np.pi/2)*sigma*A*(1+erf((x-mu)/np.sqrt(2*sigma**2)))

def gaussexpconvolv(x,A,sigma,tau,x0):
    return A*np.exp(sigma**2/(2*tau**2))*np.exp(-(x-x0)/(tau))*(1+erf(1/np.sqrt(2)*((x-x0)/sigma-sigma/tau)))  #A*np.exp(-(x-x0)/(tau))*(1+erf((x-x0)/np.sqrt(2*sigma**2)))*np.exp(sigma**2/(2*tau**2))

def integralfunc(integrand, x0, x1):
    integral= integrate.quad(integrand, x0, x1,epsabs=1e-10, epsrel=1e-10, limit=5000)[0]
    return integral

def gaussexpfunc(x,sigma,mu,A,C,cut):
	ret=[]
	if type(x)==float:
		x=[x]
	cutv=cut
	for s in np.array(x):
		if s<cutv:
			ret.append(A*np.exp(-1/2*((s-mu)/sigma)**2))
		else:
			ret.append(A*np.exp(-1/2*((cutv-mu)/sigma)**2)*np.exp(-C*(s-cutv)))
	if len(ret)==1:
		return ret[0]
	else:
		return ret
		
def expfunc(x,mu,A,C):
	ret=[]
	if type(x)==float:
		x=[x]
	for s in np.array(x):
		ret.append(A*np.exp(-C*(s-mu)))
		
	if len(ret)==1:
		return ret[0]
	else:
		return ret


def readAeff(fileAeff):
        hi, ed = np.genfromtxt(fileAeff, unpack=False)
        histAeff=hi
        edgesAeff=ed
        SquareMetersToSquareCm = 100*100*1.0
        histAeff = SquareMetersToSquareCm*np.array(histAeff)
        return interpolate.InterpolatedUnivariateSpline(edgesAeff, histAeff, k=3)


def NumberofSimulatedEvents(Gamma, Detector, ind_mode=True):
	if ind_mode:
		DETECTOR=LoadDETECTOR(Detector)
	else:
		DETECTOR=[Detector]
	ndict= np.load(localPath+"MCode/data/NumberNeutrinos"+str(Gamma)+".npy")[()]
	nEvents = [int(ndict[i][2]) for i in DETECTOR]
	return sum(nEvents)
	
def NumberofSigEvents(Gamma, Detector):
	DETECTOR=LoadDETECTOR(Detector)
	ndict= np.load(localPath+"MCode/data/NumberNeutrinos"+str(Gamma)+".npy")[()]
	nEvents = [int(ndict[i][0]) for i in DETECTOR]
	return sum(nEvents)
	
def getstrongestsources(NVSS, GalPlane=True, OnlyGalPlane=False, N=100):
	
	flux_pre=np.array(NVSS["Flux"])
	zenith_pre=np.array(NVSS["Dec(2000)"]) # in degrees
	ra_pre=np.array(NVSS["RA(2000)"])			# in degrees
	flux=[]
	zenith=[]
	ra=[]

	for i in range(len(zenith_pre)):
		galcoord=galactic(ra_pre[i]*np.pi/180.,zenith_pre[i]*np.pi/180.)[1]
		if (zenith_pre[i]>np.radians(-5)):
			if ((galcoord>np.radians(5) or galcoord<np.radians(-5) or GalPlane==True) and OnlyGalPlane==False):
				flux.append(flux_pre[i])
				zenith.append(zenith_pre[i])
				ra.append(ra_pre[i])
			elif (OnlyGalPlane==True and abs(galcoord)<np.radians(5)):
				flux.append(flux_pre[i])
				zenith.append(zenith_pre[i])
				ra.append(ra_pre[i])
				
	ind = np.argpartition(flux, -N)[-N:]
	Azimuth=np.radians(ra)[ind]
	Zenith=dec2zen(np.radians(zenith)[ind])
	return Zenith, Azimuth


def getweakestsources(NVSS, N=100):
    minval=np.min(NVSS["Flux"])*5
    inds = np.where(NVSS["Flux"]<minval)
    ind=np.random.choice(inds[0], N)
    Flux=np.array(NVSS["Flux"])[ind]
    Azimuth=np.radians(np.array(NVSS["RA(2000)"])[ind])
    Zenith=dec2zen(np.radians(np.array(NVSS["Dec(2000)"])[ind]))
    return Zenith, Azimuth

def HMS2deg(input_data ,mode="RA"):
    RA, DEC, rs, ds = '', '', 1, 1
    if mode=="Dec":
        D, M, S = [float(i) for i in input_data]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        deg = D + (M/60) + (S/3600)
        DEC = '{0}'.format(deg*ds)
   
    elif mode=="RA":
        H, M, S = [float(i) for i in input_data]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        deg = (H*15) + (M/4) + (S/240)
        RA = '{0}'.format(deg*rs)
    else:
        print " KEY ERROR"
    if RA=='':
        return float(DEC)
    else:
        return float(RA)

def readNVSS(skip_line=16):
	a = open(catalogPath+'NVSS/FullNVSSCat.text', 'r')
	field_names=["RA(2000)", "Dec(2000)", "Flux"]
	subnames=["h m s","d m s", "mJy"]
	NVSS=dict()
	
	temp_arrays=[]
	for i in field_names:
	    temp_arrays.append([])
	    
	##########Create Index Mask##################
	index_mask=[]
	for i in range(skip_line):
	    a.readline()
	subname_line=a.readline().split()
	for subs in subnames:
	    temp=subs.split()
	    for x in temp:
	        i=0
	        while i<len(subname_line):
	            if subname_line[i]==x:
	                index_mask.append(i)
	                subname_line[i]=' '
	                break
	            i=i+1              
	
	####Read Data from File and Convert to Degrees
	for line in a:
	    if field_names[0] in line or "h" in line or "page" in line :
	        continue
	    if len(line.split())<14:
	        continue
	    curind=0
	    temp=line.split()
	    for i,subs in enumerate(subnames):
	        if len(subs.split())==3:
	            if "RA" in field_names[i]:
	                temp_arrays[i].append(HMS2deg([temp[index_mask[curind]],temp[index_mask[curind+1]],temp[index_mask[curind+2]]], mode="RA"))
	            elif "Dec" in field_names[i]:
	                Dec=HMS2deg([temp[index_mask[curind]],temp[index_mask[curind+1]],temp[index_mask[curind+2]]], mode="Dec")
			if Dec>-5:
				temp_arrays[i].append(Dec)
			else:
				j=0
				while j<i:
					del temp_arrays[j][-1]
					j+=1
	            else:
	                print "Key Error for transformation from HMS to DEG"
	            curind+=3
	        if len(subs.split())==1:
			if len(temp_arrays[i])==len(temp_arrays[i-1]): continue            
			if temp[index_mask[curind]][0]!="<":
	                	temp_arrays[i].append(float(temp[index_mask[curind]]))
			else:
	                	j=0
	                	while j<i:
	                   		del temp_arrays[j][-1]
	                    		j+=1	         
	#######Write into the dictionary
	for i in range(len(temp_arrays)):
	    NVSS[field_names[i]]=temp_arrays[i]
	
	return NVSS

def createNVSSSkymap(NVSS, npix, UseGalPlane=True, OnlyGalPlane=False, Norm=False, AeffCorrection=False ,nEvents=1):
	map_delta = np.zeros(npix)
	declination_pre=np.array(NVSS["Dec(2000)"])*np.pi/180.
	ra_pre=np.array(NVSS["RA(2000)"])*np.pi/180.
	declination=[]
	ra=[]

	if(AeffCorrection):
		Aeff_interpolated=getEffAreaVsDec()		# Effective area function(Dec angle in deg)
												# 						in equatorial coord.

	for i in range(len(declination_pre)):
		galcoord=galactic(ra_pre[i],declination_pre[i])[1]
		if (declination_pre[i]>np.radians(-5)):
			if ((galcoord>np.radians(5) or galcoord<np.radians(-5) or UseGalPlane==True) and OnlyGalPlane==False):
				declination.append(declination_pre[i])
				ra.append(ra_pre[i])
			elif (OnlyGalPlane==True and abs(galcoord)<np.radians(5)):
				declination.append(declination_pre[i])
				ra.append(ra_pre[i])
	declination=np.array(declination)
	ra=np.array(ra)
	
	hit_bool = H.ang2pix(nside,dec2zen_noticecube(declination),ra)
	for i in range(len(hit_bool)):
			if(AeffCorrection):
				map_delta[hit_bool[i]] += NVSS["Flux"][i]*Aeff_interpolated(declination[i])
			else:
				map_delta[hit_bool[i]] += NVSS["Flux"][i]

	# (Re-)normalizaion:
	if Norm:
		map_delta=renormSkymap(map_delta, nEvents)
	else:
		map_delta=adjustUnitsSkymap(map_delta)
	return map_delta


def remove_doubles_from_list(x):
	return [k for i,k in enumerate(x) if k not in x[:i]]


def sort_lists(*lists):
    list0_sort=sorted(lists[0])
    indices=[lists[0].index(k) for k in list0_sort]
    return_list=[]
    return_list.append(list0_sort)
    for j in range(1,len(lists)):
        return_list.append([lists[j][i] for i in indices])
    return return_list


def double_sort_list(listA, listB):
	listA_sort=sorted(listA)
	indices=[listA.index(k) for k in listA_sort]
	listB=[listB[i] for i in indices]
	return listA_sort, listB, indices


####Function for calculating the best fit Phi for a given Gamma from Sebastians best Fit Data
### Warning: Interpolation between Grid Points not implemented yet
def calc_flux_norm_from_grid(GAMMA, fit_range=[]):
	path=localPath+"MCode/scan_astro_gamma_astro_IC59_IC79_IC86-2011_IC86-2012-13-14_wIce_align_new_fit_final_results"
	dct=np.loadtxt(path, skiprows=1)
	gammas=[dct[i][10] for i in range(len(dct))]
	phis=[dct[i][7] for i in range(len(dct))]
	LLH=[dct[i][14] for i in range(len(dct))]
	if len(fit_range)==2:
		LLH_scan=[-dct[i][14] for i in range(len(dct)) if dct[i][10]==GAMMA and dct[i][7]<fit_range[1] and dct[i][7]>fit_range[0]]
		Phizeroscan=[dct[i][7] for i in range(len(dct)) if dct[i][10]==GAMMA and dct[i][7]<fit_range[1] and dct[i][7]>fit_range[0]]
	else:
		LLH_scan=[-dct[i][14] for i in range(len(dct)) if dct[i][10]==GAMMA]
		Phizeroscan=[dct[i][7] for i in range(len(dct)) if dct[i][10]==GAMMA]		
	popt, pcov = curve_fit(lambda x,a,b,c: a*(x-b)**2+c, np.array(Phizeroscan), np.array(LLH_scan), p0=[5,0.61,2000])
	return popt[1]
	#plt.scatter(Phizeroscan, LLH_scan)
	#x=np.linspace(0.0,3.0, 100)
	#y=[func(i,popt[0],popt[1], popt[2]) for i in x]
	#plt.plot(x, y)
	#plt.show()
	
### Calculate Conversion Factor for Phi0 for a given Gamma ("newGamma") assuming the sum of the weights (amount astrophysical neutrinos with energy over 200TeV) beeing constant
def convert_flux_norm(newGamma):
	datapath     = localPath+"DATA/"
	if not os.path.exists(datapath+"Fluxnorm.npy"):
		norm_dict=dict()
		GAMMA,Phi=get_best_fit_values()
		DETECTOR=LoadDETECTOR("1111")
		x=0.
		y=0.
		for det in DETECTOR:
			mc, exp, config = simulationLoad(det, datapath, GAMMA)
			mask=[mc["trueE"]>200000]  ##Cut on Events with trueE higher than 200TeV
			astro=mc["astro"][mask]
			energy=mc["trueE"][mask]
			x+=sum(astro)
			y+=sum(1/Phi*astro*(energy/(100*10**3))**(GAMMA-newGamma))
		norm_dict[newGamma]=x/y
		print("Calculated Fluxnorm....{}".format(x/y))
		with open(datapath+"Fluxnorm.npy", "w+b") as f:
			pickle.dump(norm_dict, f)
		return x/y
	else:
		with open(datapath+"Fluxnorm.npy", "rb") as f:
			norm_dict=pickle.load(f)
		if newGamma in norm_dict:
			print("Read Fluxnorm from File")
			return norm_dict[newGamma]
		else:
			GAMMA,Phi=get_best_fit_values()
			DETECTOR=LoadDETECTOR("1111")
			x=0.
			y=0.
			for det in DETECTOR:
				mc, exp, config = simulationLoad(det, datapath, GAMMA)
				mask=[mc["trueE"]>200000]  ##Cut on Events with trueE higher than 200TeV
				astro=mc["astro"][mask]
				energy=mc["trueE"][mask]
				x+=sum(astro)
				y+=sum(1/Phi*astro*(energy/(100*10**3))**(GAMMA-newGamma))
			norm_dict[newGamma]=x/y
			print("Calculated Fluxnorm....{}".format(x/y))
			with open(datapath+"Fluxnorm.npy", "w+b") as f:
				pickle.dump(norm_dict, f)
			return x/y

###Get Information about the 'current' Project as defined in the config.ini
def get_current_project():
	globalsettings=ConfigParser.ConfigParser()
	globalsettings.read('config.ini')
	gamma   =  globalsettings.getfloat('curProject', 'Gamma')
	name    =  globalsettings.get('curProject', 'Name')
	return gamma, name


###Get Last best fit values as defined in the config.ini, Phi0 is just the prefactor without factors of 10^x
def get_best_fit_values():
	globalsettings=ConfigParser.ConfigParser()
	globalsettings.read('config.ini')
	gamma   =  globalsettings.getfloat('Basics', 'Gamma')
	phi_zero=globalsettings.getfloat('Basics', 'Phi0')
	return gamma, phi_zero

##calculate new flux weights for new (gamma, phi) from best fit MC
def convert_flux_weight(weights, energys, gamma_star, phi_zero_star):  ### set phi zero and gamma to best fit values
	gamma,phi_zero=get_best_fit_values()
	return phi_zero_star/phi_zero*weights*(energys/(100*10**3))**(gamma-gamma_star)

def mu_convert_to_all_years(x, ndict, DETECTOR):
	return x*np.sum([int(ndict[i][0]) for i in DETECTOR])/ndict[DETECTOR[0]][0]

#####Load Datasample
def LoadDETECTOR(DETECTOR_settings):
	DETECTOR=[]
	for i in range(len(DETECTOR_settings)):
		if DETECTOR_settings[i]=="1":
			if i==3:
				DETECTOR.append("IC59")
			if i==2:
				DETECTOR.append("IC79")
			if i==1:
				DETECTOR.append("IC86_11")
			if i==0:
				DETECTOR.append("IC86_12_13_14")
	return DETECTOR


def Livetimes(DETECTOR, datapath=localPath+"DATA/"):
	config = ConfigParser.ConfigParser()
	livetimes=[]
	for i in DETECTOR:
		if i=="IC59":
			config.read(datapath + "IC59_PS_Rene.cfg")
		elif i=="IC79":
			config.read(datapath + "IC79_PS_Rene.cfg")
		elif i=="IC86_11": 
			config.read(datapath + "IC86-2011_PS_Rene.cfg")
		elif i=="IC86_12_13_14":
			config.read(datapath + "IC86-2012-13-14_wIce_align_new.cfg")
		livetimes.append(float(config.get("fit_settings", "livetime")))
	return livetimes
			
	
	
def Load(dataset, datapath, Gamma_new):
	
	globalsettings=ConfigParser.ConfigParser()
	globalsettings.read('config.ini')
	cut_angle=globalsettings.getfloat('Basics', 'CutAngle')
	spline_angle=globalsettings.getfloat('Basics', 'SplineUpTo')
		
	fileinfo=ConfigParser.ConfigParser()
	if Gamma_new!=-1:
		fileinfo.read(datapath+"gamma"+str(Gamma_new)+".ini")
	else:
		gamma, _ =get_best_fit_values()
		fileinfo.read(datapath+"gamma"+str(gamma)+".ini") 
	print "Read MC: {}".format(datapath + fileinfo.get(dataset,"mc"))
	mc = np.load(datapath + fileinfo.get(dataset,"mc"))	
	
	
	print "For Detector Configuration: " + str(dataset) 
	return mc[mc["trueDec"]>np.radians(spline_angle)] 
	

def simulationLoad(dataset, datapath, Gamma_new=-1, exp_bool=False, splinemode=False, nNeutrinos=False):  ##Set filenames in corresponding .ini file for chosen gamma value
	exp = []
	mc=[]          
	fileinfo=ConfigParser.ConfigParser()
	gamma_bf, phi_zero_bf =get_best_fit_values()
	#if Gamma_new!=-1:
	#	fileinfo.read(datapath+"gamma"+str(Gamma_new)+".ini")
	#else:
	#	fileinfo.read(datapath+"gamma"+str(gammabf)+".ini")
	fileinfo.read(localPath+"DATA/"+"gamma"+str(gamma_bf)+".ini")
	if Gamma_new==-1: Gamma_new=gamma_bf
	
	config = ConfigParser.ConfigParser()
	print localPath+"DATA/"
	config.read(localPath+"DATA/"+fileinfo.get(dataset,"config"))
	livetime = float(config.get("fit_settings", "livetime"))  
	print "For Detector Configuration: " + str(dataset) + " the total uptime is "+ str(livetime)+" s"
		
	#####################Load Files################################################################		
	if exp_bool==False:
		print "Read MC: {}".format(localPath+"DATA/" + fileinfo.get(dataset,"mc"))
		mc = np.load(localPath+"DATA/" + fileinfo.get(dataset,"mc"))	
  		if "trueE" not in mc.dtype.fields:
			mc=  append_fields(mc, "trueE", mc["MCPrimaryEnergy"], dtypes=np.float, usemask=False)	    
		if "ow" not in mc.dtype.fields:
			mc=  append_fields(mc, "ow", mc["orig_OW"], dtypes=np.float, usemask=False)	
		mc = mc[np.where(mc["dec"]>np.radians(-5))]                                    
		mc = mc[["ra", "dec", "logE", "sigma", "trueRa", "trueDec", "trueE", "ow", "conv", "prompt", "astro"]]    
		mc = append_fields(mc, "sinDec", np.sin(mc["dec"]), dtypes=np.float, usemask=False)
		if gamma_bf!=Gamma_new:
			mc["astro"]=convert_flux_weight(mc["astro"], mc["trueE"],Gamma_new,convert_flux_norm(Gamma_new))
		
	if exp_bool==True or nNeutrinos==True:
		print "Read ExpData: {}".format(localPath+"DATA/"+ fileinfo.get(dataset,"exp"))
		exp = np.load(localPath+"DATA/" + fileinfo.get(dataset,"exp"))	
		exp = exp[["ra", "dec", "logE", "sigma"]]  
		exp = exp[np.where(exp["dec"]>np.radians(-5))]    
		exp = append_fields(exp, "sinDec", np.sin(exp["dec"]), dtypes=np.float, usemask=False)
		
		
	if len(mc)>0 and splinemode==False:
		dt = mc.dtype.descr
		for i in range(len(dt)):
			if mc.dtype.names[i]=="conv" or mc.dtype.names[i]=="astro":
				dt[i] = (dt[i][0], "<f8")
			else:
				dt[i] = (dt[i][0], "<f4")       
		mc = mc.astype(dt)
		
	if len(exp)>0:
		dt = exp.dtype.descr
		for i in range(len(dt)):
			dt[i] = (dt[i][0], "<f4")
		exp = exp.astype(dt)
		
	return mc, exp, livetime
	
	
def Reweight(dataset, esample):
	event_data=np.load(localPath+"MCode/data/NumberNeutrinos.npy")[dataset]
	reweightfac=event_data[3]/event_data[2]
	print "Reweighting with factor "+ str(reweightfac)
	return esample*reweightfac
			
            

### PRINT TIME DIFFERENCE TO STARTTIME FROM NOW
def printRunTime(starttime):
	print "Duration of run: "+str(round(tm.time()-starttime,2))+"s"
	return


### NORMS HISTOGRAMM BY DEFINING HIGHEST PEAK AS fac=1
def normHist(hist, fac=1.0):
	maxVal= float(max(hist))
	nHist = []
	for i in range(0,len(hist)):
		nHist.append(float(hist[i]*1.0)/(float(1.0*maxVal)/fac))
	return nHist


### NORMS HISTOGRAMM BY DEFINING HIGHEST PEAK AS fac=1
#~ def rescaleHist(hist, fac=1.0):
	#~ sumVal = sum(hist)
	#~ nHist = []
	#~ for i in range(0,len(hist)):
		#~ nHist.append(float(hist[i]*1.0)/(float(1.0*sumVal)/fac))
#~ 
	#~ return nHist
	
### RETURNS ERROR ON MEAN OF A GIVEN STD. LIST
def errOnMean(err, NMeasure):
	errM =[]
	for i in range(0,len(err)):
		errM.append(err[i]*1.0/np.sqrt(NMeasure))

	return errM

### RETURNS ERROR ON A DIFFERENCE OF TWO MEASURED QUANTITIES
def getDifferenceError(err1, err2):
	nErr = []
	for i in range(0,len(err1)):
		nErr.append(np.sqrt(np.power(err1[i],2)+np.power(err2[i],2)))

	return nErr


### RETURNS abs VALUE FROM EACH ELEMENT OF GIVEN LIST
def absList(thisList):
	nList = []
	for i in range(0, len(thisList)):
		nList.append(np.absolute(thisList[i]))

	return nList
		
		
### RETURNS the phase angle FROM EACH ELEMENT OF GIVEN LIST
def phaseList(thisList):
	nList = []
	for i in range(0, len(thisList)):
		nList.append(np.angle(thisList[i]))

	return nList
	

#### RETURNS BIN NUMBER OF VALUE val FOR BINNING ENDGES bins
def getBinNumber(bins, val):
	for i in range(0,len(bins)):
		if val < bins[i]:
			return (i-1)
	print "ERROR: value out of Histograms Bin Range."
	return 0


#### RETURNS LIST OF N RANDOM VALUES FOLLOWING ARBITRARY DISTR.
def arbDistrRandom(hist, bins, N):

	if(len(bins) > len(hist)):
		nEdges = setNewEdges(bins)
	else:
		nEdges = bins
	
	atmosSpline = interpolate.InterpolatedUnivariateSpline(nEdges, hist)
	i = 0
	ret = []
	while i < N:
		y = np.random.uniform(0., 1.)
		x = np.random.uniform(bins[0],bins[len(bins)-1])
		if (y < atmosSpline(x)):
			ret.append(x)
			i+=1

	return ret



##### GENERATES EVENTS FOR ATM. SPEC.
#def getAtmosNu(N_Atmos, Spline_Zenith, fullSphereMode=False, Milagro=None):
#	i=0
#	Map_List = []
#	if fullSphereMode == True:
#		cut = 1.0
#	else:
#		cut = 0.0
#	if Milagro == None:
#		while i < N_Atmos:
#			cosThetaRan = np.random.uniform(-1.,cut)
#			if np.random.uniform(0,1) < Spline_Zenith((-1.0)*abs(cosThetaRan)):
#				t_ran = np.arccos(cosThetaRan)
#				p_ran = np.random.uniform(0., 2*np.pi)
#				Map_List.append([t_ran,p_ran])
#				i=i+1
#	else:
#		while i < N_Atmos:
#			cosThetaRan = np.random.uniform(-1.,cut)
#			if np.random.uniform(0,1) < Spline_Zenith((-1.0)*abs(cosThetaRan)):
#				t_ran = np.arccos(cosThetaRan)
#				foundAzimuth = False
#				while foundAzimuth == False:
#					p_ran = np.random.uniform(0., 2*np.pi)
#					if np.random.uniform(0.0,1.0) < Milagro(t_ran, p_ran):
#						foundAzimuth = True
#				Map_List.append([t_ran,p_ran])
#				i=i+1
#
#	return Map_List
#

#### RETURNS LIST OF N RANDOM VALUES FOLLOWING ARBITRARY DISTR.
def arbDistrRandomOLD(hist, bins, N):  ### DONT USE IF AVOIDABLE!!! ###
	i = 0
	ret = []
	while i < N:
		y = np.random.uniform(0., 1.)
		x = np.random.uniform(bins[0],bins[len(bins)-1])
		curBin = getBinNumber(bins, x)	
		if (y < hist[curBin]):
			ret.append(x)
			i+=1
	return ret


#### RETURNS LIST OF N RANDOM VALUES 
def equRandom( N):	
	i = 0
	ret = []
	while i < N:
		x = np.random.uniform(-1, 0.)
		ret.append(x)
		i+=1
	return ret


#### CHANING GIVEN DISTR. BY SLIGHTLY DEFORMING IT IN DIFFERENT WAYS (.., V2, V3)
## version 1
def changeDistrPerc(hist, totPercChange):
	singlePerc = totPercChange*1.0/(len(hist)-1)
	halfPerc = totPercChange*1.0/2
	newHist= []

	for i in range(0, len(hist)):
		curPerc = (i*singlePerc)-halfPerc
		newHist.append(hist[i]*(1+curPerc))
	
	return newHist


## version 2
def changeDistrPercV2(hist, totPercChange):
	singlePerc = totPercChange*1.0/(len(hist)-1)
	newHist= []

	for i in range(0, len(hist)):
		curPerc = (i*singlePerc)
		newHist.append(hist[i]*(1+curPerc))
	
	return newHist


## version 3
def changeDistrPercV3(hist, totValueToAdd):
	newHist= []

	for i in range(0, len(hist)):
		newHist.append(hist[i]+totValueToAdd)

	return newHist



#### RETURNS SIGN OF GIVEN NUMBER
def returnSign(number):

	if number < 0:
		return (-1)
	if number > 0:
		return (+1)
	if number == 0:
		#~ print "WARNING: NO SIGN FOR ZERO NUMBER!"
		return 0


#### RETURNS CENTER VALUES OF ALL BINS IN LIST OF EDGES
def retBinCenters(edges):
	ret = []
	for i in range(0, len(edges)-1):
		ret.append((edges[i]+edges[i+1])*1.0/2.0)
	return ret


#### RETURNS EFF. Cl WITHOUT CONSIDERING al0
def retEffCl(alm, lmax, twoLPlus=0):
	curEffCl = []
	#curEffCl.append(1.0) ## !!! APPENDED BECAUSE C0 IS NOT CALCUABLE !!! ###
	if len(alm)==1:	
		alm=alm[0]
		for l in range(1,lmax+1):
			aSum = 0.
			for m in range(-l,l+1):
				if m != 0:
					val = alm[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
					aSum += 1.0/(2.0*l+twoLPlus)*abs(val)*abs(val)
			curEffCl.append(aSum)
	elif len(alm)==2:
		alm0=alm[0]
		alm1=alm[1]
		for l in range(1,lmax+1):
			aSum = 0.
			for m in range(-l,l+1):
				if m != 0:
					val1 = alm1[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
					val0 = alm0[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
					aSum += 1.0/(2.0*l+twoLPlus)*abs(val0)*abs(val1)
			curEffCl.append(aSum)
	return np.array(curEffCl)

#### STANDARD CALC FOR Cl
def retCl(alm, lmax):
	curCl = []

	for l in range(0,lmax+1):
		aSum = 0.
		for m in range(-l,l+1):
			val = alm[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
			aSum += 1.0/(2.0*l+1)*abs(val)*abs(val)
		curCl.append(aSum)
	
	return curCl

#### STANDARD CALC FOR Cl from cross-spectrum:
def retCrossCl(alm, blm, lmax, USE_ZERO=True):
	curCl = []

	for l in range(0,lmax+1):
		aSum = 0.
		for m in range(-l,l+1):
			if (USE_ZERO or m!=0):
				val1 = alm[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
				val2 = blm[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
				aSum += 1.0/(2.0*l+1.0)*np.conj(val1)*val2
		curCl.append(aSum)	
	return curCl
	
#### STANDARD CALC FOR Cl from cross-spectrum TEST FUNCTION:
def retCrossClTest(alm, blm, lmax, USE_ZERO=True):
	curCl = []

	for l in range(0,lmax+1):
		aSum = 0.
		for m in range(-l,l+1):
			if (USE_ZERO or m!=0):
				val1 = alm[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
				val2 = blm[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
				aSum += 1.0/(2.0*l+1.0)*np.abs(np.conj(val1)*val2)	### gives the abs value
		curCl.append(aSum)	
	return curCl

#### STANDARD CALC FOR Cl_log
def calcClLogFromAlm(alm_all, lmax):
	curClLog = []
	
	for l in range(1,lmax+1):
		aSum = 0.
		for m in range(-l,l+1):
			if m != 0:
				val = alm_all[H.sphtfunc.Alm.getidx(lmax, l, abs(m))]
				aSum += 1.0/(2.0*(l))*np.log10(abs(val))
		curClLog.append(aSum)
	
	return curClLog
	


##### GENERATES A SINGLE POINT SOURCE
def getOnePSAnywhere(Mu, Spline_Pointspread,Lifetime=1.):
	Map_List = []
	#Position der Quelle Isotrop auf der Kugel
	theta = np.arccos(np.random.uniform(-1.,1.))
	phi = np.random.uniform(0., 2*np.pi) #0.

	NeutrinosPerSource = Mu
	ran_phi   = np.random.uniform(0, 2*np.pi,NeutrinosPerSource)
	ran_theta = get_ran_theta(NeutrinosPerSource ,Spline_Pointspread)*np.pi/180.

	for ran_t,ran_p in zip(ran_theta,ran_phi):
		z=(np.cos(theta)*np.cos(ran_t))-(np.sin(theta)*np.sin(ran_t)*np.sin(ran_p))
		x=0
		y=0
		x =-(np.sin(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.cos(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.cos(phi)*np.sin(theta)*np.cos(ran_t))
		y = (np.cos(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.sin(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.sin(phi)*np.sin(theta)*np.cos(ran_t))
		fill_t = np.arccos(z)
		fill_p = np.arctan2(y,x)
		if (fill_p < 0):
			ran_theta = get_ran_theta(NeutrinosPerSource ,Spline_Pointspread)*np.pi/180.

	for ran_t,ran_p in zip(ran_theta,ran_phi):
		z=(np.cos(theta)*np.cos(ran_t))-(np.sin(theta)*np.sin(ran_t)*np.sin(ran_p))
		x=0
		y=0
		x =-(np.sin(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.cos(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.cos(phi)*np.sin(theta)*np.cos(ran_t))
		y = (np.cos(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.sin(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.sin(phi)*np.sin(theta)*np.cos(ran_t))
		fill_t = np.arccos(z)
		fill_p = np.arctan2(y,x)
		if (fill_p < 0):
			fill_p = 2*np.pi+fill_p
		Map_List.append([fill_t,fill_p])

	retPhis = [i[1] for i in Map_List]
	retThetas = [i[0] for i in Map_List]
	return np.array(retThetas), np.array(retPhis)


##### CALC. ANGLE BETWEEN TWO VECTORS
def getAngleBetween(theta1, phi1, theta2, phi2):
	scalarPr = np.sin(theta1)*np.sin(phi1)*np.sin(theta2)*np.sin(phi2)+ np.sin(theta1)*np.cos(phi1)*np.sin(theta2)*np.cos(phi2) + np.cos(theta1)*np.cos(theta2)
	angle = np.arccos(scalarPr)
	return angle


##### CALC. ANGLE BETWEEN TWO VECTORS
def getQuickAngleBetween(cosT1,sinT1, cosP1, sinP1, cosT2, sinT2, cosP2, sinP2):
	scalarPr = sinT1*sinP1*sinT2*sinP2+sinT1*cosP1*sinT2*cosP2+cosT1*cosT2
	return scalarPr


##### CALC. NEW EDGES BY CENTER OF FORMER ONES (N=> N-1 edges)
def setNewEdges(edges):
	newEdges = []
	for i in range(0,len(edges)-1):
		newVal = (edges[i]+edges[i+1])*1.0/2
		newEdges.append(newVal)
	return newEdges

##### CALC. NEW EDGES BY CENTER OF FORMER ONES (N=> N-1 edges)
def setOldEdges(edges):
	newEdges = []
	for i in range(0,len(edges)-1):
		newVal = (edges[i]+edges[i+1])*1.0/2
		newEdges.append(newVal)
	oldEdges = [edges[0]-(edges[1]-edges[0])*1.0/2]+newEdges+[edges[-1]+(edges[-1]-edges[-2])*1.0/2]
	return oldEdges


##### INVERTS 2D ARRAY 
def invertArray(arr):
	#newArr = [[0]*len(arr)]*len(arr[0])
	#print newArr
	#for i in range(0,len(arr)):
	#	for k in range(0,len(arr[i])):
	#		print str(i)+" "+str(k)+" "+str(arr[i][k])
	#		newArr[k][i] = arr[i][k]
	newArr = zip(*arr[::-1])
	return newArr

#### PLOTS A RANDOM SPLINE 
def plotSpline(spl, xmin, xmax, col="r", lab="", linestyle="--"):
	xSpline = np.linspace(xmin, xmax, 10000)
	ySpline = spl(xSpline)

	print np.array(xSpline)
	print np.array(ySpline)
	print linestyle
	print col
    
	#plt.figure()
	plt.plot(xSpline, ySpline, linestyle=linestyle, color=col, label=lab, linewidth=2.0)
	plt.title("Spline")
	plt.xlabel("a.u.")
	plt.ylabel("rel. prob.")
	plt.show()

### PLOTS almWeights AND effAlmWeights
def getAlmWeightPlane(almWeights, lmax):
	plot_arr = []

	for l in range(0,lmax):
		plot_arr.append([])
		for m in range(0, lmax):
			if m < l+1:
				plot_arr[l].append(almWeights[H.sphtfunc.Alm.getidx(lmax, l,abs(m))])
			else:
				plot_arr[l].append(0.0)
	
	return plot_arr

	
### GETS Alm Plane FROM list
def getAlmPlane(almList, lmax, showLZero=True):
	plot_arr = []
	for l in range(0,lmax):
		plot_arr.append([])
		for m in range(0, lmax):
			if m < l+1:
				if m == 0 and showLZero == False:
					plot_arr[l].append(0.0)
				else:
					plot_arr[l].append(almList[H.sphtfunc.Alm.getidx(lmax, l,abs(m))])
			else:
				plot_arr[l].append(0.0)
	return plot_arr


### GETS Alm Plane FROM list
def getAlmSignPlane(deviations, weights,lmax):
	
	return 0
	

### GETS Al0 List FROM list
def getAl0List(thisList, lmax):
	nList = []
	for l in range(0, lmax+1):
		nList.append(thisList[H.sphtfunc.Alm.getidx(lmax, l,0)])
	return nList


### GETS Al0 Difference-List FROM list
def getAl0DiffList(sig_almList, almList, lmax, errAlmList=[]):
	diff = []
	for l in range(0, lmax+1):
		if len(errAlmList) >1:
			diff.append((sig_almList[H.sphtfunc.Alm.getidx(lmax, l,0)]-almList[H.sphtfunc.Alm.getidx(lmax, l,0)])*1.0/errAlmList[H.sphtfunc.Alm.getidx(lmax, l,0)])
		else:
			diff.append((sig_almList[H.sphtfunc.Alm.getidx(lmax, l,0)]-almList[H.sphtfunc.Alm.getidx(lmax, l,0)])*1.0)

	return diff


### GETS Alm Norm Difference-List FROM list
def getAl0NormDiffList(sig_almList, almList, errAlmList=[]):
	return getAl0DiffList(absList(sig_almList), absList(almList), absList(errAlmList))


### GETS Alm Difference-Plane FROM list
def getAlmDiffPlane(sig_almList, almList, errAlmList=[], showLZero=True):
	diff = []
	for i in range(0, len(almList)):
		if len(errAlmList) >1:
			diff.append((sig_almList[i]-almList[i])*1.0/errAlmList[i])
		else:
			diff.append((sig_almList[i]-almList[i])*1.0)

	diffPlane = getAlmPlane(diff, showLZero)
	return diffPlane


### GETS Alm Norm Difference-Plane FROM list
def getAlmNormDiffPlane(sig_almList, almList, errAlmList=[], showLZero=True):
	return getAlmDiffPlane(absList(sig_almList), absList(almList), absList(errAlmList), showLZero)


#### CALC. weights for Alm #####
def calcAlmWeights(alm_original, alm_new, alm_original_errors, alm_new_errors):
	w = []
	for i in range(0, len(alm_original)):
		w.append((abs(alm_new[i])-abs(alm_original[i]))*1.0/np.sqrt(np.power(abs(alm_original_errors[i]),2)+np.power(abs(alm_new_errors[i]),2)))

	return w


#### CALC. almD2 from Alm #####
def calcD2Alm(alm_original, alm_new, alm_original_errors, weights,norm=False):
	thisDSquared = 0
	for i in range(1, len(alm_original)):
		thisDSquared += returnSign(abs(alm_new[i])-abs(alm_original[i]))*np.power(abs(alm_original[i])-abs(alm_new[i]),2)*1.0/np.power(abs(alm_original_errors[i]),2)*weights[i]
	if norm == True:
		thisDSquared = thisDSquared*1.0/sum(absList(weights))
	return thisDSquared


#### CALC. D2 #####
def calcD2(cl_new, cl_original,  cl_original_errors, weights, lmax, norm=True, limits=[]):
	if len(limits)==0:
		curDSquared = 0
		for i in range(1,lmax+1):
			curDSquared += 1.0/(lmax)*np.power(cl_new[i]-cl_original[i],2)*1.0/(np.power(cl_original_errors[i],2))*weights[i]*returnSign(cl_new[i]-cl_original[i])
		if norm == True:
			curDSquared = curDSquared*1.0/sumFromTo(absList(weights), 0, lmax+1)
		return curDSquared
	elif len(limits) == 2:
		curDSquared = 0
		effLMAX = 1+limits[1]-limits[0]
		for i in range(limits[0],limits[1]+1):
			curDSquared += 1.0/(effLMAX+1)*np.power(cl_new[i]-cl_original[i],2)*1.0/(np.power(cl_original_errors[i],2))*weights[i]*returnSign(cl_new[i]-cl_original[i])
		if norm == True:
			curDSquared = curDSquared*1.0/sumFromTo(absList(weights), limits[0], limits[1]+1)
		return curDSquared	
	else:
		print "ML CALCULATION ERROR FOR D2: Calculation limits given, but of wrong structure."


#### CALC. Chi2 #####
def calcChi2(sig, org, err, limits=[]):
	if len(limits)==0:
		curChi = 0
		for i in range(0,len(org)):
			curChi += np.power(sig[i]-org[i],2)*1.0/(np.power(err[i],2))
		#print "curChi: "+str(curChi)
		return curChi
	elif len(limits) == 2:	
		curChi = 0
		effLMAX = 1+limits[1]-limits[0]
		for i in range(limits[0],limits[1]+1):
			curChi += np.power(sig[i]-org[i],2)*1.0/(np.power(err[i],2))
		#print "curChi: "+str(curChi)
		return curChi
	else:
		print "ML CALCULATION ERROR FOR Chi2: Calculation limits given, but of wrong structure."


#### CALC. effD2 #####
def calcEffD2(cl_new, cl_original, cl_original_errors, weights, lmax, norm=True, limits=[]):
	if len(limits)==0:
		curEffDSquared = 0.0
		for i in range(0,lmax):
			if cl_original_errors[i] != 0:
				curEffDSquared += 1.0/lmax*np.power(cl_new[i]-cl_original[i],2)*1.0/(np.power(cl_original_errors[i],2))*weights[i]*returnSign(cl_new[i]-cl_original[i])
		if norm == True:
			curEffDSquared=curEffDSquared*1.0/sumFromTo(absList(weights), 0, lmax)
		return curEffDSquared
	elif len(limits) == 2:
		curEffDSquared = 0.0
		#effLMAX = 1+limits[1]-limits[0]
		#print str(limits[0])+" => "+str(limits[1])
		for i in range(limits[0]-1,limits[1]):
			if cl_original_errors[i] != 0:
				curEffDSquared += 1.0/lmax*np.power(cl_new[i]-cl_original[i],2)*1.0/(np.power(cl_original_errors[i],2))*weights[i]*returnSign(cl_new[i]-cl_original[i])
		if norm == True:
			curEffDSquared=curEffDSquared*1.0/sumFromTo(absList(weights), 0, lmax) #, limits[0]-1, limits[1]
		return curEffDSquared
	else:
		print "ML CALCULATION ERROR FOR effD2: Calculation limits given, but of wrong structure."		


#### CALC. D2Log ####
def calcD2Log(cl_log_sig, cl_log_org, cl_err_org, weights, lmax, norm=True, limits=[]):
	if len(limits)==0:
		curDsquaredLog = 0.0
		for i in range(0,lmax):
			curDsquaredLog += 1.0/lmax*np.power(cl_log_sig[i]-cl_log_org[i],2)*1.0/(np.power(cl_err_org[i],2))*weights[i]*returnSign(cl_log_sig[i]-cl_log_org[i])
		if norm == True:
			curDsquaredLog = curDsquaredLog*1.0/sumFromTo(absList(weights), 0, lmax)
		return curDsquaredLog
	elif len(limits) == 2:
		curDsquaredLog = 0.0
		#effLMAX = 1+limits[1]-limits[0]
		for i in range(limits[0]-1,limits[1]):
			curDsquaredLog += 1.0/lmax*np.power(cl_log_sig-cl_log_org[i],2)*1.0/(np.power(cl_err_org[i],2))*weights[i]*returnSign(cl_log_sig[i]-cl_log_org[i])
		if norm == True:
			curDsquaredLog = curDsquaredLog*1.0/sumFromTo(absList(weights), limits[0]-1, limits[1])
		return curDsquaredLog
	else:
		print "ML CALCULATION ERROR FOR D2 Log: Calculation limits given, but of wrong structure."		


#### CALC. effAlmD2 from Alm #####
def calcEffD2Alm(alm_original, alm_new, alm_original_errors, weights, lmax):
	thisDSquared = 0

	for l in range(1,lmax+1):
		for m in range(0,l+1):
			if m != 0:
				thisDSquared += returnSign(abs(alm_new[H.sphtfunc.Alm.getidx(lmax, l,abs(m))])-abs(alm_original[H.sphtfunc.Alm.getidx(lmax, l,abs(m))]))*np.power(abs(alm_original[H.sphtfunc.Alm.getidx(lmax, l,abs(m))])-abs(alm_new[H.sphtfunc.Alm.getidx(lmax, l,abs(m))]),2)*1.0/np.power(abs(alm_original_errors[H.sphtfunc.Alm.getidx(lmax, l,abs(m))]),2)*weights[H.sphtfunc.Alm.getidx(lmax, l,abs(m))]
			
	return thisDSquared


#### SAVES FULL Alm LIST TO FILE
def saveAlmList(alm_list, saveFilepath, prefix):
	absList = []
	for i in range(0,len(alm_list)):
		absList.append(abs(alm_list[i]))
	np.savetxt(saveFilepath, absList)
	print "Alm-Lists data successfully saved..."


#### FEHLERFORTPFLANZUNG ZWEIER FEHLERLISTEN ZU EINER FEHLERLISTE			
def sumErrorDiffList(err1, err2):
	nlist = []
	for i in range(0, len(err1)):
		nlist.append(np.sqrt(np.power(err1[i],2)+np.power(err2[i],2)))
	if(len(err1) != len(err2)):
		print "WARNING: TWO LISTS OF DIFFERENT LENGTH USED FOR ERROR SUMMATION!"
	return nlist


#### SAVES ALL OPENED FIGURES TO GIVEN FILE ###
def saveAll(figureList, path="/net/scratch_icecube4/user/kalaczynski/Analysis_stuff/plots", allFigureTitles=[], transparency=False, dots=160):
	figureList = killTwinElements(figureList)
	allFigureTitles = killTwinElements(allFigureTitles)
	for i in range(0,len(figureList)):
		plt.figure(figureList[i])
		if len(figureList) > 0:
			if int(i) < len(allFigureTitles):
				plt.savefig(path+str(allFigureTitles[int(i)]), dpi=dots, transparent=transparency, bbox_inches='tight')
				print "saved figure \""+str(allFigureTitles[i])+"\"("+str(figureList[i])+") successfully."
			else:
				plt.savefig(path+"figure"+str(figureList[i])+".png", dpi=dots, transparent=transparency, bbox_inches='tight')
				print "saved figure ("+str(figureList[i])+") successfully."
	return True


#### RETURNS IMAGINARY PART OF LIST
def getIm(thisList):
	nList = []
	for i in range(0, len(thisList)):
		nList.append(np.imag(thisList[i]))
	return nList


#### CALC. WEIGHTS FOR Cl, effCl, almCl, effAlmCl
def calcWeights(sig, atm, errSig, errAtm):
	w = []
	for i in range(0,len(sig)):
		if errAtm[i] != 0:
			w.append((sig[i]-atm[i])*1.0/np.sqrt(np.power(errAtm[i],2)))#+np.power(errSig[i],2)
		else:
			w.append(0.)
	return w


#### CONTROLS PLOTS AND SUBPLOTS
def plotControl(myPlot, legend=0, title=""):
	if (type(myPlot) != list and type(myPlot) != tuple):
		if myPlot == -1:
			ret = plt.figure()
		else:
			ret = plt.figure(myPlot, ((1+np.sqrt(5))*6,12))
			GLOB_FIG_NUMBERS.append(myPlot)
	elif type(myPlot) == list or type(myPlot) == tuple:
		if len(myPlot) == 1 and myPlot[0]==-1:
			ret = plt.figure()
		elif len(myPlot)==2:
			plt.figure(myPlot[0], ((1+np.sqrt(5))*6,12))
			ret = plt.subplot(221+myPlot[1])
			GLOB_FIG_NUMBERS.append(myPlot[0])
		elif len(myPlot)==4:
			plt.figure(myPlot[0], ((1+np.sqrt(5))*6,12))
			ret = plt.subplot(myPlot[1]*100+myPlot[2]*10+1+myPlot[3])
			GLOB_FIG_NUMBERS.append(myPlot[0])
	if legend != 0:
		plt.legend(loc=legend)
		#plt.setp(plt.legend().get_lines(), linewidth=2.0)
	if title != "":
		FIG_TITLES.append(title)
		plt.title(title)
	else:
		FIG_TITLES.append("figure"+str(myPlot))
	return ret



#### CREATES PURE SIGNAL SKYMAP                                      ##### NEW #####
#def getPureSigMapList(N_nu,mu,Source_PS_Spline, signalSpline):
#	map_list = []
#
#	#hist, edges = np.genfromtxt(SignalZenith,unpack=False)
#	#hist = normHist(hist, 0.95)  
#	#signalSpline = interpolate.InterpolatedUnivariateSpline(edges,hist)
#	count = 0
#
#	while(N_nu > len(map_list)):
#		count += 1
#		gen_one_pointsource(mu,map_list,signalSpline, Source_PS_Spline)
#		sys.stdout.write("\r finished Pointsource... "+str(count)+"            ")
#		sys.stdout.flush()
#	sys.stdout.write("\n")	
#
#	if N_nu < len(map_list):
#		while(N_nu < len(map_list)):
#			ranIndex = np.random.random_integers(0,len(map_list)-1)
#			del map_list[ranIndex]
#
#	print " => total number events: "+str(len(map_list))
#
#	retPhis = [i[1] for i in map_list]
#	retThetas = [i[0] for i in map_list]
#	return np.array(retThetas), np.array(retPhis)


#### REDUCES ALM PLANE TO CERTAIN LENGTH                                      ##### NEW #####
def reduceAlmPlane(plane, limit):
	nAlmPlane =[]
	for i in range(0, limit+1):
		nAlmPlane.append([])
		for k in range (0,limit+1):
			nAlmPlane[i].append(plane[i][k])
	return nAlmPlane


#### KILLS ALL TWICE OR MORE TIMES APPEARING ELEMENTS ####           		     ##### NEW #####
def killTwinElements(li):
	nli = []
	for i in range(0,len(li)):
		if not li[i] in nli:
			nli.append(li[i])
	return nli
				

#### GIVES MIN. AND MAX. OF A LIST OF LISTS
def getMinMaxRanges(li, puffer=False, res=200):
	foundNaN = False
	if type(li[0]) == list or type(li[0]) == np.ndarray:
		nMin = li[0][0]
		nMax = li[0][0]
		if np.isnan(nMin):
			foundNaN = True

		for i in range(0, len(li)):
			for k in range(0,len(li[i])):
				if not np.isnan(li[i][k]):
					if not np.isnan(nMin):
						if li[i][k] < nMin:
							nMin = li[i][k]
						elif li[i][k] > nMax:
							nMax = li[i][k]
					else:
						nMin = li[i][k]
						nMax = li[i][k]
				else:
					foundNaN = True
	else:
		nMin = li[0]
		nMax = li[0]
		if np.isnan(nMin):
			foundNaN = True

		for i in range(0, len(li)):
			if not np.isnan(li[i]):
				if not np.isnan(nMin):
					if li[i]< nMin or np.isnan(nMin):
						nMin = li[i]
					elif li[i] > nMax:
						nMax = li[i]
				else:
					nMin = li[i]
					nMax = li[i]		
			else:
				foundNaN = True
	if foundNaN == True:
		print "ML-WARNING: Found NaN Values in List for calculating min and max borders for plotting"
				
	if puffer==True:
		nMin = nMin-(nMax-nMin)*1.05/res
		nMax = nMax+(nMax-nMin)*1.05/res
		
	return nMin, nMax

	
#### RENORMS A GIVEN SKYMAP TO EVENT NUMBER ####
def renormSkymap(thisMap, NEvents):
	norm = float(NEvents)*1.0/sum(thisMap)
	newMap = thisMap*norm
	return newMap
	
#### ADJUSTS THE SKYMAP TO MATCH THE UNITS OF THE MC (mJy -> # EVENTS) ####
def adjustUnitsSkymap(thisMap):
	#E_avg = 62922635.8362 # GeV (IC)
	E_avg = 5.79*10**(-15) # GeV (NVSS)
	units_conv = 6.2415091*10**(-24)
	factor = units_conv*1.0/E_avg
	newMap = thisMap*factor
	return newMap
	
	
#### CREATES SAVING DIRECTORY ####
def prepareDirectory(direc, subs=False, saved_plots=False):
	if (not(os.path.exists(direc))):
		os.makedirs(direc)
		print "Created Folder {}".format(direc)
	
	if (not(os.path.exists(direc+"plots/")) and subs):
		os.mkdir(direc+"plots/")
		print "Created Folder {} in {}".format("plots/", direc)
	
	if (not(os.path.exists(direc+"data/")) and subs):
		os.mkdir(direc+"data/")
		print "Created Folder {} in {}".format("data/", direc)
			
	if (not(os.path.exists(direc+"condor_unite_plots/")) and saved_plots):
		os.mkdir(direc+"condor_unite_plots/")
		print "Created Folder {} in {}".format("condor_unite_plots/", direc)
	
	
	
#### SAVES ALL sin() AND cos() OF CERTAIN ANGLES TO GLOBAL VARIABLE ####
def getGlobalSinCos(Nside, globPixCosTheta, globPixSinTheta, globPixCosPhi, globPixSinPhi):
	Npix = H.nside2npix(Nside)
	theta, phi = H.pix2ang(Nside, np.arange(0,Npix))
	#for i in range(0,npix):
	globPixCosTheta += (np.cos(theta))
	globPixSinTheta += (np.sin(theta))
	globPixCosPhi += (np.cos(phi))
	globPixSinPhi += (np.sin(phi))
	
	print "... finished calculating all possible sin() & cos() values for healpy pixels."

	
#### SAVES ALL PIXEL NEIGHBOURS TO A CERTAIN LIST
def getAllNeighbours(Nside, neighList):
	Npix = H.nside2npix(Nside)
	#neighList = []
	for i in range(0, Npix):
		neighList.append(H.get_all_neighbours(Nside, i))
	print "... finished calculating all possible neighbours for healpy pixels."	
	
	
#### CENTERS ALL HITS TO PIXEL CENTER
def centerHits(Nside, th, ph):
	retTheta, retPhi = H.pix2ang(Nside,H.ang2pix(Nside,th, ph))
	return retTheta, retPhi
	
def power_law(x, *p):
	return np.power(x, -p[0])*p[1]
	
def power_law_const(x, *p):
	return np.power(x, -p[0])*p[1]+p[2]
	
#### GET EXPONENTIAL FIT STUFF ####
def exponential(x, a, perc=0.7, x0=0., bw=1.):
    return (1.-perc)*a*bw*np.exp(-1.*a*(x-x0))

def exp_diff(a, x=[], perc=0.7, x0=0., bw=1., histo=[], weights=[]):
    histo=np.array(histo)
    if len(weights)==0:
        weights = np.ones(len(histo))
    return (histo-exponential(x, a, perc, x0, bw))*weights

def exp_int(x, a, perc=0.7, x0=0.):
    #x0 is integration starting point ->> integrate from x0 to x
    return 1.-(1.-perc)*np.exp(-1.*a*(x-x0))
    
def exp_int_oneMinus(x, a, perc=0.7, x0=0.):
    #x0 is integration starting point ->> integrate from x0 to x
    return (1.-perc)*np.exp(-1.*a*(x-x0))

def exp_int_inverse(cut, a=1000., perc=0.7, x0=0.):
    #x0 is integration starting point ->> integrate from x0 to x
    return x0-(np.log((1.-cut)/(1.-perc))*1./a)

def fitExp(edges, histo, err_histo, a=1., perc=0.7, x0=0., plot=False, N=1000.):
	"""
	input:
	- edges as x-values, bin edges will be corrected to bin centers
	- histo as y-values
	- err_histo as error on y-values, are given as weight to leastsq()
	- function arguments a, percentile, x0 integration start
	- plot is plotting boolean
	
	returns:
	- fit parameter a
	- percentile
	- x0 integration start value
	- bin width bw
	- error on fit parameter a    
	"""
	edges=np.array(edges)
	histo=np.array(histo)
	err_histo=np.array(err_histo)
	
	if len(edges) == len(histo)+1:
			edges = setNewEdges(edges)
	bin_w=edges[1]-edges[0]    
	p0=[a]
	
	weights_nonzero = []
	for e in err_histo:
		if e !=0:
			weights_nonzero.append(1./e)
		else:
			weights_nonzero.append(0.)
    
	args_nonzero=(edges, perc, x0, bin_w, histo, weights_nonzero)
	result_nonzero = leastsq(func=exp_diff, x0=p0, args=args_nonzero, full_output=True)
	
	weights = []
	for i in range(0, len(err_histo)):
		if err_histo[i]!=0:
			weights.append(1./err_histo[i])
		else:
			#weights.append(np.sqrt(N*1./exponential(edges[i], result_nonzero[0][0], perc=perc, x0=x0, bw=bin_w)))
			weights.append(0.)
	#~ print max(weights)
	#~ print min(weights)
	args=(edges, perc, x0, bin_w, histo, weights)
	result = leastsq(func=exp_diff, x0=p0, args=args, full_output=True)
	(popt, pcov, infodict, errmsg, ier) = result
	#print np.isinf(weights).sum()
	#print "Result: "+str(result)
	print "popt "
	print popt
	if len(histo) > len(p0):
			print "(np.array(exp_diff(popt, *args))**2).sum() " + str((np.array(exp_diff(popt, *args))**2).sum())
			print "(len(histo) - len(p0)) "+str(len(histo) - len(p0))
			s_sq = (np.array(exp_diff(popt, *args))**2).sum() / (len(histo) - len(p0))
			pcov = pcov * s_sq
	else:
		print "FIT ERROR"

	hist_fit = exponential(edges, result[0], perc=perc, x0=x0, bw=bin_w)
	
	err = np.sqrt(np.diag(pcov))
	#print "error: "+str(err)
	
	if plot == True:
			plt.plot(edges, hist_fit, label="fitted data", color="#CC3333", linestyle="--")
	print popt[0], perc, x0, err[0]
	return [popt[0], perc, x0, bin_w, err[0]]


#### CAUCHY FIT STUFF ####
def cauchy(x, *p):
		s,t,n = p
		return s*n*1./(np.pi*(s**2+(x-t)**2))

def cauchy_int(x, *p):
		s,t,n = p
		return (0.5 + np.arctan((x-t)/s)/np.pi)*n

def cauchy_inv_int(x, *p):
		s,t,n=p
		n=1.
		return s*np.tan(np.pi*(x*1./n-0.5))+t

def fitCauchy(data, edges, histo, plot=False):
		if len(edges) == len(histo)+1:
			edges = setNewEdges(edges)
				
		norm = sum(histo)*(edges[1]-edges[0])
		p0=[0.0001, np.mean(data), 1./norm]
		coeff, var_matrix = curve_fit(cauchy, edges, histo, p0=p0)
		hist_fit = cauchy(edges, *coeff)
		
		err = np.sqrt(np.diag(var_matrix))
		coeff[0]=abs(coeff[0])
		coeff[2]=abs(coeff[2])
		
		if plot == True:
				plt.plot(edges, hist_fit, label="fitted data", color="c", linestyle="--")
				
		return coeff, err

#### GAUSS FUNCTION FOR FITTING
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2*1.0/(2.*sigma**2))

#### GAMMA FUNCTION FOR FITTING ####
def gammaDistr(x, *p):
		k, theta, s, norm = p
		#~ print p
		return GM.pdf(x, k, loc=s, scale=theta)*norm
		
#### GET GAMMA MEDIAN ####
def getGammaMed(a, scale, loc):
	return GM.median(a, loc=loc, scale=scale)

#### RETURNS MINIMUM & MAXIMUM INDEX OF A LIST
def getMinMaxIndex(li):
	mi = min(li)
	ma = max(li)
	for i in range(0,len(li)):
		if li[i] == mi:
			minIndex = i
		if li[i] == ma:
			maxIndex = i
	return (minIndex, maxIndex)			


#### FIT GAUSSIAN TO GIVEN DISTRIBUTION
def fitGaussian(data, edges, histo, plot=False, nLabel=""):
	if len(edges) == len(histo)+1:
		edges = setNewEdges(edges)
		
	p0 = [max(histo), edges[getMinMaxIndex(histo)[1]], np.std(data)]
	coeff, var_matrix = curve_fit(gauss, edges, histo, p0=p0)
	hist_fit = gauss(edges, *coeff)

	err0 = np.sqrt(var_matrix[0][0])
	err1 = np.sqrt(var_matrix[1][1])
	err2 = np.sqrt(var_matrix[2][2])
	if nLabel =="":
		nLabel = r"Gauss Fit: $\sigma=$"+str(round(coeff[2], 2))+" $\pm$ "+str(round(err2, 2))
	if plot == True:
		plotGaussian(edges, coeff[0], coeff[1], coeff[2], nLabel=nLabel)
		#plt.plot(edges, hist_fit, label='Fitted data')
		
	return (coeff[0],coeff[1], coeff[2], err0, err1, err2)
	
#### FIT GAMMA DISTRIBUTION ####
def fitGammaDistr(data, edges, histo, plot=False, shift=0.0):
	if len(edges) == len(histo)+1:
		edges = setNewEdges(edges)
		
	e		= np.mean(data)#+edges[getMinMaxIndex(histo)[1]]
	var	= np.std(data)**2
	#~ shift=0.0
	norm=sum(histo)*(edges[1]-edges[0])
	#~ print "Norm: "+str(norm)
	#~ print "Mean e: " +str(e)
	#~ print "Var: " +str(var)
	p0 	= [e**2/var, var/e, shift, norm]
	#~ plotGammaDistr(edges, *p0, color="r", linestyle="--")
	coeff, var_matrix = curve_fit(gammaDistr, edges, histo, p0=p0, factor=1, maxfev=10000) #, full_output=True)
	hist_fit = gammaDistr(edges, *coeff)
	
	err = [np.sqrt(var_matrix[0][0]), np.sqrt(var_matrix[1][1]), np.sqrt(var_matrix[2][2]), np.sqrt(var_matrix[3][3])]
	
	if plot == True:
		plotGammaDistr(edges, *coeff)
	

	#~ print "Set Parameters p0= "+str(p0)	
	#~ print "relative errors: "
	#~ for i in range(0, len(coeff)):
		#~ print str(i)+": "+str(err[i]/coeff[i])
	return (coeff[0],coeff[1], coeff[2], coeff[3], err[0], err[1], err[2], err[3]) #, coeff[2], , err2
	
#### PLOT GAUSSIAN
def plotGaussian(x, norm, mu, sigma, color="blue", linestyle="--", nLabel=""):
	y = gauss(x, *[norm, mu, sigma])
	plt.plot(x, y, color=color, linestyle=linestyle, label=nLabel)	

#### PLOT GAMMA DISTR
def plotGammaDistr(x, k, theta, s, norm, color="k", linestyle="-"):
	y = gammaDistr(x, *[k, theta, s, norm])
	plt.plot(x, y, color=color, linestyle=linestyle)
	
	
#### GET SIGNIFICANCE FOR 50% OF ALL MEASUREMENTS WOULD LEAD TO 90% CONFIDENCE
def getSignificanceFromGauss(meanSig, meanBg, stdBg, errSig, errBg, errStdBg):
	cutOff 			= erfinv(0.8)*np.sqrt(2)*stdBg+meanBg   ### 90% C.L. 0.8=2*0.9-1.
	cutOff_err		= np.sqrt(np.power(erfinv(0.8)*np.sqrt(2)*errStdBg,2)+np.power(errBg,2))
	
	significance 	= (meanSig-meanBg)*1.0/stdBg
	sig_err 	= np.sqrt(  np.power(errSig*1.0/stdBg,2)
								+np.power(errBg*1.0/stdBg,2)
								+np.power(significance*1.0/stdBg*errStdBg,2)  )
	
	print "\n \t CUTOFF AT 90% CONFIDENCE FOR BG: "+str(cutOff)+" \n \t 50%(Median) FOR SIG IS AT: "+str(meanSig)
	print "\t MEDIAN IS AT SIGNIFICANCE OF: "+str((meanSig-meanBg)*1.0/stdBg)+" sigma \n"
	
	return cutOff, significance, cutOff_err, sig_err
	
def getSignificanceFromStud(df, loc, scale, stdBG, medSig, medBG):
	#t.pdf(x, df, loc, scale) is identically equivalent to t.pdf(y, df) / scale with y = (x - loc) / scale
	cutOff=t.ppf(0.9, df, loc, scale)
	significance = (medSig-medBG)/stdBG
	#~ print "stdBG "+str(stdBG)
	
	print "\n \t CUTOFF AT 90% CONFIDENCE FOR BG: "+str(cutOff)+" \n \t 50%(Median) FOR SIG IS AT: "+str(medSig)
	print "\t MEDIAN IS AT SIGNIFICANCE OF: "+str(significance)+" sigma \n"
	
	return cutOff, significance
	
def getSignificanceFromExp(edges, medS, err_medS, p70, histo, histo_cummulative, percentile, a, err_a, N=1000):
	if len(edges) == len(histo_cummulative):
		edges=setOldEdges(edges)
	elif len(edges) == len(histo_cummulative)+1:
		print "Edges are already okay!"
	else:
		print "Something went wrong with the edges :("
	it = np.where(edges<=medS)[0][-1]
	if medS<p70:
		alpha = percentileofscore(histo, medS)/100.
		err_alpha = np.sqrt(alpha*(1.-alpha)*1./N)
		alpha_oneMinus = 1.-alpha
		
	else:
		alpha = exp_int(medS, a, perc=percentile, x0=p70) # fit method, better for x>p90			
		term1 = np.power(0.5*(exp_int(medS, a, perc=percentile*(1.+1./np.sqrt(N)), x0=p70)-exp_int(medS, a, perc=percentile-(1.-percentile)/np.sqrt(N), x0=p70)), 2)
		term2 = np.power(exp_int_oneMinus(medS, a, perc=percentile, x0=p70)*(medS-p70), 2)*np.power(err_a, 2)
		err_alpha = np.sqrt(term1+term2)
		alpha_oneMinus = exp_int_oneMinus(medS, a, perc=percentile, x0=p70)
			
	print "\n \t CUTOFF AT 90% CONFIDENCE FOR BG: "+str(p70)+" \n \t 50%(Median) FOR SIG IS AT: "+str(medS)
	print "\t MEDIAN IS AT SIGNIFICANCE QUANTILE OF: "+str(alpha)+" OR (1-Q)="+str(alpha_oneMinus)+"+-" +str(err_alpha)+" \n"
	return [alpha, alpha_oneMinus, err_alpha]
	
### calculate Sigma from q-quantile and corresponding error###
def getSigmaFromExp(q, err_q, onemin=False):
	if onemin:
		sigma     = erfcinv(2.*q)*np.sqrt(2.)
	else:
		sigma     = erfinv(2.*q-1.)*np.sqrt(2.)
	err_sigma = err_q*np.sqrt(2.*np.pi)*np.exp(np.power(erfinv(2.*q-1.), 2.))
	return sigma, err_sigma

def getSignificanceFromCauchy(s, t, n, errs, errt, errn, tSig, errtSig):
	#### t = median, s = width, n=norm ####
	s=abs(s)
	n=abs(n)
	cutOff = cauchy_inv_int(0.9, s, t, n)
	var_s = np.power(np.tan(np.pi*(0.9/n-0.5))*errs, 2)
	var_t = np.power(errt, 2)
	var_n = np.power(np.pi*s*0.9*errn/((n*np.sin(np.pi*(1-0.9/n)))**2), 2)
	cutOff_err = np.sqrt(var_s+var_t+var_n)
	#widthBG = s
	
	significance = (tSig-t)*1./s
	sig_err      = np.sqrt( np.power(errtSig*tSig/s, 2) + np.power(errt*t/s, 2) + np.power(significance*errs/s, 2))
	
	print "\n \t CUTOFF AT 90% CONFIDENCE FOR BG: "+str(cutOff)+" \n \t 50%(Median) FOR SIG IS AT: "+str(tSig)
	print "\t MEDIAN IS AT SIGNIFICANCE OF: "+str(significance)+" sigma \n"
	
	return cutOff, significance, cutOff_err, sig_err
	
def linInterpolateDeffN(N1,N2,Deff1,Deff2, resultDeff=-1, resultN=-1):
	if resultN==-1:
		m    = (N2-N1)*1.0/(Deff2-Deff1) #slope of y=m*x+b
		ret1 = (resultDeff-Deff1)*m+N1
		return ret1
	elif resultDeff==-1:
		m    = (Deff2-Deff1)*1.0/(N2-N1) #slope of y=m*x+b		
		ret1 = (resultN-N1)*m+Deff1
		return ret1
	
#### SIG=A=K, SCALE=THETA, LOC=S ####
def getSignificanceFromGamma(k, theta, s, errk, errtheta, errs, bk, btheta, bs, berrk, berrtheta, berrs):
	Sig			=[k, theta, s]
	print "Sig: "+str(Sig)
	errSig	=[errk, errtheta, errs]
	BG			=[bk, btheta, bs]
	print "BG: "+str(BG)
	errBG 	=[berrk, berrtheta, berrs]
	cutOff 			= GM.ppf(0.9, BG[0], loc=BG[1], scale=BG[2])
	print "cutOff: "+str(cutOff)
	
	BGmed_err	= np.power(errBG[2]*1.0*gammaMed(*BG),2) + np.power(errBG[0]*1.0*gammaMedDeriv(errBG[0], errBG[2]), 2) 				## squared
	sigmed_err	= np.power(errSig[2]*1.0*gammaMed(*Sig),2) + np.power(errSig[0]*1.0*gammaMedDeriv(errSig[0], errSig[2]), 2) ## squared
	std_err = np.power(errBG[0]*BG[1]*1.0/2., 2)/BG[0] +errBG[1]**2 * BG[0]
	
	print"medBG: "+str(gammaMed(*BG))+" +- "+str(np.sqrt(BGmed_err))
	print "medSig: " +str(gammaMed(*Sig))+" +- "+str(np.sqrt(sigmed_err))
	print "stdBG: "+str(np.sqrt(gammaVar(*BG[0:-1]))) +" +- "+str(np.sqrt(std_err))
	
	significance= (gammaMed(*Sig)-gammaMed(*BG))/(np.sqrt(gammaVar(*BG[0:-1])))
	sign_err		= np.sqrt(BGmed_err*1.0/gammaVar(*BG[0:-1]) + sigmed_err*1./gammaVar(*BG[0:-1]) + np.power((gammaMed(*Sig)-gammaMed(*BG))/gammaVar(BG[0], scale=BG[2])*std_err,2))
	
	print "\n \t CUTOFF AT 90% CONFIDENCE FOR BG: "+str(cutOff)+" \n \t 50%(Median) FOR SIG IS AT: "+str(gammaMed(*Sig))
	print "\t MEDIAN IS AT SIGNIFICANCE OF: "+str(significance)+" sigma \n"
	
	return cutOff, significance, sign_err
	
def gamma2gauss(k, theta, s, errk, errtheta, errs):
	BG			=[k, theta, s]
	errBG 	=[errk, errtheta, errs]
	
	med     = gammaMed(*BG)
	med_err	= np.sqrt(np.power(errBG[2]*1.0*gammaMed(*BG),2) + np.power(errBG[0]*1.0*gammaMedDeriv(errBG[0], errBG[2]), 2))
	std     = np.sqrt(gammaVar(*BG[0:-1]))
	std_err = np.power(errBG[0]*BG[1]*1.0/2., 2)/BG[0] +errBG[1]**2 * BG[0]
	return med, std, med_err, std_err
	
	
	
#### GET SIGNIFICANCE ERROR FOR 50% OF ALL MEASUREMENTS WOULD LEAD TO 90% CONFIDENCE
#def getSignificanceErrorFromGauss(meanSig, meanBg, stdBg, errSig, errBg, errStdBg,  significance):
#	errSignSquared = np.power(errSig*1.0/stdBg,2)+np.power(errBg*1.0/stdBg,2)+np.power(significance*1.0/stdBg*errStdBg,2)
#	return np.sqrt(errSignSquared)
#### GAMMA MEDIAN APPROXIMATIONS ####

def gammaMed(a, scale, loc):
	return GM.median(a, loc=loc, scale=scale)
def gammaMedDeriv(a, scale):
	return scale/3.0 *(9.*a +1.2*a - 0.16) /(9.*a +1.2*a + 0.04)
def gammaVar(a, scale):
	return a*scale**2
	
#### CALCULATE N_SOU NECESSARY TO BE SENSITIVE BY USING LINEAR INTERPOLATION
def linInterpol(p1x, p1y, p2x, p2y, value):
	"""
	- two points with coordinates p1x/y and p2x/y
	- y value -> corresponding x return value
	"""
	m = (p2y-p1y)*1.0/(p2x-p1x)
	ret = (value-p1x)*m+p1y
	return ret


#### CALCULATE N_SOU NECESSARY TO BE SENSITIVE BY USING LINEAR INTERPOLATION USING ERRORS
def linInterpol_withErr(p1x, p1y, p2x, p2y, p1x_err=0., p2x_err=0., value=erfinv(0.8)*np.sqrt(2), value_err=0.0, p1y_err=0., p2y_err=0.):
	"""
	- two points with coordinates p1x/y and p2x/y
	- y value -> corresponding x return value
	- possible errors on all values
	- returns: y value and y error
	"""

	term1 = np.power(   -1.0/np.power((p2x-p1x),2)* ( (p2x-value)*p1y+(value-p1x)*p2y)  + 1.0/(p2x-p1x)*p1y,2 ) *np.power(p2x_err,2) #p2x_err
	term2 = np.power(   1.0/np.power(p2x-p1x,2)* ( (p2x-value)*p1y+(value-p1x)*p2y)  - 1.0/(p2x-p1x)*p2y,2 ) *np.power(p1x_err,2) #p1x_err
	term3 = np.power(   -1.0/(p2x-p1x)*(p2y-p1y),2 ) *np.power(value_err,2) #value error	
	term4 = np.power(1. + (p1x-value)*1./(p2x-p1x), 2) *np.power(p1y_err,2) #p1y error
	term5 = np.power((value-p1x)*1./(p2x-p1x), 2) *np.power(value_err,2) #p2y error
		
	m    = (p2y-p1y)*1.0/(p2x-p1x) #slope of y=m*x+b
	ret1 = (value-p1x)*m+p1y       # ret1 = m*value+b
	ret2 = np.sqrt(term1+term2+term3+term4+term5)
	
	return ret1, ret2

#### GETS ALL LABEL ITEMS FROM LEGEND
def getLegendItems(fig):
	plt.figure(fig)
	legend = plt.legend()
	lines =  legend.legendHandles
	texts = legend.get_texts()
	stringTexts = []
	
	for i in range(0, len(texts)):
		stringTexts.append(texts[i].get_text())
		
	return [lines, stringTexts]


#### GETS ALL LABEL ITEMS FROM LEGEND
def renameLegendItem(fig, position, newName):
	plt.figure(fig)
	legend = plt.legend()
	texts = legend.get_texts()
	texts[position].set_text(newName)
		
	return True
	
	
#### RENAMES ITEM IN LEGEND
def renameLegendItem(fig, position, newName):
	plt.figure(fig)
	legend = plt.legend()
	texts = legend.get_texts()
	texts[position].set_text(newName)
		
	return True	
	

#### DELETS ITEM FROM LEGEND
def deleteLegendItem(fig, position):
	if type(fig) == int:
		canv = plt.figure(fig)
		ax = canv.get_axes()[0]
	else:
		canv = plt.figure(fig[0])
		ax = canv.get_axes()[fig[1]]
		
	handles, labels = ax.get_legend_handles_labels()
	for i in position:
		handles.remove(handles[i])
		labels.remove(labels[i])
	ax.legend(handles, labels)
	print str(len(handles))+", "+str(len(labels))
	
	if type(fig) == int:
		plt.figure(fig)
	else:
		plt.figure(fig[0])
		
	return True
	
	
#### DELETES LAST ITEM FROM LEGEND
def deleteLastLegendItem(fig):
	if type(fig) == int:
		canv = plt.figure(fig)
		ax = canv.get_axes()[0]
	else:
		canv = plt.figure(fig[0])
		ax = canv.get_axes()[fig[1]]
		
	handles, labels = ax.get_legend_handles_labels()
	handles.remove(handles[-1])
	labels.remove(labels[-1])
	ax.legend(handles, labels)
	
	if type(fig) == int:
		plt.figure(fig)
	else:
		plt.figure(fig[0])
		
	return True


#### GETS TABLES FROM A SPLINE FOR QUICK RANDOM GENERATOR
def getTablesFromSpline(spline, precision=10000, boundaries=[0.0, 90.0]):
	norm = spline.integral(boundaries[0], boundaries[1])
	squareArea = norm*1.0/precision
	upperLimit = []
	lowerLimit = []
	cutOff = [boundaries[0]]
	curCut = boundaries[0]
	while curCut < boundaries[1]:
		start = spline(curCut)

		if start <= 0.0: # just to be sure... 
			start = 0.0001

		stepSize = squareArea*1.0 / abs(start)
			
		if start < 0.0001:	 #and spline(curCut+squareArea*1.0 / abs(0.001)) > start:
			stepSize = squareArea*1.0/abs(0.0001)
			#print stepSize
		#print str(curCut)+" "+str(stepSize)+" "+str(start)+" "+str(spline(curCut+squareArea*1.0 / abs(0.0001)))
				
		end = spline(curCut+stepSize)
		if start < spline(curCut+stepSize) or curCut == boundaries[0]:
			stepSize = squareArea*1.0/spline(curCut+stepSize)
			#print "JO"

		if stepSize > 1.0 and len(cutOff) < precision/100:
			print "ML-WARNING: StepSize is at about: "+str(stepSize)+" at i="+str(len(cutOff))+" val: "+str(start)
			
		curCut = curCut+stepSize

		cutOff.append(curCut)
		upperLimit.append(max(start, end))
		lowerLimit.append(min(start, end))
		#print "CUTOFF: "+str(curCut)+", MIN: "+str(min(start,end))+", MAX: "+str(max(start,end))

	del cutOff[-1]
	del upperLimit[-1]
	del lowerLimit[-1]
	return cutOff, lowerLimit, upperLimit



#### GETS RANDOM NUMBERS FROM SET OF TABLE USING QUICK RANDOM GENERATOR
def quickRandomNumbers(spline, tables, numbers):
	li = []
	numberSquares = len(tables[1])
	while len(li) < numbers:
		square = np.random.randint(0.0, high=numberSquares)
		y = np.random.uniform(0.0, tables[2][square])
		x = np.random.uniform(0.0, 1.0)*(tables[0][square+1]-tables[0][square])+tables[0][square]
		if y < tables[1][square]:
			li.append(x)
		else:
			if spline(x) > y:
				li.append(x)

	return li


#### RETURNS LIST OF CLOSEST DISTANCES TO NEXT HIT
def getClosestDistance(theta, phi, closestOnly=True):
	cosThHit = np.cos(theta)
	sinThHit = np.sin(theta)
	cosPhiHit = np.cos(phi)
	sinPhiHit = np.sin(phi)
	
	minDist = np.array([])
	#curDist = []
	if closestOnly:
		for i in range(0, len(theta)):
			cosAngle = getQuickAngleBetween(cosThHit[i], sinThHit[i], cosPhiHit[i], sinPhiHit[i], cosThHit, sinThHit, cosPhiHit, sinPhiHit)	
			cosAngle = [round(elem,13) for elem in cosAngle]
			curDist = np.arccos(cosAngle)
			cleaned = np.delete(curDist, np.where(curDist==min(curDist)))
			minDist.append(min(cleaned)*180.0/np.pi)
			#print min(cleaned)
	else:
		for i in range(0, len(theta)):
			cosAngle = getQuickAngleBetween(cosThHit[i], sinThHit[i], cosPhiHit[i], sinPhiHit[i], cosThHit, sinThHit, cosPhiHit, sinPhiHit)	
			cosAngle = [round(elem,13) for elem in cosAngle]
			curDist = (np.arccos(cosAngle)*180.0/np.pi)
			#cleaned = np.delete(curDist, np.where(curDist==min(curDist)))
			#print type(curDist)
			#print len(curDist)
			minDist = np.concatenate((minDist,curDist))
			print len(minDist)
			#print min(cleaned)	
	return minDist


#### RETURNS R MATRIX FOR FELDMAN COUSIN CALCULATIONS
def getFeldmannCousinR(gaussD2, param, pres=200, mini=-1.0, maxi=50.0, accepted=0.9, Neyman=False):
	## mini and maxi should be given by getMinMaxRanges(allD2, puffer=True)
	R = []
	p = []
	integral = []
	
	### calculate p matrix (2dim.) and integrals over gaussian D2 distributions ###
	for thisParam in range(0, len(param)):
		R.append([])
		if gaussD2[thisParam][0] == 0.0: 
			p.append([float('NaN') for i in range(0, pres)])
			integral.append(float('NaN'))
			continue
		diff = (maxi-mini)*1.0/pres
		p.append([])
		integral.append(quad(lambda x: gauss(x, *gaussD2[thisParam]), gaussD2[thisParam][1]-10.0*gaussD2[thisParam][2], gaussD2[thisParam][1]+10.0*gaussD2[thisParam][2])[0])
		for i in range(pres):
			x = mini+(maxi-mini)*1.0/pres*i
			#pixelSize = quad(lambda x: gauss(x, *gaussD2[thisParam]), x-diff/2.0, x+diff/2.0)[0]
			pixelSize = diff
			p[thisParam].append(gauss(x, *gaussD2[thisParam])*1.0/integral[thisParam]*pixelSize)
			#print pixelSize
			
	### rotate 2dim. list ###
	p_rot = zip(*p[::1])
			
	### calculate 2dim. R-matrix ###
	for i in range(pres):
		x = mini+(maxi-mini)*1.0/pres*i
		for thisParam in range(0, len(param)):
			if gaussD2[thisParam][0] == 0.0:
				R[thisParam].append(0.0)		#[float('NaN') for i in range(0, len(param)+1)])
				continue
			R[thisParam].append(p[thisParam][i]/max(p_rot[i]))
	
	#print R
	### calculate upper and lower limit using p and R matrices

	p_cp = copy.deepcopy(p)
	R_cp = copy.deepcopy(R)
	acceptedBins = []
	#print integral 
	
	for thisParam in range(0, len(param)):
		#print "PARAM: "+str(param[thisParam])
		if gaussD2[thisParam][0] == 0.0: 
			acceptedBins.append([float('NaN') for i in range(0, pres)])
			continue 
		sumUp = 0.0
		acceptedBins.append([])
		s=1
		while sumUp <= accepted:
			if Neyman:
				curIndex = len(R[thisParam])-s
			else:
				curIndex = R[thisParam].index(max(R[thisParam]))
			sumUp += p[thisParam][curIndex]
			acceptedBins[thisParam].append(curIndex)
			p[thisParam][curIndex] = -1.0
			R[thisParam][curIndex] = -1.0
			s += 1
			#print sumUp
			
	#print acceptedBins
	
	### return ###	
	return [p_cp, R_cp, acceptedBins]
	
	
#### RETURNS R MATRIX FOR FELDMAN COUSIN CALCULATIONS ----- NEW ------
def getFeldmannCousinR_2(D2hist, D2ed, param, mini=-1.0, maxi=50.0, accepted=0.9, Neyman=False):
	""" 
	- D2hist is array of TS hists
	- D2ed is edges of all TS hists
	- param is Number of Sources
	- mini and maxi should be given by getMinMaxRanges(allD2, puffer=True)
	- accepted -> Confidence interval/Limit	
	"""
	R = []
	p = []
	integral = []
	pres = len(D2ed)
	
	### calculate p matrix (2dim.) and integrals over gaussian D2 distributions ###
	for thisParam in range(0, len(param)):
		R.append([])
		if param[thisParam]== 0.0: 
			p.append([float('NaN') for i in range(0, pres)])
			integral.append(float('NaN'))
			continue
		diff = D2ed[1]-D2ed[0]
		p.append([])
		integral.append(sum(D2hist[thisParam])) #*diff)
		print "integral: ", integral[-1]
		for i in range(pres):
			#~ x = mini+(maxi-mini)*1.0/pres*i
			#~ #pixelSize = quad(lambda x: gauss(x, *gaussD2[thisParam]), x-diff/2.0, x+diff/2.0)[0]
			#~ pixelSize = diff
			#~ p[thisParam].append(gauss(x, *gaussD2[thisParam])*1.0/integral[thisParam]*pixelSize)
			#~ #print pixelSize
			p[thisParam].append(D2hist[thisParam][i]*1./integral[thisParam])
			
	### rotate 2dim. list ###
	p_rot = zip(*p[::1])
			
	### calculate 2dim. R-matrix ###
	for i in range(pres):
		for thisParam in range(0, len(param)):
			if param[thisParam]== 0.0:
				R[thisParam].append(0.0)		#[float('NaN') for i in range(0, len(param)+1)])
				continue
			if max(p_rot[i]) != 0:
				R[thisParam].append(p[thisParam][i]/max(p_rot[i]))
			else:
				R[thisParam].append(0.)
	
	#print R
	### calculate upper and lower limit using p and R matrices

	p_cp = copy.deepcopy(p)
	R_cp = copy.deepcopy(R)
	acceptedBins = []
	#print integral 
	
	for thisParam in range(0, len(param)):
		#print "PARAM: "+str(param[thisParam])
		if param[thisParam]== 0.0: 
			acceptedBins.append([float('NaN') for i in range(0, pres)])
			continue 
		sumUp = 0.0
		acceptedBins.append([])
		s=1
		while sumUp <= accepted:
			if Neyman:
				curIndex = len(R[thisParam])-s
			else:
				curIndex = R[thisParam].index(max(R[thisParam]))
			sumUp += p[thisParam][curIndex]
			acceptedBins[thisParam].append(curIndex)
			p[thisParam][curIndex] = -1.0
			R[thisParam][curIndex] = -1.0
			s += 1
			#print sumUp
			
	#print acceptedBins
	
	### return ###	
	return [p_cp, R_cp, acceptedBins]
	
#### RETURNS Neyman LIMITS CALCULATIONS ----- NEW ------
def getExperimentalLimits(D2, mini=-1.0, maxi=50.0, accepted=0.9, precision=200):
	""" 
	- D2 test statistics distribution
	- mini and maxi should be given by getMinMaxRanges(allD2, puffer=True)
	- accepted -> Confidence interval/Limit	
	"""
	param = len(D2)
	err_accept = []
	val = accepted*(1.-accepted)
	D2histos = []
	for curChange in range(0, param):
			histo, edges = np.histogram(D2[curChange], bins=precision, range=(mini, maxi))
			n = sum(histo)
			histo = np.array(histo)*1./n
			D2histos.append(histo)
			err_accept.append(np.sqrt(val/len(D2[curChange])))
	D2edges = setNewEdges(edges)
	#print err_accept
	
	quantiles = []
	err_q = []
	for curChange in range(0, param):
		quantiles.append(np.percentile(D2[curChange], (1.-accepted)*100))
		v1 = np.percentile(D2[curChange], (1.-(accepted+err_accept[curChange]))*100)
		v2 = np.percentile(D2[curChange], (1.-(accepted-err_accept[curChange]))*100)
		err_q.append(abs((v1-v2)/2.))
	
	return [D2histos, D2edges, np.array(quantiles), np.array(err_q)]


#### CALCULATES PARTIAL SUM OF CERTAIN LIST BETWEEN i AND e INDEX #####
def sumFromTo(toSum, start, end):
	thisSum = 0.0
	for i in range(start, end):
		thisSum += toSum[i]
	return thisSum


#### WRITE TO TABLEFILE
#def write2table(filepath, key, value, opt=[]):
#	if os.path.exists(filepath) or "new" in opt:
#		if os.path.exists(filepath):
#			dictionary = pickle.load(open(filepath, "rb"))
#		else:
#			dictionary = {}
#		
#		if type(key) == str:
#			if key in dictionary:
#				if "hard" in opt:
#					dictionary[key] = value
#				else:
#					print "ML-ERROR: Not replacing dictionary entry cause permission failed."
#			else:
#				dictionary.update({key : value})
#		else:
#			if key[0] in dictionary:
#				if key[1] in dictionary[key[0]]:
#					dictionary[key[0]][key[1]] = value
#				else:
#					dictionary[key[0]].update({key[1] : value})
#			else:
#				dictionary.update({key[0] : {key[1] : value}})
#
#		pickle.dump(dictionary,open(filepath, 'wb'))
#	else:
#		print "ML-ERROR: Failed to open file: "+str(filepath)

	
#### WRITE TO DICTIONARY
def write2Dict(dictionary, key, value, opt=[]):
	if len(key) > 1:
		if key[0] in dictionary:
			if type(dictionary[key[0]]) == dict:
				dictionary = write2Dict(dictionary[key[0]],key[1:],value, opt=opt)
			else:
				dictionary[key[0]] = {key[1] : ""}
				dictionary[key[0]] = write2Dict(dictionary[key[0]],key[1:],value, opt=opt)
		else:
			dictionary.update({key[0]: {}})
			dictionary = write2Dict(dictionary[key[0]],key[1:],value, opt=opt)
	else:
		dictionary[key[0]] = value
	return dictionary
	
#### REMOVES FROM DICTIONARY
def removeFromDict(dictionary, key):
	if len(key) > 1:
		if key[0] in dictionary:
			if type(dictionary[key[0]]) == dict:
				dictionary = removeFromDict(dictionary[key[0]],key[1:])
			else:
				print "Failed level 2"
		else:
			print "Failed level 1"
	else:
		del dictionary[key[0]]
	return dictionary
	

#### WRITE TO TABLEFILE
def write2table(filepath, key, value, opt=[]):
	if os.path.exists(filepath) or "new" in opt:
		if os.path.exists(filepath):
			dictionary = pickle.load(open(filepath, "rb"))
		else:
			dictionary = {}
			
		if type(value) == int and value == -1:
			removeFromDict(dictionary, key)
		else:
			write2Dict(dictionary, key, value, opt=opt)
		pickle.dump(dictionary,open(filepath, 'wb'))
	else:
		print "ML-ERROR: Failed to open file: "+str(filepath)


#### WRITE SIGNIFICANCES TO TABLEFILE
def writeSignificances2table(name, key, value, opt=[]):
	path = localPath+"MCode/"
	write2table(path+name, key, value, opt=opt)
	
def writePickle(fileName, thing=None, nFig=-1, typ="fig"):
	"""
	- Saves object to pickle file
	- Give filename without .pickle
	- thing= object
	- if typ=fig: save figure
	"""
	if typ == "fig": 
		figure=plotControl(nFig)
		pickle.dump(figure, open(fileName+".pickle", 'wb'))
		print("Figure has been saved to: "+ fileName +".pickle")
	else:
		pickle.dump(thing, open(fileName+".pickle", "wb"))
		print "Object has been saved to: "+ fileName +".pickle"
	
def loadPickle(fileName):
	"""
	- Give filename without .pickle
	- Returns object
	"""
	print("Load file: "+fileName+".pickle")
	lpick=pickle.load(open(fileName+".pickle", 'rb'))
	return lpick

def load_IC86_11(basepath, exp_bool=False):    
	mc = np.load(os.path.join(basepath, "IC86-2011/baseline/dataset_10602_11077_11191.npy"))
	
	config = ConfigParser.ConfigParser()
	config.read(os.path.join(basepath, "IC86-2011_PS_Rene.cfg"))
	livetime = float(config.get("fit_settings", "livetime"))            # sec
	
	mc = mc[["ra", "dec", "logE", "sigma", "trueRa", "trueDec", "trueE", "ow", "conv", "prompt", "astro"]]
	mc = mc[np.where(mc["dec"]>0.)]    
	mc = append_fields(mc, "sinDec", np.sin(mc["dec"]), dtypes=np.float, usemask=False)
	
	if exp_bool==True:
		exp = np.load(os.path.join(basepath, "IC86-2011/data/dataset_ic86_burnsample_ic86_fulldata_wo_burnsample.npy"))
		exp = exp[["ra", "dec", "logE", "sigma"]]
		exp = exp[np.where(exp["dec"]>0.)]
		exp = append_fields(exp, "sinDec", np.sin(exp["dec"]), dtypes=np.float, usemask=False)
	else:
		exp = []
	
	return mc, exp	
	
def load_IC79(basepath, exp_bool=False, shuffle_bool=True):
	  
    #mc = np.load(os.path.join(basepath, "IC79/baseline/dataset_6308_6850.npy")) OLD            
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(basepath, "../DiffuseData/IC79_PS_Rene.cfg"))
    livetime = float(config.get("fit_settings", "livetime"))            # sec
    if exp_bool==False:
			mc = np.load(os.path.join(basepath, "IC79/baseline/IC79_6308_6850.npy"))
			mc = mc[mc["logE"] > 1.]   
			mc = mc[np.where(mc["dec"]>0.)]                                         # WARUM ?
			mc = mc[["ra", "dec", "logE", "sigma", "trueRa", "trueDec", "trueE", "ow", "conv", "prompt", "astro"]]    
			mc = append_fields(mc, "sinDec", np.sin(mc["dec"]), dtypes=np.float, usemask=False)
			exp = []
    elif exp_bool==True and shuffle_bool==True:
			exp = np.load(os.path.join(basepath, "IC79/data/IC79_data.npy"))
			#exp = np.load(os.path.join(basepath, "IC79/data/dataset_ic79_burnsample_ic79_fulldata_wo_burnsample.npy")) OLD
			exp = exp[["ra", "dec", "logE", "sigma"]]  
			exp = exp[exp["logE"] > 1.]  
			exp = exp[np.where(exp["dec"]>0.)]    
			exp = append_fields(exp, "sinDec", np.sin(exp["dec"]), dtypes=np.float, usemask=False)
			mc=[]
    elif exp_bool==True and shuffle_bool==False:
			print "READING REAL DATA"
			exp = np.load(os.path.join(basepath, "IC79/data/IC79_data_full_info_non_blind.npy"))
			exp = exp[["ra", "dec", "logE", "sigma"]]  
			exp = exp[exp["logE"] > 1.]  
			exp = exp[np.where(exp["dec"]>0.)]    
			exp = append_fields(exp, "sinDec", np.sin(exp["dec"]), dtypes=np.float, usemask=False)
			#print "IN WORK"
			mc=[]
    else:
			print "Parameter error :'("
    
    return mc, exp

#### READ SIGNIFICANCES FROM FILE
def readSignificancesFromTable(name = "smeared_significances.dict"):
	path = localPath+"MCode/"
	dictionary = pickle.load(open(path+name, "rb"))
	return dictionary

#### CONVERT ZENITH TO DECLINATION
#### NOTE: Converts zen to dec by (zen-np.pi/2.0) and turns map for the healpix coordinate system by np.pi-... .
#def zen2dec(zenith): 
	#dec = []
	#if type(zenith)==float or type(zenith)==np.float64:
		#declin=(np.pi/2. - (zenith-np.pi/2.0)) #gives healpy coordinate in rad
	#else:
		#for zen in zenith:
			#dec.append(np.pi/2. - (zen-np.pi/2.0)) #gives healpy coordinate in rad
		#declin=np.array(dec)
	#return declin
	
#def zen2dec_noticecube(zenith): 
	#dec = []
	#if type(zenith)==float or type(zenith)==np.float64:
		#declin=(np.pi/2. - zenith) #gives healpy coordinate in rad
	#else:
		#for zen in zenith:
			#dec.append(np.pi/2. -zen) #gives healpy coordinate in rad
		#declin=np.array(dec)
	#return declin

#def dec2zen_noticecube(dec):
	#"""
	#gives zen in rad from dec in rad, proper coordinates...not stupid icecube ones ;-)
	#"""
	#zen = []
	#if type(dec)==float or type(dec)==np.float64:
		#zeni=(np.pi/2. - dec) #gives healpy coordinate in rad
	#else:
		#for d in dec:
			#zen.append(np.pi/2. - dec) #gives zen in rad
		#zeni=np.array(zen)
	#return zeni
	
#def dec2zen(dec):
	#"""
	#gives zen in rad from dec in rad
	#"""
	#zen = []
	#if type(dec)==float or type(dec)==np.float64:
		#zeni=(np.pi/2. + dec) #gives healpy coordinate in rad
	#else:
		#for d in dec:
			#zen.append(np.pi/2. + dec) #gives zen in rad
		#zeni=np.array(zen)
	#return zeni
	
def zen2dec_noticecube(zenith): 
	dec = []
	if type(zenith)==float or type(zenith)==np.float64:
		declin=(np.pi - zenith) #gives healpy coordinate in rad
	else:
		for zen in zenith:
			dec.append(np.pi -zen) #gives healpy coordinate in rad
		declin=np.array(dec)
	return declin	
	
def zen2dec(zenith): 
	dec = []
	for zen in zenith:
		dec.append(np.pi/2. - (zen-np.pi/2.0)) #gives healpy coordinate in rad
	return np.array(dec)

def dec2zen_noticecube(dec):
	"""
	gives zen in rad from dec in rad, proper coordinates...not stupid icecube ones ;-)
	"""
	zen = []
	for d in dec:
			zen.append(-d+np.pi/2.) #gives zen in rad
	return np.array(zen)
	
def dec2zen(dec):
	"""
	gives zen in rad from dec in rad
	"""
	zen = []
	for d in dec:
			zen.append(d+np.pi/2.) #gives zen in rad
	return np.array(zen)	

#### GETS 5 LARGEST SOURCES FROM A PDF USING A SPLINE
def getLargestSources(edges, histo, NSources, n=1):
	summed = 0.0
	edgeAbove = len(histo)
	for i in np.arange(len(histo)-1, -1, -1):
		if summed+NSources*histo[i] <= n*1.0:
			summed += NSources*histo[i]
			edgeAbove = i
			#print str(summed)+" index: "+str(i)
		else:
			break
	border = edgeAbove*1.0-0.5 - (1.0*n-summed)/(histo[edgeAbove-1]*NSources)
	return border


##### EXTRACTS PARAMETERS FROM OPTIONS
def getParameters(connect=False):
	diction = {}
	for i in range(0,len(sys.argv)):
		 if len(str(sys.argv[i]).split("=")) > 1:
		 	diction.update({str(sys.argv[i]).split("=")[0]: str(sys.argv[i]).split("=")[1]})
	if connect:
		conn = " "
		for i in range(0,len(sys.argv)):
			if len(str(sys.argv[i]).split("=")) > 1:
		 		conn += str(sys.argv[i])+" "
		return conn
	else:	
		return diction	


#### CALCULATES A WEIGHTING FACTOR FOR A SMEARED EVENT USING THE ENERGY
def getLikelihoodE(splineSig, splineBg, E, log=True):
	fac = []
	a = 0
	b = 0
	c = 0 
	for i in range(0, len(E)):
		if splineSig(E[i]) < 0.0:
			a += 1
			print "ML-ERROR: SIGNAL LIKELIHOOD "+str(splineSig(E[i]))+"< 0.0 FOR E="+str(E[i])+" AT i="+str(i)+" => CHECK SPLINES!"
			fac.append(0.0)
		elif splineBg(E[i]) <= 0.0:
			b += 1
			this = 1+splineSig(E[i])*(0.000009)
			if log: fac.append(np.log10(this))
			else: fac.append(this)
		else:
			this = 1+splineSig(E[i])*1.0/splineBg(E[i])
			if log: fac.append(np.log10(this))
			else: fac.append(this)
			c += 1

	print "Found: "+str(a)+" of negative sig-spline, "+str(b)+" of negative bg-spline, "+str(c)+" ok."
	return fac
	
	
#### CALCULATE NEW WEIGHTING FACTOR ----LJS----

def getLikelihoodE2(splineSig, splineBg, E, nTot, nSig):
	fac = []
	a = 0
	b = 0
	c = 0 
	for i in range(0, len(E)):
		if splineSig(E[i]) < 0.0:
			a += 1
			print "ML-ERROR: SIGNAL LIKELIHOOD "+str(splineSig(E[i]))+"< 0.0 FOR E="+str(E[i])+" AT i="+str(i)+" => CHECK SPLINES!"
			fac.append(0.0)
		elif splineBg(E[i]) <= 0.0:
			b += 1
			this = splineSig(E[i])*nSig*1.0/(0.00001*(nTot-nSig)+nSig*splineSig(E[i])) ##????
			fac.append(this)
		else:
			this = splineSig(E[i])*nSig*1.0/(splineBg(E[i])*(nTot-nSig)+nSig*splineSig(E[i]))
			fac.append(this)
			c += 1

	print "Found: "+str(a)+" of negative sig-spline, "+str(b)+" of negative bg-spline, "+str(c)+" ok."
	return fac

#### NORMS A HISTOGRAM CONSISTING OF MULTIPLE STACKED DATA SETS ####
def norm_multi_hist(ndata, useFac=False, fac=1.):
    if useFac:
        sumVal=fac
    else:
        sumVal = sum(ndata[0])
    ndata_normed = ndata/sumVal
    return ndata_normed, sumVal
        
def rescaleHist(hist, fac=1.0, useFac=False):
    nHist = []
    if not useFac:
        sumVal = sum(hist)
    else:
        sumVal = fac
    for i in range(0,len(hist)):
        nHist.append(float(hist[i]*1.0)/(float(1.0*sumVal)))
    return nHist, sumVal

def poisson(x, *p):
    lamb, norm = p
    return (lamb**x/factorial(x)) * np.exp(-lamb)*norm
    
def set_yticks(nFigure, start, stop, num, lim=True, decimals=4):
    plt.figure(nFigure)
    if lim:
        plt.ylim([start*0.9, stop*1.1])
    ticks = np.around(np.linspace(start, stop, endpoint=True, num=num), decimals)
    plt.yticks(ticks, ticks)

#### GET Quantile FROM SPLINE INTEGRAL
def quantileOfSpline(spline, border1, border2, quantile, pres=0.001):
	integral = spline.integral(border1, border2)
	if integral <= 0: print "WARNING: QUANTILE OF +- SPLINES IS NOT CORRECTLY DETERMINABLE!"
	curr = 0.0
	stepSize=0.1
	tooLow = False
	while(True):
		curInt = spline.integral(border1, curr*border2)
		if abs((curInt-quantile*integral)/(quantile*integral)) < pres:
			return curr*border2
		else:
			if (curInt-quantile*integral) < 0:
				if tooLow == False: 
					tooLow = True
					stepSize = stepSize*1.0/10
				else: curr = curr+stepSize
			else: 
				if tooLow == True: 
					tooLow = False
					stepSize = stepSize*1.0/10
				else: curr = curr-stepSize
			

##### MOVE LEGEND LABELS #####
def reorderLegendLabels(subfig, order):
	
	handles, labels = subfig.get_legend_handles_labels()
	nLabels = []
	nHandles = []
	
	for i in order:
		nLabels.append(labels[i])
		nHandles.append(handles[i])
		
	subfig.legend(nHandles, nLabels)
	return nHandles, nLabels
	
#### CREATE STRING FOR MU_SOURCES ###
def createMuString(MU_SOURCES):
	if MU_SOURCES == int(MU_SOURCES): 
		MU_SOURCES = int(MU_SOURCES)
		MU_STRING = str(MU_SOURCES)
	else:
		MU_STRING = str(int(MU_SOURCES))+"_"+str(int((MU_SOURCES-int(MU_SOURCES))*10))
	return MU_STRING
	
def memory_usage_psutil():     # return the memory usage in MB  ## inspired from http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
	process = psutil.Process(os.getpid())
	mem = process.memory_info()[0] / float(2 ** 20)
	return mem
	
def spetoca(spe): ## from spherical to cartesian
	x=np.cos(spe[0])*np.sin(spe[1])
	y=np.sin(spe[0])*np.sin(spe[1])
	z=np.cos(spe[1])
	return [x,y,z]

###rotate a vector with coordinates ra, dec around another vector with coordinates true_ra, true_dec around an angle of alpha
def rotate_to_valid_angle(true_ra, true_dec, ra, dec, alpha): 
    true_phi = true_ra
    true_zen = np.pi/2. - true_dec
    phi = ra
    zen = np.pi/2. - dec
    true_cart=spetoca([true_phi, true_zen])
    cart=spetoca([phi, zen])
    res=np.zeros(3)
    #alpha=np.random.uniform(0., 2*np.pi, 1)
    f=1-np.cos(alpha)
    cos=np.cos(alpha)
    sin=np.sin(alpha)
    res[0]=(true_cart[0]**2*f+cos)*cart[0]+(true_cart[0]*true_cart[1]*f-true_cart[2]*sin)*cart[1]+(true_cart[0]*true_cart[2]*f+true_cart[1]*sin)*cart[2]
    
    res[1]=(true_cart[1]*true_cart[0]*f+true_cart[2]*sin)*cart[0]+(true_cart[1]**2*f+cos)*cart[1]+(true_cart[1]*true_cart[2]*f-true_cart[0]*sin)*cart[2]
    
    res[2]=(true_cart[2]*true_cart[0]*f-true_cart[1]*sin)*cart[0]+(true_cart[2]*true_cart[1]*f+true_cart[0]*sin)*cart[1]+(true_cart[2]**2*f+cos)*cart[2]
    
    ra_ret=m.atan2(res[1],res[0])
    dec_ret=np.pi/2-m.acos(res[2])
    if ra_ret<0.0:
        ra_ret=ra_ret+2*np.pi
    res_sphe=[ra_ret, dec_ret]
    return res_sphe

## Recursive Search for Folders containing Files with the given Project_name  
def recfinddir(subdir, Project_Name):
    ret_list=[]
    dir_list=os.listdir(subdir)
    dir_list=[i for i in dir_list if os.path.isdir(os.path.join(subdir+"/"+i))==True and i!='logs' and i!='jobs' and i!='data' and i!='plots']
    rem=[i for i in dir_list if Project_Name in i]
    for i in dir_list:
        path=os.path.join(subdir+"/"+i)
        temp=recfinddir(path, Project_Name)
        if len(temp)>0 and 'CPU' in path:
            if "Mu" in path:
                if int(path[path.index("NSou")+4:len(path)])>0:
					if len(NNeutrinos)==0:
						NNeutrinos.append(path[path.index("Ev")+2:path.index("LMAX")])
					NSou=path[path.index("NSou")+4:len(path)]
					Mu=path[path.index("Mu")+2:path.index("NSou")]
					fixN_ind=[i for i,k in enumerate(temp) if 'fixN' in k]
					if not (len(temp)==len(fixN_ind)):
						collect_mu.append(Mu)
						collect_N.append(NSou)
					if len(fixN_ind)>0:
						for l in fixN_ind:
							NFixed.append(NSou)
							NFixed_mu.append(Mu)
							NFixed_pos.append(temp[l][len(temp[l])-5:len(temp[l])-4])		
                else:
					collect_BG.append(temp[0][temp[0].index("PROJECT")+8:len(temp[0])])              
            elif "NSou" in path:
                if int(path[path.index("NSou")+4:len(path)])>0:
                    collect_pS_N.append(path[path.index("NSou")+4:len(path)])
            else:
				collect_BG.append(temp[0][temp[0].index("PROJECT")+8:len(temp[0])])
            
        ret_list=ret_list+temp
    ret_list+=rem
    return ret_list
        
##crawl through all simulated files and create information file for a given Project_Name
def create_smart_setup(Detector_setup, Project_Name, GAMMA ):
	global collect_mu
	collect_mu=[]
	global collect_N
	collect_N=[]
	global collect_pS_N
	collect_pS_N=[]
	global collect_BG
	collect_BG=[]
	global NNeutrinos
	NNeutrinos=[]
	global NFixed
	NFixed=[]
	global NFixed_pos
	NFixed_pos=[]
	global NFixed_mu
	NFixed_mu=[]	
	recfinddir(localPath+"Analysis_stuff/condor/1Y_Skymaps_MCgen_useE_"+Detector_setup+"_E"+str(GAMMA)+"_useDiff", Project_Name )
	if os.path.exists(localPath+"Analysis_stuff/condor/"+Detector_setup+"_Skymaps_Experiment_MCgen_useE_1111_E"+str(GAMMA)+"_useDiff"):
		recfinddir(localPath+"Analysis_stuff/condor/"+Detector_setup+"_Skymaps_Experiment_MCgen_useE_1111_E"+str(GAMMA)+"_useDiff", Project_Name)
	numbers=[i for k, i in enumerate(collect_mu) if i not in collect_mu[:k]]
	NFix_pos=[i for k, i in enumerate(NFixed_pos) if i not in NFixed_pos[:k]]
	NFix_num=[i for k, i in enumerate(NFixed) if i not in NFixed[:k]]
	
	output=dict()
	for i in numbers:
	    temp=[l for k,l in enumerate(collect_N) if collect_mu[k]==i]
	    output[i]=[str(k) for k in sorted([int(s) for s in temp])]
	for i in NFix_pos:
		tempN=dict()
		for j in NFix_num:
			tempmu=[]
			tempmu=[l for k,l in enumerate(NFixed_mu) if NFixed_pos[k]==i and NFixed[k]==j]
			tempN[j]=sorted(tempmu)
		output[i]=tempN
		
	output["PS"]=collect_pS_N
	output["BG"]=collect_BG
	output["Mus"]=numbers
	output["NNeutrinos"]=NNeutrinos
	datapath = open(localPath+"Analysis_stuff/condor/"+Detector_setup+Project_Name+".pkl", 'w+b')
	pickle.dump(output, datapath)
	print output

def smart_setup(Project_Name, Type, Detector_setup, GAMMA, Mu=-1, NSou=-1, Pos=-1, fit_func="gaussexpconvolv", reset=False):
		
		
	if not os.path.exists(localPath+"Analysis_stuff/condor/"+Detector_setup+Project_Name+".pkl"):
		print "There is no set-up file available for the given project.....Trying to run create smart setup in order to create it. This might take a little while.." 	
		sys.stdout.flush()
		create_smart_setup(Detector_setup, Project_Name, GAMMA)	
	data=np.load(localPath+"Analysis_stuff/condor/"+Detector_setup+Project_Name+".pkl")	
	BG_index=-1
	for i,k in enumerate(data['BG']):
		if "BG" in k:
			BG_index=i
			break
	print "Set Up File loaded..."
	if Type=="PureSig":
		if BG_index==-1: ### Project does not have BG File.		
			return ["Sig_N"+i+Project_Name for i in data['PS']] , [int(j) for j in data['PS']] , data['BG'][BG_index], data['NNeutrinos']
		else:
			return ["Sig_N"+i+Project_Name for i in data['PS']] , [int(j) for j in data['PS']] , "", data['NNeutrinos']
						
	if Type=="VarSig":
		if str(Mu)=="-1":
			print "No Mu given....Attempting to take next unprocessed Mu from list"
			path = localPath+"MCode/unsmeared_significances_1Samples_delta_MCgen_useE_"+Detector_setup+"_E"+str(GAMMA)+"_useDiff_zs_normL1000"+Project_Name+"_"+fit_func+".dict"
			print path
			if os.path.exists(path):
				dictionary = pickle.load(open(path, "rb"))
				for i in data["Mus"]:
					if 'Mu'+str(float(i.replace("_", "."))) in dictionary:
						print "Mu"+i+" already processed"
					else:
						Mu=i
						print "Mu="+i+" chosen"
						if BG_index==-1: ### Project does not have BG File.	
							return Mu,["Sig_N"+i+Project_Name for i in data[Mu]]+["Sig_N0"+Project_Name], data[Mu]+['0'], data['NNeutrinos'], ""
						else:
							return Mu,["Sig_N"+i+Project_Name for i in data[Mu]]+["Sig_N0"+Project_Name], data[Mu]+['0'], data['NNeutrinos'], data['BG'][BG_index]
				print "Error: No Mu to process. You could try to set one manually using argument Mu=something you wanna do."
				return False
			else:
				print " Uiii. It's the beginning of a new Project, hm? :)"
				Mu=data["Mus"][0]
				if BG_index==-1: ### Project does not have BG File.	
					return Mu,["Sig_N"+i+Project_Name for i in data[Mu]]+["Sig_N0"+Project_Name], data[Mu]+['0'], data['NNeutrinos'], ""
				else:
					return Mu,["Sig_N"+i+Project_Name for i in data[Mu]]+["Sig_N0"+Project_Name], data[Mu]+['0'], data['NNeutrinos'], data['BG'][BG_index]
		else:
			print data
			if BG_index==-1:
				return Mu,["Sig_N"+i+Project_Name for i in data[str(Mu)]]+["Sig_N0"+Project_Name], data[str(Mu)]+['0'], data['NNeutrinos'], ""
			else:
				return Mu,["Sig_N"+i+Project_Name for i in data[str(Mu)]]+["Sig_N0"+Project_Name], data[str(Mu)]+['0'], data['NNeutrinos'], data['BG'][BG_index]				
	if Type=="Sensitivity":
		summation=[]
		for i in data["Mus"]:
			summation+=data[i]
		return data["Mus"], summation 
	if Type=="CompareExpBG":
		return 0,0,0, data['NNeutrinos'], data['BG'][BG_index]
	if Type=="InfoPlotter":
		return data['NNeutrinos']
	if Type=="FixedNFixedPos":
		if NSou==-1 and Pos==-1:
			print " Auto-Setup is not implemented yet...and probably never will. So please just manually set an N and a Position from the name convention that you have chosen"
		else:
			return data[Pos][str(NSou)]+['-1'], ["Sig_N"+str(NSou)+Project_Name+"_"+Pos+"fixN"]+["Sig_N0"+Project_Name] , [NSou], data['NNeutrinos'] ,data['BG'][0], Pos
	if Type=="PureSigMus":
			return [int(j) for j in data['PS']]
