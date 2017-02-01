############################ IMPORTS ####################################################
import healpy as H
import numpy as np
import cPickle as cPickle
import time as tm
import sys, os, math
import copy
import cmath
import numpy.lib.recfunctions
from scipy.signal import convolve2d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import moment, t
from random import shuffle
from injector_stand_alone import PointSourceInjector2
import scipy
from functions4 import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib.cm as cm
import numpy.ma as ma

import socket
print "running on machine: "+str(socket.gethostname())	

#if (not 'physik.rwth-aachen.de' in socket.gethostname() or ('cuda' in socket.gethostname())) and not "Ubuntu" in socket.gethostname():
	#import pycuda.driver as cuda
	#import pycuda.autoinit
	#from pycuda.compiler import SourceModule
	#print "SKUA IS ABOUT TO BE USED!"


if socket.gethostname()=='theo-UL50VT':
        localPath="/home/theo/fusessh/"
else:
        localPath = "/net/scratch_icecube4/user/glauch/"

############################ CLASSES ##############################################################

class MergeAnalysis:
	##CONSTRUCTOR##
	def __init__(self,  theta_src=[] , phi_src=[], skymaps=[], nside=512,nlmax=10):
		if len(skymaps)>0:
			self.map_delta=skymaps[0].map_delta
			for i in range(1,len(skymaps)):
				self.map_delta=[x + y for x, y in zip(self.map_delta , skymaps[i].map_delta)]
		else:
			print "Big Error! No Skymaps given!"
			
		print "Maximum Value in the Array bigger zero?" + str(np.max(self.map_delta))

	#variables
		self.l_max=skymaps[0].l_max  ### l_max should be the same for all maps
		self.cl_means=[]
		self.npix=H.nside2npix(nside)
		self.theta_sou=theta_src
		self.phi_sou=phi_src
		self.cl_errors=[]
		self.effCl_means=[]
		self.effCl_errors=[]
		self.cl_all=[]
		self.cl_auto_all=[]
		self.cl_auto_cat=[]
		self.al0_all=[]
		self.effCl_all=[]
		self.useAlm=True
		self.alm_all=[]
		self.catalog=""
		self.NUMBER_RAN_RUNS	=	1
		self.SAVE_ALL_DATA = False
		self.SAVE_Cls = True
		self.useClLog=False
		if self.useAlm == True:
			self.alm_all = []
			self.alm_all_abs = []
			self.alm_all_phase = []			
			#if nlmax > 8:
			#	self.almLimit = 8 # Maximal l (used for saving alms!)
			#else:
			self.almLimit = self.l_max
	def set_catalogmap(self, catalog_map):
		self.catalog_map=catalog_map

	def set_catalog(self, catalog):
		self.catalog=catalog

	def saveCleffClList(self, nTitle="", prefix=""):
		if self.SAVE_ALL_DATA:
			np.savetxt(nTitle+"clAll_"+prefix+".txt", self.cl_all) ### Cl
			print "Cl data successfully saved to... "+nTitle+"_clAll.txt"
			np.savetxt(nTitle+"_al0All_"+str(self.NUMBER_RAN_RUNS)+".txt", self.al0_all)
			#~ print "al0 data successfully saved to... "+nTitle+"_al0All.txt"
			
		np.savetxt(nTitle+"effClAll_"+prefix+".txt", self.effCl_all)
		print "effCl data successfully saved to... "+nTitle+"_effClAll.txt"
		#self.saveSignalInfo(nTitle)

	def saveCls(self, nTitle="", prefix=""):
		np.savetxt(nTitle+"ClAll_"+prefix+".txt", self.cl_all)
		print "Cl data successfully saved to... "+nTitle+"ClAll_"+prefix+".txt"
		np.savetxt(nTitle+"ClAutoAll_"+prefix+".txt", self.cl_auto_all)
		print "Cl data successfully saved to... "+nTitle+"ClAutoAll_"+prefix+".txt"
	
	def saveCatCl(self, nTitle=""):
		#if not os.path.exists(nTitle+"CatCl.txt"): 
		np.savetxt(nTitle+"CatCl.txt", self.cl_auto_cat)
		print "Cl data successfully saved to... "+nTitle+"CatCl.txt"

	def analyseMap(self):
		## generate alm, Cl, effCl
		if self.catalog=="NVSS":
			print "Calculate Cross Correlations for NVSS Sky Survey"
			cl_delta, alm_delta, alm_delta1 = H.anafast(self.map_delta, self.catalog_map,  lmax=self.l_max, alm=True) 
			cl_auto_delta, alm_auto_delta = H.anafast(self.map_delta, lmax=self.l_max, alm=True)	# map auto-correlation
			cl_auto_cat, alm_auto_cat = H.anafast(self.catalog_map, lmax=self.l_max, alm=True)	# cat auto-correlation
			#effClcur = retEffCl([alm_delta, alm_delta1], lmax=self.l_max)
		else:
			cl_auto_delta, alm_auto_delta = H.anafast(self.map_delta, lmax=self.l_max, alm=True)	
			#effClcur = retEffCl([alm_delta], lmax=self.l_max)
			
		if self.SAVE_ALL_DATA:			
			al0curr = absList(getAl0List(alm_delta, self.l_max))
			
		if self.useClLog == True:
			cl_log_delta = calcClLogFromAlm(alm_delta, 400)
			self.cl_log_all.append(cl_log_delta)
		if self.useAlm == True:
			alm_delta = H.anafast(self.map_delta, lmax=self.almLimit, alm=True)[1]

		## save alm, Cl, effCl
		if self.SAVE_ALL_DATA:
			self.saveFirstAlms(alm_delta)
			self.cl_all.append(cl_delta)
			self.al0_all.append(al0curr)
		
		if self.SAVE_Cls:
			if self.catalog=="NVSS":
				self.cl_all.append(cl_delta)
			self.cl_auto_all.append(cl_auto_delta)
			if len(self.cl_auto_cat)==0:
				self.cl_auto_cat=cl_auto_cat

		#self.effCl_all.append(effClcur)
		
		if self.useAlm == True:
			self.alm_all.append(alm_delta)
			self.alm_all_abs = []
			self.alm_all_im = []
			for i in range(0,len(self.alm_all)):
				self.alm_all_abs.append([])
				self.alm_all_im.append([])
				self.alm_all_phase.append([])
				for k in range(0, len(self.alm_all[i])):
					self.alm_all_abs[i].append(abs(self.alm_all[i][k]))#[k]
					self.alm_all_im[i].append(np.imag(self.alm_all[i][k]))
					self.alm_all_phase[i].append(np.angle(self.alm_all[i][k]))	
		else:
			print "***"
			print "WARNING!! alm NOT saved!!!"
			print "***"
						
                        
	def saveAlms(self, nTitle="", prefix=""): 
		
		# Commented code appends the values to the same file:
		### absolute values
		#dataSource_abs=np.DataSource("/.automount/net_rw/net__scratch_icecube4/user/kalaczynski/Analysis_stuff/"+nTitle+"almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt")
		#if(dataSource_abs.exists):
			#with open(nTitle+"almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt",'a') as f_handle:
				#np.savetxt(f_handle,absList(self.alm_all))
			#print "Alm data successfully added to... "+nTitle+"almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt"
		#else:
			#np.savetxt(nTitle+"almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt", absList(self.alm_all))
			#print "Alm data successfully saved to... "+nTitle+"almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt"
			
		np.savetxt(nTitle+"almAllAbs_"+prefix+".txt", absList(self.alm_all))
		print "Alm data successfully saved to... "+nTitle+"almAllAbs_"+prefix+".txt"
		
		# this one as well:
		### phases
		#dataSource_phase=np.DataSource(nTitle+"almAllPhase_"+str(self.NUMBER_RAN_RUNS)+".txt")
		#if(dataSource_phase.exists):
			#with open(nTitle+"almAllPhase_"+str(self.NUMBER_RAN_RUNS)+".txt",'a') as f_handle:
				#np.savetxt(f_handle,phaseList(self.alm_all))
			#print "Alm data successfully added to... "+nTitle+"almAllPhase_"+str(self.NUMBER_RAN_RUNS)+".txt"
		#else:
			#np.savetxt(nTitle+"almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt", phaseList(self.alm_all))
			#print "Alm data successfully saved to... "+nTitle+"almAllPhase_"+str(self.NUMBER_RAN_RUNS)+".txt"
			
		np.savetxt(nTitle+"almAllPhase_"+prefix+".txt", phaseList(self.alm_all))
		print "Alm data successfully saved to... "+nTitle+"almAllPhase_"+prefix+".txt"
		
        
	def calcMeans(self, alm=False, log=False):
		if self.SAVE_ALL_DATA:
			self.cl_means = np.mean(self.cl_all, axis=0)
			self.cl_errors = np.std(self.cl_all, axis=0)

		self.effCl_means = np.mean(self.effCl_all, axis=0)
		#~ print "bla"
		self.effCl_errors = np.std(self.effCl_all, axis=0)
	
	def saveSignalInfo(self, nTitle):
		np.savetxt(nTitle+"_signalInfo_"+str(self.NUMBER_RAN_RUNS)+".txt", self.generatedSigEvents) ### Cl
		#~ print "detailed signal information successfully saved to... "+nTitle+"_signalInfo_"+str(self.NUMBER_RAN_RUNS)+".txt"
		#~ np.savetxt(nTitle+"_input_"+str(self.NUMBER_RAN_RUNS)+".txt", self.input_parameters) ### Cl
		#~ print "input parameters successfully saved to... "+nTitle+"_input_"+str(self.NUMBER_RAN_RUNS)+".txt"
	
	
class multiPoleAnalysis:
	#### CONSTRUCTOR ###
	def __init__(self,nEvents, nReadDir, nSaveDir, detector, nNEvents, nlmax=1000, nPSSmearingPath="", nNRanRuns=1):

		self.NUMBER_SIMULATED_EVENTS = nNEvents
		self.NUMBER_RAN_RUNS = nNRanRuns
		self.l_max = nlmax
		self.pltColor = "k"
		self.DETECTOR=detector

		self.ls = [i for i in range(0, self.l_max+1)]
		self.lsEff = [i for i in range(1, self.l_max+1)]
		self.lsLog = [i for i in range(1,400+1)]
		self.SAVE_DIR = nSaveDir
		self.READ_DIR = nReadDir
		self.GAMMA=0.
		self.GAMMA_BF,_=get_best_fit_values()
		self.cl_all=[]
		self.cl_auto_all=[]
		self.effCl_all = []
		self.cl_log_all = []
		self.al0_all = []
		self.firstAlms_abs = []
		self.firstAlms_phi = []
		self.failedFileRead = 0
		self.emptyfile=0
		self.successFileRead = 0
		self.failedFileNumber = []
		self.generatedSigEvents = []
		self.MuReal_all = []
		self.wrongHemi_all = []
		self.l_show = 3
		self.l_save = 8 # 8
		self.useAlm = True
		self.SAVE_Cls = True
		if self.useAlm == True:
			self.alm_all = []
			self.alm_all_abs = []
			self.alm_all_phase = []			
			if nlmax > 20:
				self.almLimit = 20 # = 20
			else:
				self.almLimit = nlmax
		self.resolutionSmearing = 2.0 ## 2 Degrees for quick smearing
		self.cosResSmearing = np.cos(self.resolutionSmearing*np.pi/180.0 )
		self.smearingMethod = "none"
		self.centerHits = True
		self.renormSkymaps = True
		self.fullSphereMode = False
		self.useClLog = False
		self.useSqrtEffCl = True
		self.MuPrecise = False
		self.MilagroAzimuth = False
		self.FixedZenith = False
		self.GalacticPlane = False
		self.multiSourceBool = False
		self.RA_ACC = False
		
		self.MC_GEN_BOOL = False
		
		
		self.PSSmearingPath = nPSSmearingPath
		if self.PSSmearingPath != "":
			self.histoSmearing, self.edgesSmearing = np.genfromtxt(str(self.PSSmearingPath), unpack=False)
			self.smearingSpline = interpolate.InterpolatedUnivariateSpline(self.edgesSmearing, self.histoSmearing, k=1)
		self.sigCheck = []
		
		self.evPerDetector = [0,0,0]
		self.RA_ACC_rejectBG = 0 
		self.RA_ACC_rejectSIG = 0
		
		self.useE = False
		self.useDiffBG=False
		self.ZEN_BAND = (-1., 0.)
		self.energyList = []		
		self.diffElist=[]
		self.Emin_GeV = np.log10(10.0)
		self.Emax_GeV = np.log10(1000000000.0)
		self.nColor=["0.25", "0.5", "0.75"]


	def setAtmZenithFile(self, nZenithFilePath, nEvents):
		if nZenithFilePath != "":
			self.zenithFilePath = nZenithFilePath
			sumHi = []
			for i in range(0, len(nZenithFilePath)):
				hi, ed = np.genfromtxt(str(self.zenithFilePath[i]), unpack=False)
				sumHi.append(nEvents*np.array(hi)*1.0/sum(hi))
				
			avHi = []
			for k in range(0, len(ed)):	
				avHi.append(0.0)
				for i in range(0, len(nZenithFilePath)):
						avHi[k] += sumHi[i][k]
				
			self.histoZenithDistrAtm = normHist(avHi, 0.95)
			self.edgesZenithDistrAtm = ed
			self.atmosSpline = interpolate.InterpolatedUnivariateSpline(self.edgesZenithDistrAtm, self.histoZenithDistrAtm)
		else:
			print "Path to Zenith distribution is missing"		
			
	#### WARNS IF FULLSPHERE IS USED ####		
	def warnIfFullsphere(self):
		if self.fullSphereMode:
			print "WARNING: Using Full-Sphere-Mode instead of Half-Sphere-Mode to generate skymaps!"
	
	#### SETTING nside ####
	def setNside(self, nNside):
		self.nside = nNside
		self.npix = H.nside2npix(self.nside)	
		
	#### SETTING classicalSmearing ####
	def setSmearingMethod(self, nSmear):
		self.smearingMethod = nSmear	
					
	#### SETTING Renormalization ####
	def setRenormalization(self, ren):
		self.renormSkymaps = ren	
		
	#### SETTING Whether to use the galactic plane or not
	def setGalPlaneSwitch(self, galPlane, onlyGalPlane):
		self.useGalPlane = galPlane
		self.onlyGalPlane = onlyGalPlane
		
	#### SETTING UseClLog ####
	def setUseClLog(self, ren):
		self.useClLog = ren	
		
	#### SETTING sqrt(effCl) CALC ####
	def setUseSqrtEffCl(self, ren):
		self.useSqrtEffCl = ren	
		
	#### SETTING Hit centering ####
	def setCenterHits(self, nCenter):
		self.centerHits = nCenter	

	#### SETTING NUMBER OF RUNS
	def setRuns(self, nRuns):
		self.NUMBER_RAN_RUNS = nRuns

	#### SETTING almLimit OF RUNS
	def setAlmLimit(self, nlim):
		self.almLimit = nlim
		
	#### SETTING NUMBER OF CHANGED RUNS
	def setRunsChanged(self, nRuns):
		self.NUMBER_RAN_RUNS_CHANGED = nRuns
		
	#### SET FULL SPHERE MODE ON OR OFF
	def setFullSphere(self, onOff):
		self.fullSphereMode = onOff

	#### SETTING THE useAln BOOLEAN VALUE
	def setUseAlm(self, boolean):
		self.useAlm = boolean
		
	#### SETS samAeffControl  #### LJS
	def setSamAeffContr(self, nSamContr):		
		self.samAeffContr = nSamContr
		
	#### Sets Signal Zenith Spline #### LJS
	def setSigZenith(self, sigZenPath):		
		self.histoZenithDistrAtm_ORIGINAL = self.histoZenithDistrAtm[:]
		self.signalZenithPath = sigZenPath
		
		self.histoZenithDistrSig 	= []
		self.edgesZenithDistrSig 	= []
		self.signalSpline 			= []
		for i in range(0, len(sigZenPath)):
			histoZenithDistrSig, edgesZenithDistrSig = np.genfromtxt(str(self.signalZenithPath[i]), unpack=False)
			histoZenithDistrSig = normHist(histoZenithDistrSig, 0.95)
			signalSpline = interpolate.InterpolatedUnivariateSpline(edgesZenithDistrSig, histoZenithDistrSig)
		
			self.histoZenithDistrSig.append(copy.copy(histoZenithDistrSig))
			self.edgesZenithDistrSig.append(copy.copy(edgesZenithDistrSig))
			self.signalSpline.append(copy.copy(signalSpline))
		#print("Set Signal Zenith Spline complete!")
		
	def setUseDiffBG(self, useDiffBG):
		self.useDiffBG=useDiffBG

	def setColor(self, nColor):
		self.nColor=nColor

	#### RESETS ALL MULTIPOLE LISTS
	def resetAllLists(self):
		self.cl_all = []
		self.alm_all = []
		self.effCl_all = []
		self.cl_log_all = []
		self.al0_all = []
		self.failedFileRead = 0
		self.successFileRead = 0
		self.generatedSigEvents = []
		self.firstAlms_abs = []
		self.firstAlms_phi = []
		self.alm_all_abs = []
		self.alm_all_phase = []
		self.MuReal_all = []
		self.energyList = []
		self.diffElist=[]
		self.wrongHemi_all = []
		self.weight_astro = []
		self.weight_conv = []
	
	def setGAMMA(self, Gamma):
		self.GAMMA=Gamma

	#### BACKUPS alms UP TO l_show
	def saveFirstAlms(self, alm_delta):
		firstAlms_abs = []
		firstAlms_phi = []
		for i in range(0, self.l_save+1):
			firstAlms_abs.append([])
			firstAlms_phi.append([])
			for k in range(0, i+1):
				firstAlms_abs[i].append(np.absolute(alm_delta[H.sphtfunc.Alm.getidx(self.almLimit, i,k)]))
				firstAlms_phi[i].append(np.angle(alm_delta[H.sphtfunc.Alm.getidx(self.almLimit, i,k)]))
		self.firstAlms_abs.append(firstAlms_abs)
		self.firstAlms_phi.append(firstAlms_phi)
	
	
	#### BACKUPS alms UP TO l_show
	def saveFirstAlmsFromAbsPhase(self, l_data):
		for this in range(0, len(self.alm_all_abs)): ### loop over all files
			firstAlms_abs = []
			firstAlms_phi = []
			for i in range(0, self.l_save+1): ### loop over l
				firstAlms_abs.append([])
				firstAlms_phi.append([])
				for k in range(0, i+1): ### loop over m
					firstAlms_abs[i].append(self.alm_all_abs[this][H.sphtfunc.Alm.getidx(l_data, i,k)])
					firstAlms_phi[i].append(self.alm_all_phase[this][H.sphtfunc.Alm.getidx(l_data, i,k)])
			self.firstAlms_abs.append(firstAlms_abs)
			self.firstAlms_phi.append(firstAlms_phi)
	
	
	#### PLOT INFOS TO SINGLE ALM WITH l AND m
	def plotSingleAlm(self, l,m, nFigure=-1):
		alm_abs = copy.deepcopy(list(self.firstAlms_abs))
		alm_phi = copy.deepcopy(list(self.firstAlms_phi))
		plotControl(nFigure)
		
		#### PLOT IN ALM PLANE ####
		axScatter = plt.subplot(121) 
		Re = [alm_abs[k][l][m]*np.cos(alm_phi[k][l][m])*100000.0 for k in range(len(alm_abs))]
		Im = [alm_abs[k][l][m]*np.sin(alm_phi[k][l][m])*100000.0 for k in range(len(alm_abs))]
		
		axScatter.scatter(Re, Im, color=self.pltColor)
		plt.grid(True)
		plt.xlabel(r'$Re(a_{lm})\cdot 10^{5}$')
		plt.ylabel(r'$Im(a_{lm})\cdot 10^{5}$')
				
		
		divider = make_axes_locatable(axScatter)
		axHistx = divider.append_axes("top", 0.5, pad=0.1, sharex=axScatter)
		axHisty = divider.append_axes("right", 0.5, pad=0.1, sharey=axScatter)

		plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)
					
		histtype = "bar"
		axHistx.hist(Re, histtype= histtype, alpha=0.5)
		axHisty.hist(Im, orientation='horizontal', alpha=0.5, histtype= histtype)
					
		for tl in axHistx.get_xticklabels():
			tl.set_visible(False)
		axHistx.set_yticks([0, 1, 2])

		for tl in axHisty.get_yticklabels():
				tl.set_visible(False)
		axHisty.set_xticks([0, 1, 2])
				
		#### PLOT IN ASB-PHASE-PLANE
		axScatter = plt.subplot(122) 
		absolute = [alm_abs[k][l][m]*100000.0 for k in range(len(alm_abs))]
		phase	 = [alm_phi[k][l][m]*180.0/np.pi for k in range(len(alm_phi))]
		axScatter.scatter(absolute, phase, color=self.pltColor)
		plt.grid(True)
		plt.xlabel(r'$|a_{l}^{m}|\cdot 10^{5}$')
		plt.ylabel(r'$\phi_{l}^{m}$')
		

		divider = make_axes_locatable(axScatter)
		axHistx = divider.append_axes("top", 0.5, pad=0.1, sharex=axScatter)
		axHisty = divider.append_axes("right", 0.5, pad=0.1, sharey=axScatter)

		plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)

		bins_phi = np.arange(-180, 180 + 10, 10)
		histtype = "bar"
		axHistx.hist(absolute, histtype= histtype, alpha=0.5)
		axHisty.hist(phase, bins=bins_phi, orientation='horizontal',alpha=0.5,histtype= histtype)

		for tl in axHistx.get_xticklabels():
			tl.set_visible(False)
		axHistx.set_yticks([0, 1, 2])

		for tl in axHisty.get_yticklabels():
			tl.set_visible(False)
		axHisty.set_xticks([0, 1, 2])
		
		### global title ###
		plt.suptitle(" $ a_{l}^{m} $for l="+str(l)+", m="+str(m)+" abs.-phase-pl.", fontsize=16)
				
				
				
	#### PLOT alms UP TO l_show
	def plotFirstAlms(self, nFigure=-1, hideSubHistos=False, nLabel="", lim=[-300,300], markersize=20.0, markerstyle="."):
		alm_abs = copy.deepcopy(list(self.firstAlms_abs))
		alm_phi = copy.deepcopy(list(self.firstAlms_phi))
		plotControl(nFigure)
		#plt.suptitle(" $ a_{\ell}^{m} $ in Complex Plane", fontsize=30)
		ls = [1] #,100,300]
		ms = [[1]] #,[1,100],[1,150, 300]]
		this = 0
		for i in range(1,self.l_show+1):# (1, 80, 300):#  ### vorgabe bis zu welchem l
			print "I = "+str(i)
			for j in range(1,i+1): #ms[i]:
				print "J = "+str(j)
				index = (i-1)*self.l_show + j
				sub_ID = self.l_show*100+self.l_show*10+index
				#print str(self.l_show*100+self.l_show*10+index)
				axScatter = plt.subplot(sub_ID) #### anpassen an l und m
				#print "i="+str(i)+" j="+str(j)+" ls(i)="+str(ls[i])+" ms(i,j)="+str(ms[i][j]) 
				Re = [alm_abs[k][ls[i-1]][ms[i-1][j-1]]*np.cos(alm_phi[k][ls[i-1]][ms[i-1][j-1]])*100000.0 for k in range(len(alm_abs))]
				Im = [alm_abs[k][ls[i-1]][ms[i-1][j-1]]*np.sin(alm_phi[k][ls[i-1]][ms[i-1][j-1]])*100000.0 for k in range(len(alm_abs))]
				
				axScatter.scatter(Re, Im, markersize, marker=markerstyle ,color=self.pltColor, label=nLabel )
				plt.grid(True)
				#plt.colorbar().set_label(r'counts')
				#plt.title(r'Bgd')
				if i == self.l_show and j==(self.l_show+1)/2:
					plt.xlabel(r'$\Re( a_{\ell}^{m} )\cdot 10^{5}$', fontsize=38)
				elif not i==self.l_show:
					plt.subplot(sub_ID).axes.get_xaxis().set_visible(False)
				if j == 1 and i==(self.l_show+1)/2:
					plt.ylabel(r'$\Im( a_{\ell}^{m} )\cdot 10^{5}$', fontsize=38)
				elif not j==1:
					plt.subplot(sub_ID).axes.get_yaxis().set_visible(False)
					
				plt.subplot(sub_ID).set_xlim([lim[0],lim[1]])
				plt.subplot(sub_ID).set_ylim([lim[0],lim[1]])
				plt.subplot(sub_ID).axes.get_xaxis().set_ticks([lim[0]*2/3, 0,lim[1]*2/3])
				plt.subplot(sub_ID).axes.get_yaxis().set_ticks([lim[0]*2/3, 0,lim[1]*2/3])	
							
				#axScatter.set_aspect(1.)
				# create new axes on the right and on the top of the current axes
				# The first argument of the new_vertical(new_horizontal) method is
				# the height (width) of the axes to be created in inches.
				if hideSubHistos == False:
					divider = make_axes_locatable(axScatter)
					axHistx = divider.append_axes("top", 0.5, pad=0.1, sharex=axScatter)
					axHisty = divider.append_axes("right", 0.5, pad=0.1, sharey=axScatter)
					# make some labels invisible
					plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)
					# now determine nice limits by hand:
					#bins_phi = np.arange(-180, 180 + 10, 10)
					histtype = "bar"
					axHistx.hist(Re, histtype= histtype)
					axHisty.hist(Im, orientation='horizontal',histtype= histtype)
					# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
					# thus there is no need to manually adjust the xlim and ylim of these
					# axis.
					#axHistx.axis["bottom"].major_ticklabels.set_visible(False)
					for tl in axHistx.get_xticklabels():
						tl.set_visible(False)
					axHistx.set_yticks([0, 1, 2])
					#axHisty.axis["left"].major_ticklabels.set_visible(False)
					for tl in axHisty.get_yticklabels():
						tl.set_visible(False)
					axHisty.set_xticks([0, 1, 2])
				else:
					plt.title(r"$\ell="+str(ls[i-1])+",\, m="+str(ms[i-1][j-1])+"$", size=40)
			this = this+1


	#### PLOT alms UP TO l_show
	def plotFirstAlmPhis(self, nFigure=-1,  hideSubHistos=False, nLabel="", radial=False):
		alm_phi = copy.deepcopy(list(self.firstAlms_phi))
		
		plotControl(nFigure)
		if self.l_show == 1:
			plt.suptitle("$a_{\ell}^{m}$ Phase Distribution \n $\ell=1,\, m=1$")
			FRAC = 1.1
		else:
			plt.suptitle(" $ a_{\ell}^{m} $ Phase Distribution", fontsize=30)
			FRAC = 1.28
		for i in range(1,self.l_show+1): ### vorgabe bis zu welchem l
			for j in range(1,i+1):
				index = (i-1)*self.l_show + j
				#print "PHI: "+str(self.l_show*100+self.l_show*10+index)
				Phis = [alm_phi[k][i][j] for k in range(len(alm_phi))]
				self.Phis = Phis
				#print "PHIS:"
				#print Phis
				if radial==False:
					fig = plt.subplot(self.l_show*100+self.l_show*10+index) #### anpassen an l und m
					plt.hist(Phis, bins=20, histtype='stepfilled', color=self.pltColor, alpha=0.5, label=nLabel)
					plt.grid(True)
					if i == self.l_show:
						plt.xlabel(r'$ \phi (a_{l}^{m}) $')
						plt.ylabel(r' counts ')
				else:
					fig = plt.subplot(self.l_show*100+self.l_show*10+index, polar=True) #### anpassen an l und m
					hi, ed = np.histogram(Phis ,bins=20, range=(-np.pi,np.pi))
					#if index == 4 or index==1:
					#	print Phis
					#	print hi
					#	print ed
					#fig = figure(nFigure)
					#ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
					
					print len(ed)
					print len(hi)
					theta = setNewEdges(ed)
					theta2 = copy.copy(theta)
					theta.append(theta[0]+2*np.pi)
					radii = list(hi*1.0/sum(hi))
					radii2 = copy.copy(radii)
					radii.append(radii[0])
					width = ed[1]-ed[0]
					print theta
					print radii
					print len(theta)
					print len(radii)
					
					##### WARNING: THETA IS SHIFTED BY 1/2 BIN DUE TO steps-post PRINTING!!!
					plt.plot(theta, radii, color=self.pltColor, drawstyle="steps-post", linewidth=2.0)
					bars = fig.bar(theta2, radii2, width=width, bottom=0.0)
					for r,bar in zip(radii, bars):
						bar.set_facecolor(color=self.pltColor)
						bar.set_alpha(0.00)
						bar.set_linewidth(0.0)
					fig.set_thetagrids([0,45,135, 180, 225, 315], frac=FRAC)
					if self.l_show > 1:
						n = 1.0*1.0/(len(ed)-1) #len(Phis)
						print "n = "+str(n)
						fig.set_rgrids([(n*0.65),(n*1.3)])

				if hideSubHistos == False and radial==False:
					print "C"
					divider = make_axes_locatable(axScatter)
					axHistx = divider.append_axes("top", 0.3, pad=0.1, sharex=axScatter)
					axHisty = divider.append_axes("right", 0.3, pad=0.1, sharey=axScatter)
					plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)
					histtype = "bar"
					axHistx.hist(Re, histtype= histtype)
					axHisty.hist(Im, orientation='horizontal',histtype= histtype)
					for tl in axHistx.get_xticklabels():
						tl.set_visible(False)
					axHistx.set_yticks([0, 1, 2])
					for tl in axHisty.get_yticklabels():
						tl.set_visible(False)
					axHisty.set_xticks([0, 1, 2])
				elif not self.l_show == 1:
					plt.title(r"$\ell="+str(i)+",\, m="+str(j)+"$", size=25)
					
	
	#### CALCULATES LIST OF ANGULAR DIFFERENCE TO CLOSEST NEIGHBOUR
	def getCharacDistance(self, closestOnly=True):
		self.closestNeighbourDistance = getClosestDistance(self.ranTheta, self.ranPhi, closestOnly=closestOnly)
		return True
		
		
	#### PLOT CHARACTERISTIC DISTANCE
	def plotCharacDistance(self, nFigure=-1, label="charac. distance"):
		plotControl(nFigure)
		self.histoCharDistance, self.edgesCharDistance = np.histogram(self.closestNeighbourDistance, bins=300, range=(0.0,5.0)) #range=(min(self.closestNeighbourDistance), max(self.closestNeighbourDistance)))
		plt.plot(setNewEdges(self.edgesCharDistance), self.histoCharDistance, drawstyle='steps-mid', color=self.pltColor, label=label)
		plt.title(r" Characteristic Distance of Skymap ")
		plt.xlabel(r"$ \alpha $ [rad.] ")
		plt.ylabel("counts")
	
	
	##### GENERATES EVENTS FOR ATM. SPEC.
	def getAtmosNu(self, N_Atmos, Spline_Zenith, fullSphereMode=False, Milagro=False, fixZenith=False, RA_ACC=False):
		i=0
		Map_List = []
		if fullSphereMode == True:
			cut = 1.0
		else:
			cut = np.cos(np.pi/2-np.radians(5))

		if Milagro == True:
			while i < N_Atmos:
				cosThetaRan = np.random.uniform(-1.,cut)
				if np.random.uniform(0,1) < Spline_Zenith((-1.0)*abs(cosThetaRan)):
					t_ran = np.arccos(cosThetaRan)																	
					foundAzimuth = False
					while foundAzimuth == False:
						p_ran = np.random.uniform(0., 2*np.pi)
						if np.random.uniform(0.0,1.0) < self.Milagro(t_ran-np.pi/2.0, p_ran):
							foundAzimuth = True
							#print t_ran
						else:
							print "NO dec:"+str(t_ran+np.pi/2.0)
					Map_List.append([t_ran,p_ran])
					i=i+1
		elif fixZenith == True:
			while i < N_Atmos:
				t_ran = self.FixedZenithList[i]
				p_ran = np.random.uniform(0., 2*np.pi)
				Map_List.append([t_ran,p_ran])
				i=i+1
		elif RA_ACC == True:
			while i < N_Atmos:
				cosThetaRan = np.random.uniform(-1.,cut)
				if np.random.uniform(0,1) < Spline_Zenith((-1.0)*abs(cosThetaRan)):
					t_ran = np.arccos(cosThetaRan)
					p_ran = np.random.uniform(0., 2*np.pi)
					z = np.random.uniform(0.0,1.0)
					if z < self.RA_ACC_spline(p_ran):
						Map_List.append([t_ran,p_ran])
						i=i+1
					else:
						#print "BG event kicked by RA acceptance."
						self.RA_ACC_rejectBG += 1
		else:
			#~ if self.useE:
			while i < N_Atmos:
				cosThetaRan = np.random.uniform(-1.,cut)
				if np.random.uniform(0,1) < Spline_Zenith((-1.0)*abs(cosThetaRan)):## Accept theta if value suits atm zenith spline
					t_ran = np.arccos(cosThetaRan)
					p_ran = np.random.uniform(0., 2*np.pi)														## Uniform phi
					E_ran = self.getAtmEventEnergy()
					Map_List.append([t_ran,p_ran])
					self.energyList.append(E_ran)
					i=i+1
			#~ else:
				#~ while i < N_Atmos:
					#~ cosThetaRan = np.random.uniform(-1.,cut)
					#~ if np.random.uniform(0,1) < Spline_Zenith((-1.0)*abs(cosThetaRan)):
						#~ t_ran = np.arccos(cosThetaRan)
						#~ p_ran = np.random.uniform(0., 2*np.pi)
						#~ Map_List.append([t_ran,p_ran])
						#~ i=i+1
				
		return Map_List
		
		
	##### Generate Diffuse Background Flux LJS
	def getDiffBG(self, N_diff, Spline_Zenith_sig, fullSphereMode=False): 
		#N_diff: Poissonian Number of diffuse background events mu=205
		#Spline_Zenith_sig: Signal Zenith spline of one out of three samples, number chosen below
		Diff_List=[]
		i=0
		if fullSphereMode == True:
			cut = 1.0
		else:
			cut = np.cos(np.pi/2-np.radians(5))
			
		#Chose sample number: 1, 2 or 3
		ran_choice 		= np.random.uniform(0.0,1.0)		
		if self.samAeffContr[0] > ran_choice: sample = 0 													#"IC40"
		elif self.samAeffContr[1]+self.samAeffContr[0] > ran_choice: sample = 1 	#"IC59"
		elif sum(self.samAeffContr) > ran_choice: sample = 2 											#"IC79"
		else: print "ML-ERROR: Could not find matching detector for point source generation!"

		print "getDiffBG"
		while i < N_diff: 																										#count up to desired number of diffuse background events
			cosThetaRan = np.random.uniform(-1.,cut) 														#Random cosine theta
			if np.random.uniform(0,1) < Spline_Zenith_sig[sample]((-1.0)*abs(cosThetaRan)): #detector acceptance check
				t_ran = np.arccos(cosThetaRan) 																		#random theta
				p_ran = np.random.uniform(0., 2*np.pi) 														#random phi
				Diff_List.append([t_ran,p_ran])
				#~ if self.useE:
				E_ran = self.getSignalEventEnergy()
				self.energyList.append(E_ran)
				self.diffElist.append(E_ran)
				i=i+1
		return Diff_List
		
	##### SAVE ENERGY VALUES ######
	

	def saveEventEnergy(self, nTitle="", histogrammsave=True ):
		s_txt=str(nTitle)+"_Energy_"+str(self.NUMBER_RAN_RUNS)+".txt"
		if histogrammsave==True:
			histo_bg, histo_diff, edges = self.makeEventHisto(self.energyList, self.diffElist)
			np.savetxt(s_txt, np.array([histo_bg, histo_diff, setNewEdges(edges)]))
		else:	
			np.savetxt(s_txt, np.array(self.energyList))

	def makeEventHisto(self, bge, diffe, bins=80):
		if self.DETECTOR=="IC59":
			hmin = -5.
			hmax = 5.
			bins=100
		else:
			hmin = 1.
			hmax = 9.
		print self.DETECTOR
		hbg, ebg = np.histogram(bge, bins=bins, range=(hmin, hmax))
		hdiff, ediff = np.histogram(diffe, bins=bins, range=(hmin, hmax))
		if len(ebg) != len(ediff):
			print "Energy bin ERROR!"
			return 0., 0., 0.
		else:
			return hbg, hdiff, ebg
		
	
	##### RETURNS True OR False DEPENDING ON WHETHER theta, phi FIT MILAGRO ANISOTROPY OR NOT #####
	def Milagro(self, theta, phi):
		if self.MilagroSkymap[H.ang2pix(self.nside,theta,phi)] > 0.5:
			return self.MilagroSkymap[H.ang2pix(self.nside,theta,phi)]
		else:
			return 1.0
			
	def createBackground(self, mc, nAtm, nAstro, transpose=False, replace=True):
		"""
		choose events from mc sample using conventional=atm and astro weights with replacing
		returns arrays of astro[theta, phi] and atm[theta, phi]
		"""
		#nAtm=nTot-nAstro
		#~ ind_atm = np.random.choice(len(mc), size=nAtm, p=mc["conv"]/np.sum(mc["conv"]), replace=replace)
		#~ ind_astro = np.random.choice(len(mc), size=nAstro, p=mc["astro"]/np.sum(mc["astro"]), replace=replace)
		
		
		# CHANGE WEIGHTS ##############
		
		#w_astro = mc["ow"]*mc["trueE"]**(self.GAMMA)
		self.sample_atm = mc[np.random.choice(len(mc), size=nAtm, p=(self.weight_conv/np.sum(self.weight_conv)), replace=replace)]
		self.sample_astro = mc[np.random.choice(len(mc), size=nAstro, p=(self.weight_astro/np.sum(self.weight_astro)), replace=replace)] #mc["astro"]/np.sum(mc["astro"])
		#~ ind_atm = ind_atm.tolist()
		#~ oversamp = [i for i, x in enumerate(ind_atm) if ind_atm.count(x) > 1]
		#~ print "oversampling atm "+str(oversamp)
		#~ print len(oversamp)

		### shuffle ###
		if replace:
			#~ if False:
			print "Shuffling in azimuth"
			temp_atm = np.random.uniform(0., np.pi*2., size=len(self.sample_atm["ra"]))
			self.sample_atm["ra"] = self.sample_atm["ra"]+temp_atm
			self.sample_atm["ra"] = self.sample_atm["ra"]%(np.pi*2)
			self.sample_atm["trueRa"] = self.sample_atm["trueRa"]+temp_atm
			self.sample_atm["trueRa"] = self.sample_atm["trueRa"]%(np.pi*2)

			temp_astro = np.random.uniform(0., np.pi*2., size=len(self.sample_astro["ra"]))
			self.sample_astro["ra"] = self.sample_astro["ra"]+ temp_astro
			self.sample_astro["ra"] = self.sample_astro["ra"]%(np.pi*2)
			self.sample_astro["trueRa"] = self.sample_astro["trueRa"]+temp_astro
			self.sample_astro["trueRa"] = self.sample_astro["trueRa"]%(np.pi*2)
		
		### set energies ###
		self.energyList.extend(self.sample_astro["logE"])	#list total astro
		self.diffElist.extend(self.sample_astro["logE"])		#only astro diff list		
		self.energyList.extend(self.sample_atm["logE"])		#alist total atm
		
		diff_array = np.array([dec2zen(self.sample_astro["dec"]), self.sample_astro["ra"]])
		atm_array  = np.array([dec2zen(self.sample_atm["dec"]), self.sample_atm["ra"]])
		#~ diff_array = np.array([self.sample_astro["dec"], self.sample_astro["ra"]])    
		#~ atm_array  = np.array([self.sample_atm["dec"], self.sample_atm["ra"]])
		
		if transpose == True:
			diff_array = np.transpose(diff_array)
			atm_array = np.transpose(atm_array)
		
		return diff_array, atm_array
		
		
		
	#### CREATING nRuns RANDOM SKYMAPS AND ANALYSE THEM ###
	def createRanSkymaps(self, theta=[], phi=[]):
		self.warnIfFullsphere()
		self.cl_all = []
		self.alm_all = []
		self.cl_log_all = []
		self.effCl_all = []
		self.al0_all = []
		self.firstAlms_abs = []
		self.firstAlms_phi = []
		
		print "Creating pure atm. Skymaps, NEvents: "+str(self.NUMBER_SIMULATED_EVENTS)
		for curRun in range(0, self.NUMBER_RAN_RUNS):
			sys.stdout.write("\r RUN: "+str(curRun+1)+"            ")
			sys.stdout.flush()

			if self.MC_GEN_BOOL == False: ##Should be True in general...
				#### GENERATE RANDOM EVENTS FROM SIMULATED ATM. DATA #####	
				if len(theta) == 0 and len(phi) == 0:
					if self.useDiffBG:
						diffThetaPhi=self.getDiffBG(self.numSigEvents_Poi, self.signalSpline, self.fullSphereMode)
						print("Number of Diffuse Simulated Background Events is: " + str(len(diffThetaPhi)))
						self.generatedSigEvents.append([len(diffThetaPhi)])
						NumBGEvents=self.NUMBER_SIMULATED_EVENTS-self.numSigEvents_Poi
					else: 
						NumBGEvents=self.NUMBER_SIMULATED_EVENTS
						
					ranThetaPhi = self.getAtmosNu(NumBGEvents, self.atmosSpline, fullSphereMode=self.fullSphereMode, Milagro=self.MilagroAzimuth, fixZenith=self.FixedZenith, RA_ACC=self.RA_ACC)	
					self.ranTheta = [i[0] for i in ranThetaPhi]
					self.ranPhi = [i[1] for i in ranThetaPhi]
					print("Number of Atm. Simulated Background Events is: " + str(len(self.ranTheta)))
					
					if self.useDiffBG:
						for i in diffThetaPhi:
							self.ranTheta.append(i[0])
							self.ranPhi.append(i[1])
					print("Number of Total Simulated Background Events is: " + str(len(self.ranTheta)))
					
				else:
					self.ranTheta = theta
					#self.ranPhi = Phi #I think this was a typo
					self.ranPhi = phi
			
			else:
				#### GENERATE RANDOM EVENTS BY CHOOSING FROM MC DATA ####
				if len(theta) == 0 and len(phi) == 0:
					if self.useDiffBG:
						diffThetaPhi, ranThetaPhi = self.createBackground(self.mc_sample_full, self.NUMBER_SIMULATED_EVENTS-self.numSigEvents_Poi, self.numSigEvents_Poi)
					else:
						diffThetaPhi, ranThetaPhi = self.createBackground(self.mc_sample_full, self.NUMBER_SIMULATED_EVENTS, 0.)
						
					self.ranTheta = np.concatenate([diffThetaPhi[0], ranThetaPhi[0]])
					self.ranPhi = np.concatenate([diffThetaPhi[1], ranThetaPhi[1]])
					
					
					print "#Signal Events = "+str(self.numSigEvents_Poi)
					print "Len(theta_astro)="+str(len(diffThetaPhi[0]))
					self.generatedSigEvents.append(len(diffThetaPhi[0]))
			
			if len(self.ranTheta) != len(self.ranPhi):
				print "ML-WARNING: phi and theta lists do not have the same length("+str(len(self.ranTheta))+" "+str(len(self.ranPhi))+")!"
				
			#~ print "min phi, max phi, len(phi): "
			#~ print min(self.ranPhi), max(self.ranPhi), len(self.ranPhi)
			#~ print "min, max, len(theta): "
			#~ print min(self.ranTheta), max(self.ranTheta), len(self.ranTheta)
			#### CREATE SKYMAP AND ANALYSE Cl, alm SPECTRA ####
			if self.useE:
				self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi, self.energyList)
			else:
				self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi)
			self.analyseMap()
			
		sys.stdout.write("\n")	
		self.NUMBER_RAN_RUNS_CHANGED = self.NUMBER_RAN_RUNS
	
	##### SAVE ANGULAR DATA #####
	
	def makeAngleHisto(self, theta, phi, bins=100):
		hbg, ebg = np.histogram(theta, bins=bins, range=(0, np.pi))
		hdiff, ediff = np.histogram(phi, bins=bins, range=(0, 2*np.pi))
		return hbg, hdiff, ebg, ediff
			
	def saveAngular(self, nTitle="",prefix="", histogrammsave=True):
		ang_txt=str(nTitle)+"angle_"+prefix+".txt"
		if histogrammsave==True:
			histo_theta, histo_phi, edges_theta, edges_phi = self.makeAngleHisto(self.ranTheta, self.ranPhi)
			np.savetxt(ang_txt, np.array([histo_theta, histo_phi, setNewEdges(edges_theta), setNewEdges(edges_phi)]))
			print "Angular Distribution saved to: " + ang_txt
		else:
			np.savetxt(ang_txt, np.array([self.ranTheta, self.ranPhi]))			
		
		
		
	def saveSouAngular(self, nTitle="",prefix=""):
		theta_sou_txt=str(nTitle)+"theta_sou_"+prefix+".txt"
		theta_sou_save=[]
		print "Type..."
		print type(self.theta_sou[0])
		if type(self.theta_sou[0])!=np.float64:
			for thetas in self.theta_sou:
				theta_sou_save=np.concatenate([thetas,theta_sou_save])
		else:
			theta_sou_save=self.theta_sou	
			
		np.savetxt(theta_sou_txt, theta_sou_save)
		print "Theta Source Distribution saved to: " + theta_sou_txt
	
	#### SAVE MU INFO ###
	def saveMuInfo(self, nTitle="",prefix=""):
		mu_txt = str(nTitle)+"muReal_all_"+prefix+".txt"
		np.savetxt(mu_txt, self.MuReal_all)
		#~ print "MU info saved to " + mu_txt
		
		#~ wh_txt = str(nTitle)+"_wrongHemi_all_"+str(self.NUMBER_RAN_RUNS)+".txt"
		#~ np.savetxt(wh_txt, self.wrongHemi_all)
		#~ print "Wrong Hemi info saved to " + wh_txt

	#### RE-EXPAND MAP FOR CERTAIN l ####
	def reexpand(self, nFigure, lmin, lmax):
		if len(self.alm_all) == 0: print "Reexpansion impossible: No alm saved!"
		else:
			self.reAlm = copy.deepcopy(self.alm_all[0]) 
			plotControl(nFigure)
			if not lmax == 0:
				for l in range(0, self.almLimit):
					for m in range(0, l+1):
						if l > lmax or l < lmin:
							self.reAlm[H.sphtfunc.Alm.getidx(self.almLimit, l,m)] = 0.0
				
			self.reExpMap = H.alm2map(self.reAlm, self.nside)
			H.mollview(self.reExpMap,fig=nFigure, title="Re-expansion", xsize=800)
			
	#### SAVE ALL DATA OR NOT ####
	def setAllData(self, all_d):
		self.SAVE_ALL_DATA = all_d
	
	#### ANALYSE MAP
	def analyseMap(self):
		## generate alm, Cl, effCl
			
		cl_delta, alm_delta = H.anafast(self.map_delta, lmax=self.l_max, alm=True)
		cl_auto_delta, alm_auto_delta = H.anafast(self.map_delta, lmax=self.l_max, alm=True)	# map auto-correlation
		#cl_auto_cat, alm_auto_cat = H.anafast(self.catalog_map, lmax=self.l_max, alm=True)	# cat auto-correlation
		
		if self.SAVE_ALL_DATA:
			al0curr = absList(getAl0List(alm_delta, self.l_max))
			al0curr_normd = absList(getAl0List(alm_delta_normd, self.l_max))
			
		effClcur = retEffCl(alm_delta, lmax=self.l_max)
		effClcur = retEffCl(alm_delta, lmax=self.l_max)	
		if self.useClLog == True:
			cl_log_delta = calcClLogFromAlm(alm_delta, 400)
			self.cl_log_all.append(cl_log_delta)
		if self.useAlm == True:
			alm_delta = H.anafast(self.map_delta, lmax=self.almLimit, alm=True)[1]

		## save alm, Cl, effCl
		if self.SAVE_ALL_DATA:
			self.saveFirstAlms(alm_delta)
			self.cl_all.append(cl_delta)
			self.al0_all.append(al0curr)
		
		self.effCl_all.append(effClcur)
		
		#if self.SAVE_Cls:
		#	self.cl_all.append(cl_delta)
		#	self.cl_auto_all.append(cl_auto_delta)
		#	if len(self.cl_auto_cat)==0:
		#		self.cl_auto_cat=cl_auto_cat
		
		if self.useAlm == False:
			print "***"
			print "WARNING!! alm NOT saved!!!"
			print "***"
			
		else:
			self.alm_all.append(alm_delta)
			self.alm_all_abs = []
			self.alm_all_im = []
			for i in range(0,len(self.alm_all)):
				self.alm_all_abs.append([])
				self.alm_all_im.append([])
				self.alm_all_phase.append([])
				for k in range(0, len(self.alm_all[i])):
					self.alm_all_abs[i].append(abs(self.alm_all[i][k]))#[k]
					self.alm_all_im[i].append(np.imag(self.alm_all[i][k]))
					self.alm_all_phase[i].append(np.angle(self.alm_all[i][k]))
			
			
		
	#### HITBOOL TO MAP SMEARED OR UNSMEARED		
	def hitBool2Map(self, theta, phi, E=[]):

		#### HIT CENTERING ...
		if self.centerHits == True:
			theta, phi = centerHits(self.nside,theta, phi)
		
		bad_indices = [] # workaround to get this work with conditional loops

		galcord=galactic(phi,zen2dec_noticecube(dec2zen(theta)))[1] # to avoid icecube coordinates bullshite
		
		if not self.useGalPlane:				#### EXCLUDING THE GALACTIC PLANE ...
			for i in range(len(galcord)):
				if (abs(galcord[i]) < np.radians(5)):
					bad_indices.append(i)
					
		elif self.onlyGalPlane:				#### OR EVERYTHING ELSE ...
			for i in range(len(galcord)):
				if (abs(galcord[i]) > np.radians(5)):
					bad_indices.append(i)

		theta=np.delete(theta, bad_indices)
		phi=np.delete(phi, bad_indices)
		
		starttime = tm.time()
		if self.smearingMethod != "none":
			print "Start smearing... "
		
		#### SMEARING ...
		self.map_delta = np.zeros(self.npix)
		#self.map_unweighted = np.zeros(self.npix)
		if self.smearingMethod == "none":
			hit_bool = H.ang2pix(self.nside,theta,phi)
			if (self.useE):
				if len(E) == 0 or len(E) < len(theta): 
					print "ML-WARNING: ENERGY INFORMATION LOST BEFORE INSERTING INTO MAP!"
					print len(E), len(theta)
				fac_llh = getLikelihoodE2(self.sigEnergySpline, self.atmEnergySpline, E, self.NUMBER_SIMULATED_EVENTS, self.NumSigEvents) 
				#### FORMER getLikelihoodE(self.sigEnergySpline, self.atmEnergySpline, E)
				for i in range(len(hit_bool)):
					self.map_delta[hit_bool[i]] += fac_llh[i]
					#~ self.map_unweighted[hit_bool[i]] += 1.
			else: 
				for i in range(len(hit_bool)):
					self.map_delta[hit_bool[i]] += 1.0
		elif self.smearingMethod == "classic":
			self.slowSmearHitBool(theta,phi)
			print "NOTE: class. smearing chosen => calc. have low performance."
		elif self.smearingMethod == "classic_GPU":
			self.slowSmearHitBool_cuda(theta, phi)
			print "NOTE: class. smearing chosen + GPU calculations!"
		else:
			self.quickSmearHitBool(theta,phi) ### QUICK CALC OR NOT ???
			print "NOTE: quick smearing chosen."
		
		if self.smearingMethod != "none":
			print "... done. Duration of smearing: "+str(round(tm.time()-starttime,2))+"s \n"
			
		### RENORMALIZATION ...
		if self.renormSkymaps == True:
			self.map_delta = renormSkymap(self.map_delta,len(theta))
			print "Renormalization: " +str(np.sum(self.map_delta))
		else:
			print "NEW NORM: "+str(sum(self.map_delta))
			
		### ROUND ENTRIES TO SAVE SPACE ON HD
		#self.map_delta = np.array([round(elm, 12) for elm in self.map_delta])

	
	#### SMEARS GIVEN MAP BY PS SPLINE
	def quickSmearHitBool(self, theta, phi):
		self.map_delta = np.zeros(self.npix)
		hit_index = H.ang2pix(self.nside,theta,phi)

		cosThHit = np.cos(theta)
		sinThHit = np.sin(theta)
		cosPhiHit = np.cos(phi)
		sinPhiHit = np.sin(phi)
		
		for index in range(len(theta)):
			self.allSmearedPix = [] # [hit_index[index]]
			self.quickSmearOneHit(hit_index[index],cosThHit[index], sinThHit[index], cosPhiHit[index], sinPhiHit[index])
			if index % 1000 == 0:
				sys.stdout.write("\r finished "+str(index)+" events      ")
				sys.stdout.flush()
		sys.stdout.write("\n")
		
	
	#### SMEAR ONE HIT ####
	def quickSmearOneHit(self, startIndex, cosThHit, sinThHit, cosPhiHit, sinPhiHit):
		#print startIndex
		neighbours = allPixNeighbours[startIndex]
		for neigh in neighbours:
			self.cosAngle = getQuickAngleBetween(cosThHit, sinThHit, cosPhiHit, sinPhiHit, globPixCosTheta[neigh],globPixSinTheta[neigh], globPixCosPhi[neigh], globPixSinPhi[neigh])
			#print "cos: "+str(self.cosAngle)
			#print "angle: "+str(np.arccos(self.cosAngle)*180.0/np.pi) 
			if self.cosAngle > self.cosResSmearing  and not (neigh in self.allSmearedPix):
					self.map_delta[neigh] += self.smearingSpline(np.arccos(self.cosAngle))
					self.allSmearedPix.append(neigh)
					self.quickSmearOneHit(neigh, cosThHit, sinThHit, cosPhiHit, sinPhiHit)


	#### SMEARS GIVEN MAP BY PS SPLINE
	def slowSmearHitBool(self, theta, phi):
			cosThHit = np.cos(theta)
			sinThHit = np.sin(theta)
			cosPhiHit = np.cos(phi)
			sinPhiHit = np.sin(phi)
			
			for hit in range(len(theta)):
				self.cosAngle = getQuickAngleBetween(cosThHit[hit], sinThHit[hit], cosPhiHit[hit], sinPhiHit[hit], globPixCosTheta,globPixSinTheta, globPixCosPhi, globPixSinPhi)
				self.cosAngle = [round(elem,13) for elem in self.cosAngle]
				self.map_delta += self.smearingSpline(np.arccos(self.cosAngle))		


	#### SMEARS GIVEN MAP BY PS SPLINE
	def slowSmearHitBool_cuda(self, theta, phi):

		npix = np.int32(self.npix)
		nhits = np.int32(len(theta))
		prec = np.int32(100000)
		pix_t, pix_p = H.pix2ang(self.nside, np.arange(npix))
		self.spline = self.smearingSpline(np.arange(0.0,np.pi+1.0/prec,1./prec)) #/ self.smearingSplineInt
		temp_map = np.empty_like(pix_t)

		self.spline = self.spline.astype(np.float64)
		theta = np.array(theta).astype(np.float32)
		phi = np.array(phi).astype(np.float32)
		pix_t = pix_t.astype(np.float32)
		pix_p = pix_p.astype(np.float32)
		temp_map = temp_map.astype(np.float64)

		hit_t_gpu = cuda.mem_alloc(theta.nbytes) ### speicherreservierung aug gpu
		hit_p_gpu = cuda.mem_alloc(phi.nbytes)
		pix_t_gpu = cuda.mem_alloc(pix_t.nbytes)
		pix_p_gpu = cuda.mem_alloc(pix_p.nbytes)
		spline_gpu = cuda.mem_alloc(self.spline.nbytes)
		map_gpu = cuda.mem_alloc(temp_map.nbytes)

		cuda.memcpy_htod(hit_t_gpu, theta) ### speicherkopierung von dec zur gpu
		cuda.memcpy_htod(hit_p_gpu, phi)
		cuda.memcpy_htod(pix_t_gpu, pix_t)
		cuda.memcpy_htod(pix_p_gpu, pix_p)
		cuda.memcpy_htod(spline_gpu, self.spline)
		cuda.memcpy_htod(map_gpu, temp_map)
		
		#print temp_map
		mod = SourceModule("""
		    __global__ void CalcMap(const float *hit_t, const float *hit_p, const float *pix_t, const float *pix_p,
                                        const double *spline, double *map_gpu,
                                        const int npix, const int nhits, const int prec){
                    int idx = threadIdx.x + blockIdx.x*blockDim.x;
                    if (idx < npix){
                        double val = 0.0;
                        double pi = acos(-1.0);
                        for( int j=0; j<nhits; j++){
                            double arg = sin(hit_t[j]) * sin(hit_p[j]) * sin(pix_t[idx]) * sin(pix_p[idx])
                                        +sin(hit_t[j]) * cos(hit_p[j]) * sin(pix_t[idx]) * cos(pix_p[idx])
                                        +cos(hit_t[j]) * cos(pix_t[idx]);
                            if (arg > 1.0){ arg = 1.0; }
                            if (arg < -1.0){ arg = -1.0; }
                            double psi = acos(arg);
                            if (psi < 0.001588) { psi = 0.001588; }
                            int i = floor(psi*prec);
                            double X = i/(1.0*prec);
                            double ps = (psi - X) * (spline[i+1] - spline[i])*prec + spline[i];
                            val += ps;
                            //val = psi;
                        }//for
                        map_gpu[idx] = val;
                    }// if
                    __syncthreads();
                }// CalcMap
            """)

		bx = np.int32(512)
		gx = np.int32(npix/bx)
		func = mod.get_function("CalcMap") ### code compilierung
		func(hit_t_gpu, hit_p_gpu, pix_t_gpu, pix_p_gpu, spline_gpu, map_gpu, npix, nhits, prec, block=(int(bx),1,1), grid=(int(gx),1,1)) ### aufruf der funktion
		#print temp_map
		cuda.memcpy_dtoh(temp_map, map_gpu) ### kopiere speicher von gpu zur cpu
		#print temp_map
		self.map_delta = temp_map


	#### CALCULATE MEANS AND ERRORS OF ANALYSED SKYMAPS
	def calcMeans(self, alm=False, log=False):
		if self.SAVE_ALL_DATA:
			self.cl_means = np.mean(self.cl_all, axis=0)
			self.cl_errors = np.std(self.cl_all, axis=0)

		self.effCl_means = np.mean(self.effCl_all, axis=0)
		#~ print "bla"
		self.effCl_errors = np.std(self.effCl_all, axis=0)
		
		if self.useClLog == True:		
			self.cl_log_means = np.mean(self.cl_log_all, axis=0)
			self.cl_log_errors = np.std(self.cl_log_all, axis=0)
		if self.useSqrtEffCl == True:
			self.sqrtEffCl_all = [np.sqrt(this_effCl) for this_effCl in self.effCl_all]
			self.sqrtEffCl_means = np.mean(self.sqrtEffCl_all, axis=0)
			self.sqrtEffCl_errors = np.std(self.sqrtEffCl_all, axis=0)
			
			
		if log == True:
			logCl = np.log(self.cl_all)
			logEffCl = np.log(self.effCl_all)
			
			self.logCl_means = np.mean(logCl, axis=0)
			self.logCl_errors = np.std(logCl, axis=0)

			self.logEffCl_means = np.mean(logEffCl, axis=0)
			self.logEffCl_errors = np.std(logEffCl, axis=0)
			
		if alm == True and self.SAVE_ALL_DATA==True:
			self.alm_means = np.mean(self.alm_all_abs, axis=0)
			self.alm_errors = np.std(self.alm_all_abs, axis=0)
			self.alm_im_means = np.mean(self.alm_all_im, axis=0)
			self.alm_im_errors = np.std(self.alm_all_im, axis=0)
				
		if len(self.effCl_means) > self.l_max+1:
			if self.SAVE_ALL_DATA:
				self.cl_means = self.cl_means[0:self.l_max+1]
				self.cl_errors = self.cl_errors[0:self.l_max+1]
			
			self.effCl_means = self.effCl_means[0:self.l_max]
			self.effCl_errors = self.effCl_errors[0:self.l_max]
			
			if log == True:
				self.logCl_means = self.logCl_means[0:self.l_max+1]
				self.logEffCl_means = self.logEffCl_means[0:self.l_max]
				self.logCl_errors = self.logCl_errors[0:self.l_max+1]
				self.logEffCl_errors = self.logEffCl_errors[0:self.l_max]


	#### PLOT LAST SKYMAP #### LJS: add Log parameter, only for visualization!
	def plotLastSkymap(self, nFigure=-1, title="Skymap", res=800, grid=True, logz=False, raiser=0.001):
		plotControl(nFigure)
		Norm=None
		if logz: Norm='log'
		for i in range(len(self.map_delta)):
			self.map_delta[i] = float(self.map_delta[i])
			if logz:
				self.map_delta[i]=self.map_delta[i]+raiser #does not affect effCl, only overall rise
		
		colormap=cm.get_cmap("gnuplot")
		colormap.set_under("0.1")
		col= "w"
		H.mollview(self.map_delta, cmap=colormap, title="", fig=nFigure, xsize=res, norm=Norm) 
		H.graticule(dpar=30,dmer=60)
		
		if grid:
			H.projtext(-2.0, -8.0, "$0^{\circ}$",lonlat=True, color=col)
			#H.projtext(28.0, -8.0, "$30^{\circ}$",lonlat=True, color=col)
			H.projtext(58.0, -8.0, "$60^{\circ}$",lonlat=True, color=col)
			#H.projtext(88.0, -8.0, "$90^{\circ}$",lonlat=True, color=col)
			H.projtext(118.0, -8.0, "$120^{\circ}$",lonlat=True, color=col)
			#H.projtext(148.0, -8.0, "$150^{\circ}$",lonlat=True, color=col)
			H.projtext(178.0, -8.0,"$180^{\circ}$",lonlat=True, color=col)

			#H.projtext(-32.0, -8.0, "$330^{\circ}$",lonlat=True, color=col)
			H.projtext(-62.0, -8.0, "$300^{\circ}$",lonlat=True, color=col)
			#H.projtext(-92.0, -8.0, "$270^{\circ}$",lonlat=True, color=col)
			H.projtext(-122.0, -8.0, "$240^{\circ}$",lonlat=True, color=col)
			#H.projtext(-152.0, -8.0, "$210^{\circ}$",lonlat=True, color=col)

			H.projtext(-2.0, 22.0, "$30^{\circ}$",lonlat=True, color=col)
			H.projtext(-2.0, 52.0, "$60^{\circ}$",lonlat=True, color=col)
			H.projtext(-2.0, 76.0, "$90^{\circ}$",lonlat=True, color=col)

			H.projtext(-2.0, -28.0, "$-30^{\circ}$",lonlat=True, color=col)
			H.projtext(-2.0, -58.0, "$-60^{\circ}$",lonlat=True, color=col)
			H.projtext(-2.0, -82.0, "$-90^{\circ}$",lonlat=True, color=col)
			

	#### PLOTS ORIGINAL ZENITH SPEC. OF ATM. SKYMAPS
	def plotZenithSpec(self,nFigure=-1, nLabel="Atm. ZenithSpectrum"):
		plotControl(nFigure)

		plt.plot(self.edgesZenithDistrAtm, self.histoZenithDistrAtm, drawstyle='steps-mid', color=self.pltColor,  linestyle="-" ,label=nLabel)
		plt.title("Atm. event distr.")
		plt.xlabel(r"$ cos(\theta) $")
		plt.ylabel(r"rel. probability")

	#### PLOT ENERGY DISTRIBUTION ####	
	def plotEnergyDistr(self, nFigure=-1, nLabel=['Signal', 'Diff. Background', 'Atm. Background'], normed=False):
		plotControl(nFigure)
		nColor=self.nColor
		
		cut=len(self.sigElist)+len(self.diffElist)
		purebg=self.energyList[cut:]
		hmin=min(self.energyList)
		hmax=max(self.energyList)

		plt.hist([self.sigElist, self.diffElist, purebg], bins= 40, range=(hmin,hmax), stacked=True, color=nColor, label=nLabel, normed=normed)
		plt.semilogy(nonposy="clip")
		plt.legend()
		plt.title("Event Energy Spectrum")
		plt.xlabel(r"E[GeV]")
		plt.ylabel("#")

	#### PLOT NORM(ALM) PLANE OF MEANS
	def pltNormAlmPlane(self, nFigure=-1):
		self.alm_plot = []
		for i in range(0, self.l_max+1):
			self.alm_plot.append([])
			for k in range(0, self.l_max+1):
				self.alm_plot[i].append(self.alm_means[H.sphtfunc.Alm.getidx(self.l_max, i,k)]*1.0) 
				if k>i:
					self.alm_plot[i][k]=0.
		if nFigure == -1:
			plt.figure()
		else:
			plt.figure(nFigure, (16,12))
		plt.imshow(self.alm_plot, interpolation='nearest')
		plt.grid(True)
		plt.xlabel("m")
		plt.ylabel("$ \ell $")
		plt.title("Norm(Alm) Distr.")
		cb = plt.colorbar()
		cb.set_label(r'$ \langle \| a_{l, atm}^{m}\|\rangle $')	


	#### PLOT Cl DISTRIBUTION
	def pltClDistr(self,nFigure=-1, nLabel=r" $ C_{l} $, No Signal", marker="o", markersize=10.0, markeredgewidth=0.0):
		plotControl(nFigure, title="cl_spectra")
		
		plt.errorbar(self.ls, self.cl_means, yerr=np.sqrt(1.0/self.l_max)*self.cl_errors, fmt=marker, color=str(self.pltColor), label=nLabel, markeredgewidth=markeredgewidth, markersize=markersize)
		plt.xlabel("$ \ell $")
		plt.ylabel("abs. val.")
		plt.title(r" $C_{l} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS))#+", Runs: "+str(self.NUMBER_RAN_RUNS)
		plt.yscale("log")
		
	
	#### PLOT EFFCL DISTRIBUTION Difference
	def pltClDistrD(self, bg_cl, bg_error, nFigure=-1, nLabel=r" $ C_{l}^{\mathrm{eff}} $, No Signal", marker="o", markersize=10.0, markeredgewidth=0.0, start=0):
		plotControl(nFigure, title="Cl_difference")
		axScatter = plt.subplot(111)
		
		self.d_cl=[]
		
		for l in range(start,len(self.cl_means)):
			self.d_cl.append((self.cl_means[l]-bg_cl[l])/(np.sqrt(1.0/self.l_max)*np.sqrt(self.cl_errors[l]**2+bg_error[l]**2)))
		
		plt.errorbar(self.ls[start:], self.d_cl,  fmt=marker, color=str(self.pltColor), label=nLabel, markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor="k", elinewidth=1., yerr=np.sqrt(1.0/self.l_max)*np.sqrt(self.cl_errors[l]**2+bg_error[l]**2)) 
		plt.xlabel("$ \ell $")
		plt.ylabel(r" $ C_{\ell} \mathrm{(Sig-BG)}$")
		#plt.title(r" $ C_{l}^{\mathrm{eff}} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)) #+", Runs: "+str(self.NUMBER_RAN_RUNS)
		#~ plt.yscale("log")
		plt.tick_params(labelsize=25.)
		plt.grid(True)
		
		plotControl(nFigure+10, title="Cl Pull")
		plt.hist(self.d_cl, bins=30, range=(-4., 4.), color=str(self.pltColor), label=nLabel, histtype="step")
		plt.xlabel("Pull")
		plt.ylabel("#")


	#### PLOT effCl DISTRIBUTION
	def pltEffClDistr(self,nFigure=-1, nLabel=r" $ C_{\ell}^{\mathrm{eff}} $, No Signal", marker="o", markersize=10.0, markeredgewidth=0.0, errorband=False, n=580):
		plotControl(nFigure)
		plt.subplots_adjust(left=0.2)
		axScatter = plt.subplot(111)
		if not errorband:
			plt.errorbar(self.lsEff, self.effCl_means*1e3,  fmt=marker, color=str(self.pltColor), label=nLabel, markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor="k", 
										yerr=np.sqrt(1.0/self.NUMBER_RAN_RUNS)*self.effCl_errors, elinewidth=1.) #np.sqrt(1.0/self.NUMBER_RAN_RUNS)*
		else:
			plt.plot(self.lsEff, self.effCl_means*1e3,  marker=marker, color=str(self.pltColor), label=nLabel, markersize=markersize, 
								markeredgewidth=markeredgewidth, markeredgecolor="k")
			plt.fill_between(self.lsEff, (self.effCl_means+self.effCl_errors)*1e3, (self.effCl_means-self.effCl_errors)*1e3, color=str(self.pltColor), alpha=0.2)
		plt.xlabel("$ \ell $")
		plt.ylabel(r" $ C_{\ell}^{\mathrm{eff}} \cdot 10^3 $")
		plt.grid(True)
		#plt.title(r" $ C_{l}^{\mathrm{eff}} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)) #+", Runs: "+str(self.NUMBER_RAN_RUNS)
		plt.yscale("log")
		plt.grid(True)
		
	#### PLOT CUMMULATIVE EFFCL DISTRIBUTION 
	def pltClCummulative(self, opt="effCl", nFigure=-1, nLabel=r" $ C_{\ell}^{\mathrm{eff}} $, No Signal", marker="o", markersize=10.0, markeredgewidth=0.0, limit=600):
		plotControl(nFigure)
		axScatter = plt.subplot(111)
		
		self.cummulative_Cl=[]	
		if opt == "effCl":	
			self.cummulative_Cl.append(self.effCl_means[0]*2.)
			for l in range(1,len(self.effCl_means)):
				self.cummulative_Cl.append(self.cummulative_Cl[-1]+2.*(l+1)*self.effCl_means[l])
			xval = self.lsEff
			
		else:			
			self.cummulative_Cl.append(self.cl_means[0])
			for l in range(1, len(self.cl_means)):
				self.cummulative_Cl.append(self.cummulative_Cl[-1]+(2.*l+1)*self.cl_means[l])
			xval = self.ls
		
		plt.errorbar(xval, self.cummulative_Cl,  fmt=marker, color=str(self.pltColor), label=nLabel+", max="+str(round(self.cummulative_Cl[limit], 3)), markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor="k") #, yerr=np.sqrt(1.0/self.l_max)*self.effCl_errors) 
		plt.xlabel("$ \ell $")
		if opt == "effCl":
			plt.ylabel(r" Cummulative $ \sum _{\ell=1}^{\ell_{max}} C_{\ell}^{\mathrm{eff}} \cdot 2(\ell+1)$") 
		else:
			plt.ylabel(r" Cummulative $ \sum _{\ell=0}^{\ell_{max}} C_{\ell}^{\mathrm{eff}} \cdot (2\ell+1)$")
		#plt.title(r" $ C_{l}^{\mathrm{eff}} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)) #+", Runs: "+str(self.NUMBER_RAN_RUNS)
		plt.yscale("linear")
		plt.grid(True)

	#### PLOT EFFCL DISTRIBUTION QUOTIENT
	def pltEffClDistrQ(self, bg_effcl, nFigure=-1, nLabel=r" $ C_{\ell}^{\mathrm{eff}} $, No Signal", marker="o", markersize=10.0, markeredgewidth=0.0):
		plotControl(nFigure, title="effCl_quotient")
		axScatter = plt.subplot(111)
		
		self.q_effCl=[]
		
		for l in range(0,len(self.effCl_means)):
			self.q_effCl.append(self.effCl_means[l]/bg_effcl[l])
		
		plt.errorbar(self.lsEff, self.q_effCl,  fmt=marker, color=str(self.pltColor), label=nLabel, markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor="k", yerr=np.sqrt(1.0/self.l_max)*self.effCl_errors) #yerr=np.sqrt(1.0/self.l_max)*self.effCl_errors,
		plt.xlabel("$ \ell $")
		plt.ylabel(r" $ C_{l}^{\mathrm{eff}} \frac{\mathrm{Sig}}{\mathrm{BG}}$")
		#plt.title(r" $ C_{l}^{\mathrm{eff}} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)) #+", Runs: "+str(self.NUMBER_RAN_RUNS)
		#~ plt.yscale("log")
		plt.grid(True)

	#### PLOT EFFCL DISTRIBUTION Difference
	def pltEffClDistrD(self, bg_effcl, bg_error, nFigure=-1, nLabel=r" $ C_{l}^{\mathrm{eff}} $, No Signal", marker="o", markersize=10.0, markeredgewidth=0.0):
		plotControl(nFigure, title=r"$C_{l}^{\mathrm{eff}}$ difference")
		axScatter = plt.subplot(111)
		
		self.d_effCl=[]
		
		for l in range(0,len(self.effCl_means)):
			self.d_effCl.append((self.effCl_means[l]-bg_effcl[l])/(np.sqrt(self.effCl_errors[l]**2+bg_error[l]**2))) #np.sqrt(1.0/self.l_max)
		
		plt.errorbar(self.lsEff, self.d_effCl,  fmt=marker, color=str(self.pltColor), label=nLabel, markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor="k", yerr=np.sqrt(1.0/self.l_max)*np.sqrt(self.effCl_errors**2+bg_error**2), elinewidth=1.) #) #yerr=np.sqrt(1.0/self.l_max)*self.effCl_errors,
		plt.xlabel("$ \ell $")
		plt.ylabel(r" $ C_{l}^{\mathrm{eff}} \mathrm{(Sig-BG)}/\sigma$")
		#plt.title(r" $ C_{l}^{\mathrm{eff}} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)) #+", Runs: "+str(self.NUMBER_RAN_RUNS)
		#~ plt.yscale("log")
		plt.tick_params(labelsize=25.)
		plt.grid(True)
		
		plotControl(nFigure+10, title=r"$C_{l}^{\mathrm{eff}}$ Pull")
		n, bins, _ = plt.hist(self.d_effCl, bins=50, color=str(self.pltColor), label=nLabel, histtype="step") #, range=(-4., 4.)
		result = fitGaussian(self.d_effCl,bins,n,plot=True, nLabel="")
		plt.vlines(result[1], 0, max(n)+10, label=r"$\mu=$"+str(round(result[1], 2))+" $\pm$ "+str(round(result[1+3], 2)), color="blue")
		#plt.hlines(0,0,0, label=r"$\sigma=$"+str(round(result[2], 2))+" $\pm$ "+str(round(result[2+3], 2)))
		print "gaussian fit result:"
		print result
		plt.ylim([0, max(n)+40])
		plt.xlabel("Pull")
		plt.ylabel("#")

	#### PLOT effCl DISTRIBUTION
	def pltAlternativeClDistr(self, nFigure=-1, opt="Cl", nLabel=""):
		plotControl(nFigure)

		if opt == "Cl":
			self.alterCl 		= [l*(l+1)*1.0/(2.0*np.pi)*self.cl_means[l] for l in range(0,len(self.cl_means))]
			self.alterCl_err	= [l*(l+1)*1.0/(2.0*np.pi)*np.sqrt(1.0/self.l_max)*self.cl_errors[l] for l in range(0,len(self.cl_means))]
			plt.errorbar(self.ls, self.alterCl, yerr=self.alterCl_err, fmt='o', color=str(self.pltColor), label=nLabel)
			plt.xlabel("$ \ell $")
			plt.ylabel("$\ell \cdot (\ell +1) / (2 \cdot \pi ) \cdot C_{\ell} $")
			plt.title(r" $ C_{\ell} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)) #+", Runs: "+str(self.NUMBER_RAN_RUNS)
			plt.yscale("log")
			
		if opt == "effCl":
			self.alterEffCl 		= [l*(l+1)*1.0/(2.0*np.pi)*self.effCl_means[l-1] for l in range(1,len(self.effCl_means)+1)]
			self.alterEffCl_err	= [l*(l+1)*1.0/(2.0*np.pi)*np.sqrt(1.0/self.l_max)*self.effCl_errors[l-1] for l in range(1,len(self.effCl_means)+1)]
			plt.errorbar(self.lsEff, self.alterEffCl, yerr=self.alterEffCl_err, fmt='o', color=str(self.pltColor), label=nLabel)
			plt.xlabel("$ \ell $")
			plt.ylabel("$\ell \cdot (\ell +1) / (2 \cdot \pi ) \cdot C_{\ell}^{\mathrm{eff}} $")
			plt.title(r" $ C_{\ell}^{\mathrm{eff}} $-Distr., lmax="+str(self.l_max)+", ev.: "+str(self.NUMBER_SIMULATED_EVENTS)+", Runs: "+str(self.NUMBER_RAN_RUNS))
			plt.yscale("log")
		
		
	
	#### SAVE Cl, effCl, Alm TO FILE
	def saveClAlm(self,nPath="", alm=True):
		if nPath == "":
			nPath = self.SAVE_DIR
		if len(self.cl_means)== self.l_max+1: 
			toSave = []
			toSave.append(self.cl_means)
			toSave.append(self.cl_errors)
			np.savetxt(nPath+"atmClRUNS"+str(self.NUMBER_RAN_RUNS)+"_StandardAtm.txt", toSave) ### Cl
			print "Cl data successfully saved..."+nPath+"atmClRUNS"+str(self.NUMBER_RAN_RUNS)+"_StandardAtm.txt"
		else:
			print "!!! ML-WARNING: SAVING Cl THAT HAVE WRONG FORMAT. => CHECK RUN PARAMETERS !!!"
		if len(self.effCl_means)== self.l_max: 
			toSave = []
			toSave.append(self.effCl_means)
			toSave.append(self.effCl_errors)
			np.savetxt(nPath+"atmEffClRUNS"+str(self.NUMBER_RAN_RUNS)+"_StandardAtm.txt", toSave) ### effCl
			print "effCl data successfully saved..."+nPath+"atmEffClRUNS"+str(self.NUMBER_RAN_RUNS)+"_StandardAtm.txt"
		else:
			print "!!! ML-WARNING: SAVING effCl THAT HAVE WRONG FORMAT. => CHECK RUN PARAMETERS !!!"

		if alm==True:
			if len(self.alm_means) >= (1./2.*self.l_max*(self.l_max+1)):
				toSave = []
				toSave.append(self.alm_means)
				toSave.append(self.alm_errors)
				np.savetxt(nPath+"atmAlmRUNS"+str(self.NUMBER_RAN_RUNS)+"_StandardAtm.txt", toSave) ### Alm
				print "Alm data successfully saved..."+nPath+"atmAlmRUNS"+str(self.NUMBER_RAN_RUNS)+"_StandardAtm.txt"
			else:
				print "!!! ML-WARNING: SAVING Alm VALUES, THAT HAVE WRONG FORMAT. => CHECK RUN PARAMETERS !!!"


	#### SET NEW PLOTTING COLOR	
	def setPltColor(self, nColor):
		self.pltColor = nColor

	
	#### READS FILE AND ADDS ITS VALUES TO EXISTING ATTRIBUTE (USED FOR CONDOR) 			##### NEW #####
	def readAlmAbsList(self, RunsPerFile, nPath):
		if not os.path.exists(nPath+"_almAllAbs_"+str(RunsPerFile)+".txt"):
			print "File "+nPath+"_almAllAbs_"+str(RunsPerFile)+".txt was not found => skipped!"
		else:
			New_alm_all  = np.loadtxt(nPath+"_almAllAbs_"+str(RunsPerFile)+".txt",unpack=False)
			self.alm_all_abs.append(New_alm_all)


	#### READS FILE AND ADDS ITS VALUES TO EXISTING ATTRIBUTE (USED FOR CONDOR) 			##### NEW #####
	def readAlmPhaseList(self, RunsPerFile, nPath):
		if not os.path.exists(nPath+"_almAllPhase_"+str(RunsPerFile)+".txt"):
			print "File "+nPath+"_almAllPhase_"+str(RunsPerFile)+".txt was not found => skipped!"
		else:
			New_alm_all  = np.loadtxt(nPath+"_almAllPhase_"+str(RunsPerFile)+".txt",unpack=False)
			self.alm_all_phase.append(New_alm_all)
			

	#### CALCULATES Cl AND effCl FROM CONDOR-ALM-LIST							##### NEW #####
	def getClEffClFromAlmList(self):
		self.cl_all=[]
		self.effCl_all=[]
		for i in range(0, len(self.alm_all)):
			self.cl_all.append(retCl(self.alm_all[i]))
			self.effCl_all.append(retEffCl(self.alm_all[i], lmax=self.l_max))

			sys.stdout.write("\r calc. all param. from run... "+str(i+1)+"            ")
			sys.stdout.flush()
		sys.stdout.write("\n")


	#### SAVE FILE INFO ABOUT NUMBER OF SOURCES AND NUMBER OF SIGNAL EVENTS
	def saveSignalInfo(self, nTitle,prefix=""):
		np.savetxt(nTitle+"_signalInfo_"+str(self.NUMBER_RAN_RUNS)+".txt", self.generatedSigEvents) ### Cl
		#~ print "detailed signal information successfully saved to... "+nTitle+"_signalInfo_"+str(self.NUMBER_RAN_RUNS)+".txt"
		#~ np.savetxt(nTitle+"_input_"+str(self.NUMBER_RAN_RUNS)+".txt", self.input_parameters) ### Cl
		#~ print "input parameters successfully saved to... "+nTitle+"_input_"+str(self.NUMBER_RAN_RUNS)+".txt"

	
	#### SAVE Cl, effCl LIST FOR ALL RUNS					##### NEW #####
	def saveCleffClList(self, nTitle="",prefix=""):
		if self.SAVE_ALL_DATA:
			np.savetxt(nTitle+"clAll_"+prefix+".txt", self.cl_all) ### Cl
			print "Cl data successfully saved to... "+nTitle+"_clAll_"+prefix+".txt"
			np.savetxt(nTitle+"al0All_"+prefix+".txt", self.al0_all)
			#~ print "al0 data successfully saved to... "+nTitle+"_al0All.txt"
			
		np.savetxt(nTitle+"_effClAll_"+prefix+".txt", self.effCl_all)
		print "effCl data successfully saved to... "+nTitle+"_effClAll_"+prefix+".txt"
		
		self.saveSignalInfo(nTitle)

		
	#### SAVE Cl_log LIST FOR ALL RUNS						##### NEW #####
	def saveClLogList(self, nTitle="",prefix=""):
		np.savetxt(nTitle+"_clLogAll_"+prefix+".txt", self.cl_log_all) ### Cl
		print "ClLog data successfully saved to... "+nTitle+"_clLogAll_"+prefix+".txt"
		
	
	#### SAVE Alm abs and phase LIST FOR ALL RUNS					##### NEW #####
	def saveAlms(self, nTitle="",prefix=""):
		np.savetxt(nTitle+"_almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt", absList(self.alm_all)) ### Cl
		print "Alm data successfully saved to... "+nTitle+"_almAllAbs_"+str(self.NUMBER_RAN_RUNS)+".txt"
		np.savetxt(nTitle+"_almAllPhase_"+str(self.NUMBER_RAN_RUNS)+".txt", phaseList(self.alm_all)) ### Cl
		print "Alm data successfully saved to... "+nTitle+"_almAllPhase_"+str(self.NUMBER_RAN_RUNS)+".txt"
		
		
	#### READ Cl, effCl LIST FOR ALL RUNS					##### NEW #####
	def readCleffClList(self, runsPerProcess, nTitle="", unpack=False, signalInf=False, cut=0, less=False, norm_l=0):
		if signalInf and os.path.exists(nTitle+"_signalInfo_"+str(runsPerProcess)+".txt"):
			curNsig = np.loadtxt(nTitle+"_signalInfo_"+str(runsPerProcess)+".txt", usecols=[0])
		else:
			curNsig=1
			
		if less:
			cutBool = bool(curNsig<=cut)
		else:
			cutBool = bool(curNsig>cut)
				
		if self.SAVE_ALL_DATA:   ## In Standart configuration SAVE_ALL_DATA is False
			if os.path.exists(nTitle+"_clAll_"+str(runsPerProcess)+".txt"):
				curCl = np.loadtxt(nTitle+"_clAll_"+str(runsPerProcess)+".txt",unpack=unpack)
				if norm_l:
					#norm_val = np.mean(curCl[norm_l-norm_range:norm_l])
					norm_val = np.sum(curCl[1:norm_l])
				else:
					norm_val = 1
				print norm_val
				if type(curCl[0]) == list or type(curCl[0]) == np.ndarray:
					for i in range(0,len(curCl)):
						self.cl_all.append(curCl[i])
						#print "current: "+str(len(self.cl_all))
				else:
					self.cl_all.append(curCl*1./norm_val)
				self.successFileRead = self.successFileRead +1
				#print "Cl data successfully read from... "+nTitle+"_clAll_10.txt"

		
		if os.path.exists(nTitle+"_effClAll_"+str(runsPerProcess)+".txt") and os.stat(nTitle+"_effClAll_"+str(runsPerProcess)+".txt").st_size != 0:
			if cutBool:
				curEffCl = np.loadtxt(nTitle+"_effClAll_"+str(runsPerProcess)+".txt", unpack=False)
				if norm_l:
					norm_val = np.sum(curEffCl[1:norm_l])
				else:
					norm_val = 1
				#~ print len(curEffCl), max(curEffCl), min(curEffCl)
				if type(curEffCl[0]) == list or type(curEffCl[0]) == np.ndarray:
					for k in range(0,len(curEffCl)):
						self.effCl_all.append(curEffCl[k])
				else:
					self.effCl_all.append(curEffCl*1./norm_val)
					#print "effCl data successfully read from... "+nTitle+"_effClAll_10.txt"
			return True
		else:
			#print "File "+nTitle+"_effClAll_"+str(runsPerProcess)+".txt was not found => skipped!"
			self.failedFileRead = self.failedFileRead +1
			self.failedFileNumber.append(nTitle)
			if os.path.exists(nTitle+"_effClAll_"+str(runsPerProcess)+".txt") and os.stat(nTitle+"_effClAll_"+str(runsPerProcess)+".txt").st_size == 0:
				self.emptyfile+=1
			return False

		
			
	#### READ Cl_log LIST FOR ALL RUNS						##### NEW #####
	def readClLogList(self, runsPerProcess, nTitle="", unpack=True):
		if not os.path.exists(nTitle+"_clLogAll_"+str(runsPerProcess)+".txt"):
			print "File "+nTitle+"_clLogAll_"+str(runsPerProcess)+".txt was not found => skipped!"
		else:
			curCl_log = np.genfromtxt(nTitle+"_clLogAll_"+str(runsPerProcess)+".txt",unpack=unpack)
			if type(curCl_log[0]) == list or type(curCl_log[0]) == np.ndarray:
				for i in range(0,len(curCl_log)):
					self.cl_log_all.append(curCl_log[i])
					#print "current: "+str(len(self.cl_all))
				print len(curCl_log[0])
			else:
				self.cl_log_all.append(curCl_log)


	#### SAVE Cl, effCl LIST FOR ALL RUNS								##### NEW #####
	def saveLastSkymap(self, nTitle="", dataType="numpy"):
		fileName = nTitle+"_Skymap_Ev"+str(self.NUMBER_SIMULATED_EVENTS)+"Nside"+str(self.nside)
		if dataType == "txt":
			np.savetxt(fileName+".txt", self.map_delta, fmt="%8g")
			print "full Skymap successfully saved to... "+fileName+".txt"
		else:
			#cPickle.dump(self.map_delta, open(fileName+".numpy", "wb"))
			self.map_delta.astype(np.float32).dump(open(fileName+".numpy", 'wb'))
			print "full Skymap successfully saved to... "+fileName+".numpy"

		
	#### CALCULATE HISTOGRAM OF Cl CORRELATIONS ####
	def calcCorrCoeff(self, plot=False):
		self.corrCoeffCl = np.corrcoef(self.cl_all, rowvar=0)
		self.corrCoeffEffCl = np.corrcoef(self.effCl_all, rowvar=0)
		
		self.all_coeff_cl = []
		self.all_coeff_effCl = []
		self.extreme_coeff_cl = []
		self.extreme_coeff_effCl = []
		for i in range(0, self.l_max+1):
			for k in range(0, self.l_max+1):
				if not k==i: 
					self.all_coeff_cl.append(self.corrCoeffCl[i][k])
					if self.corrCoeffCl[i][k] > 0.2:
						self.extreme_coeff_cl.append([i, k])
					if not (k==0 or i==0):
						self.all_coeff_effCl.append(self.corrCoeffEffCl[i-1][k-1])
						if self.corrCoeffEffCl[i-1][k-1] > 0.2:
							self.extreme_coeff_effCl.append([i, k])
				else:
					self.corrCoeffCl[i][k] = 0.0
					if not k==0:
						self.corrCoeffEffCl[i-1][k-1] = 0.0
		self.histoClCorrCoeff, self.edgesClCorrCoeff= np.histogram(self.all_coeff_cl, bins=100, range=(-1.0,1.0))
		self.histoEffClCorrCoeff, self.edgesEffClCorrCoeff= np.histogram(self.all_coeff_effCl, bins=100, range=(-1.0,1.0))
		
		
		
	#### CALCULATE HISTOGRAM OF Cl CORRELATIONS ####
	def plotCorrCoeff(self, nFigure=-1, opt="Cl", plane=False):
		plotControl(nFigure)

		if opt=="Cl":
			if plane == False:
				plt.plot(setNewEdges(self.edgesClCorrCoeff), self.histoClCorrCoeff, drawstyle='steps-mid', color=self.pltColor, label="corr. coeff.")
				plt.title(r" corr. coeff. distribution $ r(C_{l}) $ ")
				plt.xlabel(r" corr. coeff. $ r(C_{l}) $")
				plt.ylabel(r" counts ")
			elif plane == True:
				plt.imshow(self.corrCoeffCl, interpolation="nearest", vmin=-0.6, vmax=0.6)#vmin=min([min(self.all_coeff_cl), min(self.all_coeff_effCl)]), vmax=max([max(self.all_coeff_cl), max(self.all_coeff_effCl)]))
				plt.xlabel(r" $ \ell_{1} $ ", fontsize=26)
				plt.ylabel(r" $ \ell_{2} $ ", fontsize=26)		
				plt.title(r" corr. coeff $ r_{\ell_{1}, \ell_{2}}(C_{\ell}) $ (without ($ \ell, \ell $)-pairs)", fontsize=20)
				cb = plt.colorbar()
						
		elif opt=="effCl":
			if plane == False:
				plt.plot(setNewEdges(self.edgesEffClCorrCoeff), self.histoEffClCorrCoeff, drawstyle='steps-mid', color=self.pltColor, label="corr. coeff.")
				plt.title(r" corr. coeff. distribution $ r(C_{l}^{\mathrm{eff}}) $")
				plt.xlabel(r" corr. coeff. $ r(C_{l}^{\mathrm{eff}}) $")
				plt.ylabel(r" counts ")
			elif plane == True:
				plt.imshow(self.corrCoeffEffCl, interpolation="nearest", vmin=-0.6, vmax=0.6)#vmin=min([min(self.all_coeff_cl), min(self.all_coeff_effCl)]), vmax=max([max(self.all_coeff_cl), max(self.all_coeff_effCl)]))
				plt.xlabel(r" $ \ell_{1} $ ", fontsize=26)
				plt.ylabel(r" $ \ell_{2} $ ", fontsize=26)		
				plt.title(r" corr. coeff $ r_{\ell_{1}, \ell_{2}}(C_{\ell}^{\mathrm{eff}}) $ (without ($ \ell, \ell $)-pairs)", fontsize=20)
				cb = plt.colorbar()



	##### PLOT "AUTOCORRELATION COEFFICIENT" OF Cl
	def plotAutoCorrCoeff(self, nFigure=-1, opt="Cl", delta=[1,2,3,5], labelAdd=""):
		plotControl(nFigure)
		col = ["k","b","r","g","c"]
		if opt=="Cl":
			self.autoCorrCoeff_Cl = []
			for i in range(0,len(delta)):
				self.autoCorrCoeff_Cl.append([])
				for l in range(0, self.l_max-delta[i]+1):
					self.autoCorrCoeff_Cl[i].append(self.corrCoeffCl[l][l+delta[i]])	
				plt.plot([k for k in range(0,self.l_max-delta[i]+1)], self.autoCorrCoeff_Cl[i], color=col[i], linestyle=linestyles[i],label=r" $ r_{\ell, \ell + \Delta \ell } $  for $ \ell = "+str(delta[i])+r" $ "+labelAdd)
			plt.title(r" autocorrelation coefficient for $ C_{\ell} $")
			plt.xlabel(r" $ \ell $ ")
			plt.ylabel(r" $ r_{\ell, \ell + \Delta \ell } (C_{\ell})$")
			
		if opt=="effCl":
			self.autoCorrCoeff_effCl = []
			for i in range(0,len(delta)):
				self.autoCorrCoeff_effCl.append([])
				for l in range(0, self.l_max-delta[i]):
					self.autoCorrCoeff_effCl[i].append(self.corrCoeffEffCl[l][l+delta[i]])	
				plt.plot([k for k in range(0,self.l_max-delta[i])], self.autoCorrCoeff_effCl[i], color=col[i], linestyle=linestyles[i],label=r" $\Delta\ell = "+str(delta[i])+r" $ "+labelAdd)
			plt.title(r" autocorrelation coefficient for $ C_{\ell}^{\mathrm{eff}} $")
			plt.xlabel(r" $ \ell $ ")
			plt.ylabel(r" $ r \left( C_{\ell}^{\mathrm{eff}},\, C_{\ell+\Delta \ell}^{\mathrm{eff}} \right) $")			
				
	
	
	##### READS COMPLETE SKYMAP FROM FILE #####
	def readSkymap(self, nTitle, unpack=False, dataType=""):
		fileName = nTitle+"_Skymap_Ev"+str(self.NUMBER_SIMULATED_EVENTS)+"Nside"+str(self.nside)
		if dataType == "":
			if os.path.exists(fileName+".txt") :
				dataType = "txt"
			elif os.path.exists(fileName+".numpy"):
				dataType = "numpy"
			else:
				print "File "+fileName+".txt / .numpy was not found => skipped!"
				return False
		
		#print "DATA-TYPE: "+str(dataType)
		if dataType == "txt":
			self.map_delta = np.genfromtxt(fileName+".txt",unpack=unpack)
		elif dataType == "numpy":
			self.map_delta = np.load(open(fileName+".numpy", 'rb'))
			#self.map_delta = cPickle.load(open(fileName+".numpy", "rb"))
			self.map_delta = np.float64(self.map_delta)
		else:
			print "ML-ERROR: Data format given not known: "+str(dataType)
			


	#### SETTING THE MilagroAzimuth BOOLEAN VALUE
	def setMilagroAzimuth(self, boolean, pathName=""):
		self.MilagroAzimuth = boolean
		print "Set Milagro to be "+str(boolean)
		if boolean == True:
			print "Read file: "+str(pathName)
			self.MilagroSkymap = np.genfromtxt(pathName,unpack=False)


	#### SETTING THE GalacticPlane BOOLEAN VALUE
	def setGalacticPlane(self, boolean, pathName=""):
		self.GalacticPlane = boolean
		print "Set GalacticPlane to be "+str(boolean)
		if boolean == True:
			print "Read file: "+str(pathName)
			self.GalacticPlaneSkymap = np.genfromtxt(pathName,unpack=False)
			

	#### SETTING THE FixedZenith BOOLEAN VALUE
	def setFixedZenith(self, boolean, pathName=""):
		self.FixedZenith = boolean
		if boolean == True:
			print "Use Zenith List from experimental data: "+str(boolean)
			print "Read file: "+str(pathName)
			self.FixedZenithList = np.genfromtxt(pathName,unpack=False)

	#### SETTING RA ACCEPTANCE #####
	def setRAAcc(self, boolean, pathName=""):
		self.RA_ACC = boolean
		if boolean == True:
			print "Use experimental RA ACCEPTANCE: "+str(boolean)
			print "Read file: "+str(pathName)
			self.RA_ACC_hi, self.RA_ACC_ed = np.genfromtxt(pathName,unpack=False)
			self.RA_ACC_hi = normHist(self.RA_ACC_hi,1.0)
			self.RA_ACC_spline = interpolate.InterpolatedUnivariateSpline(self.RA_ACC_ed, self.RA_ACC_hi, k=1)

	#### SET MONTE CARLO DATA SET ####
	def setMCSample(self, mc_sample, astro, conv, mc_bool, gamma=-2.07):
		self.MC_GEN_BOOL = mc_bool
		self.GAMMA = gamma
		#self.mc_sample_full = mc_sample
		self.weight_astro  = astro
		self.weight_conv = conv
		
		#~ _, indices = np.unique(mc_sample["trueE"], return_index=True)
		self.mc_sample_full = mc_sample #np.delete(mc_sample, np.where(np.in1d(range(len(mc_sample["trueE"])), indices)==False)[0], 0)
		#~ print "test vals"
		#~ print len(self.weight_astro)
		#~ print len(self.mc_sample_full)
			
		print "NOTE: set MC sample, length="+str(len(self.mc_sample_full))
	
	
	#### SETTING ENERGY USE TO BE TRUE OR FALSE
	def setUseE(self, boolean):
		self.useE = boolean
		if boolean == True:
			print "NOTE: Set useE to be "+str(boolean)
		
				
	#### SET ENERGY DISTRIBUTION FOR ATM EVENTS
	def setAtmEnergyDistr(self,fileName):
		self.atmE_Histo, self.atmE_Edges = np.genfromtxt(fileName, unpack=False)
		self.atmE_Histo = normHist(self.atmE_Histo,fac=0.95)
		#self.atmE_Histo_ORIGINAL, self.atmE_Edges_ORIGINAL = np.genfromtxt(fileName, unpack=False)
		#self.genSigE_Histo = [0.0 for i in range(0,len(self.atmE_Histo))]
		self.atmEnergySpline = interpolate.InterpolatedUnivariateSpline(self.atmE_Edges, self.atmE_Histo)
	
						
	#### SET ENERGY DISTRIBUTION FOR ATM EVENTS								### new new
	def setSignalEnergyDistr(self,fileName):
		self.sigE_Histo, self.sigE_Edges = np.genfromtxt(fileName, unpack=False)
		self.sigE_Histo = normHist(self.sigE_Histo,fac=0.95)
		self.sigEnergySpline = interpolate.InterpolatedUnivariateSpline(self.sigE_Edges, self.sigE_Histo)
		
	#### SET NUMBER OF SIMULATED EVENTS
	def setNumSigEvents(self, nEvents):
		if nEvents <0.:
			print "Warning: negative Number of Signal Events!"
		if nEvents > self.NUMBER_SIMULATED_EVENTS:
			print "Warning: Number of Signal Events larger than Number of Simulated Events!"
		else:
			self.NumSigEvents=nEvents
			self.numSigEvents_Poi=np.random.poisson(lam=nEvents)
			
	#### SET SPECTRAL INDEX GAMMA FOR ENERGY ####
	def setSpectralIndex(self, g):
		self.GAMMA = g
		if self.GAMMA > 0:
			print "SIGN ERROR FOR SPECTRAL INDEX, GAMMA SHOULD BE NEGATIVE"
	
	#### CALCULATE MU REAL ####
	def calcMuReal(self, mu, path="realMuTables_1samples_E-2.dict"):
		path2 		 = localPath+"MCode/data/"
		muRealDict = np.load(open(path2+path, 'rb'))
		muReal     = 0.0
		for i in range(0,len(muRealDict["Mu"+str(mu)][0])):
				muReal += muRealDict["Mu"+str(mu)][0][i]*i
		self.curMuReal = muReal
		
	#### SET MU REAL CORRECTIONS ####
	#### DONT USE, SEE SETACCEPTANCESPLINE() ####
	#def setMuCorrection(self, a, b):
	#	self.slope = a
	#	self.ax    = b
	
	#### SET NMAX ####
	def setNMax(self):
		self.NMax = int(round(self.NumSigEvents*1./self.curMuReal))
		self.NMax_Poi = np.random.poisson(lam=self.NMax)
	
	
	#### CHOOSE ZENITH BAND OR ALL SKY ### NEEDED FOR ZENITH BAND CALCULATIONS ####
	def chooseZenBand(self, interval=[-1.,0.]):
		self.ZEN_BAND = interval
		print "CHOSE ZENITH BAND INSTEAD OF HEMISPHERE! "+str(self.ZEN_BAND)
		
	
	#### CHOOSE SPECIFIC M OR NOT ####
	def setSpecM(self, spec_m=False):
		self.SPEC_M = spec_m
		
	
	#### GENERATES RANDOM ENERGY ACCORDING TO DATA DISTRIBUTION
	def getAtmEventEnergy(self):
		found = False
		while (found == False):
			E = np.random.uniform(self.Emin_GeV, self.Emax_GeV)
			if np.random.uniform(0.0,1.0) < self.atmEnergySpline(E):
				return E
					
	#### GENERATES RANDOM ENERGY ACCORDING TO DATA DISTRIBUTION
	def getSignalEventEnergy(self):											### new new
		found = False
		count = 0
		delta_ed = (self.atmE_Edges[1]-self.atmE_Edges[0])
		while (found == False):
			E = np.random.uniform(self.Emin_GeV, self.Emax_GeV)
			if np.random.uniform(0.0,1.0) < self.sigEnergySpline(E):
				found = True
				return E
				
				#### NEW ADD-ON TO REDUCE ATM. DISTR. ON THE FLY ####
				#for k in range(0,len(self.atmE_Edges)):
				#	if E < self.atmE_Edges[k]+0.5*delta_ed or k == len(self.atmE_Edges)-1: 
				#		if self.atmE_Histo[k] > 0.5*sum(self.atmE_Histo_ORIGINAL)*1.0/self.NUMBER_SIMULATED_EVENTS or count >= 25:
				#			self.atmE_Histo[k] -= sum(self.atmE_Histo_ORIGINAL)*1.0/self.NUMBER_SIMULATED_EVENTS
				#			found = True
				#			if count >= 25: 
				#				print "Found no appropriate signal event in exp. E distr. => took atmospheric!"
				#				return self.getAtmEventEnergy()
				#			else: return E
				#			break
				#		else:
				#			found = False
				#			count += 1
				#			break
			
		
			
###########################################
##########  SUBCLASS FOR   ################
########## VAR. ATM. SPEC. ################
###########################################

class varZenithAnalysis(multiPoleAnalysis):
	#### CONSTRUCTOR ###
	def __init__(self, nZenithFilePath, nEvents, nReadDir, nSaveDir, nNEvents, nNRanRuns, nNZenPercMods, nZenTotMod, nlmax=1000, nPSSmearingPath=""):

		multiPoleAnalysis.__init__(self, nZenithFilePath, nEvents, nReadDir, nSaveDir, nNEvents=nNEvents, nlmax=nlmax, nPSSmearingPath=nPSSmearingPath, nNRanRuns=nNRanRuns)
		
		self.changePerc = []
		self.DsquaredDistr = []
		self.effDsquaredDistr = []
		self.almDsquaredDistr = []
		self.almEffDsquaredDistr = []

		self.NUMBER_RAN_RUNS = nNRanRuns
		self.NUMBER_RAN_RUNS_CHANGED = nNRanRuns

		self.ZEN_PERC_MOD_NO = nNZenPercMods
		self.ZEN_PERC_MOD_TOT = nZenTotMod
		self.halfPerc = (self.ZEN_PERC_MOD_TOT*1.0)/2

		self.histoZenithDistrAtm_ORIGINAL = self.histoZenithDistrAtm[:]


	#### CALC NEW PERCENTAGE BY curChange NUMBER ####
	def calcNewPerc(self,nCurChange):
		self.curPerc = (self.ZEN_PERC_MOD_TOT/(self.ZEN_PERC_MOD_NO-1)*nCurChange)-self.halfPerc
		self.curChange = nCurChange
		return self.curPerc



	#### SET NEW PERCENTAGE #####
	def setNewPerc(self,nCurPerc):
		self.curPerc = nCurPerc
		self.curChange = "NaN"
		return True


	#### GENERATE NEW ZENITH SPEC. BY CHANING ATM BY self.curChange ####
	def genNewSpec(self):
		print "CurChange Percentage: "+str(self.curPerc)

		self.histoZenithDistrAtm = normHist(self.histoZenithDistrAtm, 0.95)
		self.histoZenithDistrAtm = changeDistrPerc(self.histoZenithDistrAtm, self.curPerc)
		self.histoZenithDistrAtm = normHist(self.histoZenithDistrAtm, 0.95)

		self.atmosSpline = interpolate.InterpolatedUnivariateSpline(self.edgesZenithDistrAtm, self.histoZenithDistrAtm)
		
		
	##### MIXES SPECTRUM FOR ATMOSPHERICAL AND BACKGROUND SPLINE ####
	def genMixedSpec(self, signalHisto, signalPercentage):
		self.curPerc = signalPercentage
		if type(signalHisto) == str:
			signalHisto, signalEdges = np.genfromtxt(signalHisto, unpack=False)
			signalHisto = signalHisto[0:30]
		print "Using signal("+str(signalPercentage*100)+"%)+background("+str((1-signalPercentage)*100)+"%) mixed skymap..."
		if len(signalHisto) == len(self.histoZenithDistrAtm):
			for i in range(0,len(signalHisto)):
				print str(self.histoZenithDistrAtm_ORIGINAL[i])+" - "+str(signalHisto[i]) 
				self.histoZenithDistrAtm[i] = (1.0-signalPercentage)*self.histoZenithDistrAtm_ORIGINAL[i] + signalPercentage*signalHisto[i]
				
			return True
		else:
			print "WARNING: Histograms not of same binning => mixing skipped!"
			return False
				

	#### GENERATE NEW ZENITH SPEC. BY CHANING ATM BY self.curChange ####
	def reformSpec(self):
		self.histoZenithDistrAtm = self.histoZenithDistrAtm_ORIGINAL[:]


	#### PLOT ZENITH DISTR. ####
	def plotZenithSpec(self,nFigure=-1, changed=False):		
		if changed==True:
			multiPoleAnalysis.plotZenithSpec(self, nFigure, nLabel="atm. changed "+str(round(self.curPerc,3)*100)+"%")
			plt.title("atm. event distr., tot.Perc-Var.="+str(self.ZEN_PERC_MOD_TOT)+"%")
		else:
			multiPoleAnalysis.plotZenithSpec(self, nFigure)



	#### PLOT NORM(ALM) PLANE OF MEANS
	def pltNormAlmPlane(self, nFigure=-1, changed=False):
		multiPoleAnalysis.pltNormAlmPlane(self, nFigure)
		if changed==True:
			plt.title("Norm(Alm) Distr., tot.Perc-Var.="+str(self.ZEN_PERC_MOD_TOT)+"%")


	#### SAVE Alm LIST		#### DON'T USE!!!
	def saveAlmList(self, nPath=""):
		if nPath =="":
			nPATH = self.SAVE_DIR
		#saveAlmList(self.alm_all, nPath+"FULL_AlmNormListsAtm_curPerc"+str(self.curPerc)+"RUNS"+str(self.NUMBER_RAN_RUNS_CHANGED)+".txt")
		saveAlmList(self.alm_all, nPath+"AlmNormListsAtm_curPerc"+str(self.curPerc)+"RUNS"+str(self.NUMBER_RAN_RUNS_CHANGED)+".txt")
		print "saving all Alm-Lists to given file..."


	### READ Alm LIST		#### DON'T USE!!!
	def readAlmList(self, nPath=""):
		if nPath =="":
			nPath = self.READ_DIR
		print "reading all Alm-Lists from given file..."

		self.alm_all  = np.genfromtxt(nPath+"FULL_AlmNormListsAtm_curPerc"+str(self.curPerc)+"RUNS"+str(self.NUMBER_RAN_RUNS_CHANGED)+".txt",unpack=False)
		print "... done."

		self.cl_all=[]
		self.effCl_all=[]
		for i in range(0, len(self.alm_all)):
			self.cl_all.append(retCl(self.alm_all[i]))
			self.effCl_all.append(retEffCl(self.alm_all[i], lmax=self.l_max))

			sys.stdout.write("\r calc. all param. from run... "+str(i+1)+"            ")
			sys.stdout.flush()
		sys.stdout.write("\n")
		



	
###########################################
##########  SUBCLASS FOR   ################
########## SIM. PS.-SIGNAL ################
###########################################

class sigAnalysis(multiPoleAnalysis):
	#### CONSTRUCTOR ###
	def __init__(self, nEvents, nReadDir, nSaveDir, detector, nNEvents, nNSources, nMuSources, nVarNSources,  nNRanRuns=1, nlmax=1000, nPSSmearingPath=""):

		multiPoleAnalysis.__init__(self, nEvents, nReadDir, nSaveDir, detector, nNEvents=nNEvents, nlmax=nlmax, nPSSmearingPath=nPSSmearingPath)
		
		self.changePerc = []
		self.DsquaredDistr = []
		self.effDsquaredDistr = []
		self.almDsquaredDistr = []
		self.almEffDsquaredDistr = []
		self.sigElist=[]
		self.NUMBER_RAN_RUNS = nNRanRuns
		self.NUMBER_RAN_RUNS_CHANGED = nNRanRuns

		self.N_SOURCES = nNSources
		self.MU_SOURCES  = nMuSources
		self.VARIATIONS_N_SOURCE = nVarNSources
		self.phi_sou   = []
		self.theta_sou = []
		self.theta = []
		self.muSim = []
		self.extended_source_angle=0.

			
		self.PS_STRETCHING = 1.0
		
	def setsigZenith(self, sigZenPath):
		self.signalZenithPath = sigZenPath
		self.histoZenithDistrAtm_ORIGINAL = self.histoZenithDistrAtm[:]
		self.atmZenithPath = self.zenithFilePath
		
		self.histoZenithDistrSig 	= []
		self.edgesZenithDistrSig 	= []
		self.signalSpline 			= []
		for i in range(0, len(sigZenPath)):
			histoZenithDistrSig, edgesZenithDistrSig = np.genfromtxt(str(self.signalZenithPath[i]), unpack=False)
			histoZenithDistrSig = normHist(histoZenithDistrSig, 0.95)
			signalSpline = interpolate.InterpolatedUnivariateSpline(edgesZenithDistrSig, histoZenithDistrSig)
		
			self.histoZenithDistrSig.append(copy.copy(histoZenithDistrSig))
			self.edgesZenithDistrSig.append(copy.copy(edgesZenithDistrSig))
			self.signalSpline.append(copy.copy(signalSpline))
		
	#### SETS STRETCHING FACTOR TO MAKE RECONSTRUCTION/PS BETTER OR WORSE ####
	def setPSStretching(self, nVal):
		self.PS_STRETCHING = nVal
		print "NOTE: Stretching Value is NOT 1.0 any more, but: "+str(nVal)
		
	def setmusim(self, musim):
		self.muSim=musim
		
	@property
	def extended_source(self):
		return self.extended_source_angle

	@extended_source.setter
	def extended_source(self,val):
		self.extended_source_angle=val


	#### SETS STRETCHING FACTOR TO MAKE RECONSTRUCTION/PS BETTER OR WORSE ####
	def setMuPreciseBool(self, nVal):
		self.MuPrecise = nVal
		print "NOTE: Mu might not be taken as poissonian. MuPrecise is set to: "+str(nVal)
		
		
	#### SETS STRETCHING FACTOR TO MAKE RECONSTRUCTION/PS BETTER OR WORSE ####
	def setMultiMuSources(self, nBool, nMu_add, nNSou_add):
		self.multiSourceBool = nBool
		if nBool == True:
			self.multiSourceExpansion = [nMu_add, nNSou_add]
		else:
			self.multiSourceExpansion = [[], []]
		print "NOTE: Multi-Mu-Sources have been activated: "+str(nBool)
		print self.multiSourceExpansion


	#### GETS A POINTSPREAD SPLINE FROM FILE ####
	def getPSSpline(self, nFile, quickRandomTables=True):
		self.histoPointSpread 		= []
		self.edgesPointSpread		= []
		self.PSSpline				= []
		self.PSQuickRandomTables	= []
	
		for i in range(0, len(nFile)):
			histoPointSpread, edgesPointSpread = np.genfromtxt(nFile[i], unpack=False)
			PSSpline = interpolate.InterpolatedUnivariateSpline(edgesPointSpread, histoPointSpread)
			if quickRandomTables == True:
				PSQuickRandomTables = getTablesFromSpline(PSSpline)
				
			self.histoPointSpread.append(histoPointSpread)
			self.edgesPointSpread.append(edgesPointSpread)
			self.PSSpline.append(PSSpline)
			if quickRandomTables == True:
				self.PSQuickRandomTables.append(PSQuickRandomTables)

	#### GETS PS+ZEN SPLINE FROM DICT #### OUTDATED!!!! DON'T USE!!!!
	def getPS_zen_spline(self, dct_name): 
		sig_dict = np.load(open(dct_name), "rb")
		keys     = sig_dict.keys()
		all_zen  = np.zeros(len(sig_dict[keys[0]][0]))
		edges    = sig_dict[keys[0]][1]
		
		for k in keys:
				all_zen += sig_dict[k][0]
				
		all_zen = normHist(all_zen, 0.95)
		self.PSzenSpline = interpolate.InterpolatedUnivariateSpline(edges, all_zen)
		
	def setAcceptanceSpline(self):
		"""
		-Read zenith values from MC, plot trueZen where zen above horizon
		-Get acceptance spline
		-Get overall slope for calculation mu_sim->mu_real
		"""
		#nBins=100
		#nRange=(-1., 0.)
		#n, bins = np.histogram(np.cos(dec2zen(self.mc_sample_full[np.where(self.mc_sample_full["dec"]>=0)]["trueDec"])), bins=nBins, 
		#												weights=self.weight_astro[np.where(self.mc_sample_full["dec"]>=0)], range=nRange)
		#self.slope = np.mean(n*0.95/max(n))
		#~ print "slope="+str(self.slope)
		#self.acc_spline = interpolate.InterpolatedUnivariateSpline(setNewEdges(bins), n*0.95/max(n))
		pkl_file = open(localPath+"MCode/data/"+self.DETECTOR+"/sim/gamma"+str(self.GAMMA_BF)+"/acceptanceSpline.pkl", 'rb')
		self.acc_spline = pickle.load(pkl_file) 
		self.slope=np.load(localPath+"MCode/data/acceptancespline.npy")[self.DETECTOR]

	def setPSReplaceMode(self, rep):
		self.nRep = bool(rep)
		print "set replace mode ..."+str(self.nRep)
		
	def setRotationMode(self, rot):
		self.psRotation = bool(rot)
		print "PS rotation is ..."+str(self.psRotation)

	#### VARIES NSources BY A GIVEN FORMULA ####
	def calcNewNSources(self, nCurChange):
		self.curNSources = int(self.N_SOURCES - (self.N_SOURCES*1.0/(self.VARIATIONS_N_SOURCE-1))*nCurChange)
		self.curMuSources = self.MU_SOURCES
		return self.curNSources


	#### VARIES NSources BY A GIVEN FORMULA ####
	def calcNewMuSources(self, nCurChange, noZero=False):
		if noZero == True:
			self.curMuSources  = self.MU_SOURCES - (self.MU_SOURCES*1.0/(self.VARIATIONS_N_SOURCE))*nCurChange #int()
		else:
			self.curMuSources  = self.MU_SOURCES - (self.MU_SOURCES*1.0/(self.VARIATIONS_N_SOURCE-1))*nCurChange #int()
		self.curNSources = self.N_SOURCES
		return self.curMuSources


	#### GENERATES SIGNAL SKYMAPS AND ANALYSES THEM ####
	def createRanSigSkymaps(self):
		self.warnIfFullsphere()
		self.cl_all = []
		self.alm_all= []
		self.cl_log_all = []
		self.effCl_all = []
		self.al0_all = []
		self.firstAlms_abs = []
		self.firstAlms_phi = []
		histoZenithDistrSig_all = []
		print "Creating signal Skymaps, NSource: "+str(self.curNSources)+" Mu: "+str(self.curMuSources)
	
		for curRun in range(0, self.NUMBER_RAN_RUNS_CHANGED):
			sys.stdout.write("\r RUN: "+str(curRun+1)+"            ")
			sys.stdout.flush()
			if self.multiSourceBool == False:
				print "curMuSources"
				print self.curMuSources
				self.ranTheta, self.ranPhi, sigNeutrinos  = self.getMapList(self.NUMBER_SIMULATED_EVENTS, self.curNSources, self.curMuSources, self.PSSpline, atmZenith=self.atmZenithPath, signalZenith=self.signalZenithPath)
			else:
				self.ranTheta, self.ranPhi, sigNeutrinos  = self.getMapList(self.NUMBER_SIMULATED_EVENTS, [self.curNSources]+self.multiSourceExpansion[1],  [self.curMuSources]+self.multiSourceExpansion[0], self.PSSpline, atmZenith=self.atmZenithPath, signalZenith=self.signalZenithPath)
			self.generatedSigEvents.append(sigNeutrinos)

			#### CREATE SKYMAP AND ANALYSE Cl, alm SPECTRA ####
			if self.useE: self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi, self.energyList)
			else: self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi)
			self.analyseMap()
			
			histoZenithDistrSig, edgesZenithDistrSig = np.histogram(np.cos(self.ranTheta), bins=30, range=(-1.,0.), normed=True)	
			histoZenithDistrSig_all.append(histoZenithDistrSig)

		sys.stdout.write("\n")		
		self.histoZenithDistrSig_mean = np.mean(histoZenithDistrSig_all, axis=0)
		self.histoZenithDistrSig_meanErr = np.std(histoZenithDistrSig_all, axis=0)*1.0/(np.sqrt(self.NUMBER_RAN_RUNS_CHANGED))


	#### ZENITH SPEC. OF SIGNAL SKYMAPS
	def plotSigZenithSpec(self, nFigure=-1):
		plotControl(nFigure)
		plt.errorbar(self.edgesZenithDistrAtm, normHist(self.histoZenithDistrSig_mean, 0.95),fmt="o", color=self.pltColor, label="N-Sources: "+str(self.curNSources)) #, yerr=self.histoZenithDistrSig_meanErr


	#### SAVE Alm LIST
	def saveAlmList(self, nPath=""):
		if nPath =="":
			nPATH = self.SAVE_DIR
		saveAlmList(self.alm_all, nPath+"FULL_AlmNormListsSig_NSou"+str(self.curNSources)+"Mu"+str(self.curMuSources)+"RUNS"+str(self.NUMBER_RAN_RUNS_CHANGED)+".txt")

	####OVERRIDE ENERGY SAVING		
	def saveEventEnergy(self, nTitle="",prefix="",  histogrammsave=True):
		s_txt=str(nTitle)+"Energy_"+prefix+".txt"
		if histogrammsave==True:
			histo_bg, histo_diff, histo_sig, edges = self.makeEventHisto(self.energyList, self.diffElist, self.sigElist)
			np.savetxt(s_txt, np.array([histo_bg, histo_diff, histo_sig, setNewEdges(edges)]))
		else:
			print len(self.energyList)
			print "Saved to {}".format(s_txt)
			np.savetxt(s_txt, np.array(self.energyList))		

		


	def makeEventHisto(self, bge, diffe, sige, bins=80):
		if self.DETECTOR=="IC59":
			hmin = -2.
			hmax = 2.
			hbins=40
		else:
			hmin = 1.
			hmax = 9.
		hbg, ebg = np.histogram(bge, bins=bins, range=(hmin, hmax))
		hdiff, ediff = np.histogram(diffe, bins=bins, range=(hmin, hmax))
		hsig, esig = np.histogram(sige, bins=bins, range=(hmin, hmax))
		if len(ebg) != len(ediff) or len(ebg) != len(esig):
			print "Energy bin ERROR!"
			return 0., 0., 0.
		else:
			return hbg, hdiff, hsig, ebg

	### READ Alm LIST
	def readAlmList(self, nPath=""):
		if nPath =="":
			nPath = self.READ_DIR
		print "reading all Alm-Lists from given file..."

		self.alm_all  = np.genfromtxt(nPath+"FULL_AlmNormListsSig_NSou"+str(self.curNSources)+"Mu"+str(self.curMuSources)+"RUNS"+str(self.NUMBER_RAN_RUNS_CHANGED)+".txt",unpack=False)
		print "... done."

		self.cl_all=[]
		self.effCl_all=[]
		for i in range(0, len(self.alm_all)):
			self.cl_all.append(retCl(self.alm_all[i]), lmax=self.l_max)
			self.effCl_all.append(retEffCl(self.alm_all[i], lmax=self.l_max))

			sys.stdout.write("\r calc. all param. from run... "+str(i+1)+"            ")
			sys.stdout.flush()
		sys.stdout.write("\n")



	#### GENERATES SIGNAL SKYMAPS AND ANALYSES THEM ####      ######## NEW ##########
	def createPureSigSkymaps(self):
		self.warnIfFullsphere()
		self.cl_all = []
		self.alm_all= []
		self.cl_log_all = []
		self.effCl_all = []
		self.al0_all = []
		self.firstAlms_abs = []
		self.firstAlms_phi = []
		
		histoZenithDistrSig_all = []
		print "Creating pure signal Skymaps, NEvents: "+str(self.NUMBER_SIMULATED_EVENTS)+" Mu: "+str(self.curMuSources)
	
		for curRun in range(0, self.NUMBER_RAN_RUNS_CHANGED):
			sys.stdout.write("\r RUN: "+str(curRun+1)+"            ")
			sys.stdout.flush()
			self.ranTheta, self.ranPhi, sigNeutrinos   = self.getPureSigMapList(self.NUMBER_SIMULATED_EVENTS, 0, self.curMuSources, self.PSSpline, self.signalSpline, useN=False)
			hit_bool_sig = H.ang2pix(self.nside,self.ranTheta,self.ranPhi)
			
			self.generatedSigEvents.append(sigNeutrinos)
			if self.useE: self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi, self.energyList)
			else: self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi)
			self.analyseMap()

			histoZenithDistrSig, edgesZenithDistrSig = np.histogram(np.cos(self.ranTheta), bins=30, range=(-1.,0.))	
			histoZenithDistrSig_all.append(histoZenithDistrSig)

		sys.stdout.write("\n")		
		self.histoZenithDistrSig_mean = np.mean(histoZenithDistrSig_all, axis=0)
		self.histoZenithDistrSig_meanErr = np.std(histoZenithDistrSig_all, axis=0)*1.0/(np.sqrt(self.NUMBER_RAN_RUNS_CHANGED))

	#### GENERATE FULL SIGNAL+BG MAPS (VS. SIGNAL+DIFF+BG) #### LJS
	def createSigBGSkymaps(self, useNsou=True, shuff=False, analyse=True):
		self.warnIfFullsphere()
		self.cl_all = []
		self.alm_all= []
		self.cl_log_all = []
		self.effCl_all = []
		self.al0_all = []
		self.firstAlms_abs = []
		self.firstAlms_phi = []
		
		histoZenithDistrSig_all = []
		
		#~ self.input_parameters = []
		
		print "Creating Full Signal+BG Skymaps, NEvents: "+str(self.NUMBER_SIMULATED_EVENTS)+" Mu: "+str(self.curMuSources)
		
		
		for curRun in range(0, self.NUMBER_RAN_RUNS_CHANGED):
			sys.stdout.write("\n RUN: "+str(curRun+1)+"          \n")
			sys.stdout.flush()
			#~ print "self.NumSigEvents="+str(self.NumSigEvents)
			#~ print "self.curNSources="+str(self.curNSources)
			#~ print "self.curMuSources="+str(self.curMuSources)
			#self.mccopy=self.mc_sample_full.copy()
			self.ranTheta, self.ranPhi, sigNeutrinos   = self.getPureSigMapList(self.NumSigEvents, self.curNSources, self.curMuSources, self.PSSpline, self.signalSpline, useN=useNsou)
			#~ self.input_parameters.append([self.NumSigEvents, self.curNSources, self.curMuSources])
			#~ plt.figure(10)
			#~ plt.hist(np.cos(self.ranTheta), bins=30, color="blue")
			#bgNEvents= self.NUMBER_SIMULATED_EVENTS-sigNeutrinos[0]
			#bgNEvents=0  UNCOMMENT THIS TO GET NO BACKGROUND 
			#print "bgNEvents" + str(bgNEvents)
			if self.MC_GEN_BOOL != True:
				self.bg_list = self.getAtmosNu(bgNEvents, self.atmosSpline, self.fullSphereMode)
				print "Length bg_list: " + str(len(self.bg_list))

				#print "vorher: " + str(self.ranTheta[0])
				self.ranTheta=self.ranTheta.tolist()
				self.ranPhi=self.ranPhi.tolist()
				#print "nachher: " + str(self.ranTheta[0])
				
				for i in self.bg_list:
					self.ranTheta.append(i[0])
					self.ranPhi.append(i[1])
					#~ self.ranTheta=np.append(self.ranTheta, self.bg_list[i][0])
					#~ self.ranPhi=np.append(self.ranPhi, self.bg_list[i][1])
				
			else:
				diffThetaPhi, ranThetaPhi = self.createBackground(self.mc_sample_full, bgNEvents, 0.)
				self.ranTheta = np.concatenate([self.ranTheta, ranThetaPhi[0]])   # join signal and background events
				self.ranPhi = np.concatenate([self.ranPhi, ranThetaPhi[1]])
			
			#~ plt.figure(10)
			#~ plt.hist(np.cos(self.ranTheta), bins=30, color="green", alpha=0.5)
			print "length theta: " + str(len(self.ranTheta))# + "; length phi: " +str(len(self.ranPhi))
			
			if shuff==True:
				shuffle(self.ranPhi)
			
			hit_bool_sig = H.ang2pix(self.nside, zen2dec(self.ranTheta), self.ranPhi)
			print "Atmospheric Neutrinos : " +str(bgNEvents)
			print "Total number of events in this map is: " +str(len(self.ranTheta))
			
			self.generatedSigEvents.append(sigNeutrinos)
			if self.useE: self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi, self.energyList)
			else: self.hitBool2Map(zen2dec(self.ranTheta), self.ranPhi)
			if analyse==True:
				self.analyseMap()
				
			histoZenithDistrSig, edgesZenithDistrSig = np.histogram(np.cos(self.ranTheta), bins=30, range=(-1.,0.))	
			histoZenithDistrSig_all.append(histoZenithDistrSig)

		sys.stdout.write("\n")		
		self.histoZenithDistrSig_mean = np.mean(histoZenithDistrSig_all, axis=0)
		self.histoZenithDistrSig_meanErr = np.std(histoZenithDistrSig_all, axis=0)*1.0/(np.sqrt(self.NUMBER_RAN_RUNS_CHANGED))

	#### PLOTS ONE SIGNAL SKYMAP FROM SCRATCH ####                               ######## NEW ##########
	def plotOnePureSigSkymap(self, nFigure=-1):
		plotControl(nFigure)

		print "Creating pure signal Skymaps, NEvents: "+str(self.NUMBER_SIMULATED_EVENTS)+" Mu: "+str(self.curMuSources)
		ranThetaSig, ranPhiSig  = self.getPureSigMapList(self.NUMBER_SIMULATED_EVENTS, self.curMuSources, self.PSSpline, self.signalSpline)
		hit_bool_sig = H.ang2pix(self.nside,ranThetaSig,ranPhiSig)
	
		self.map_delta = np.zeros(self.npix)
		for i in range(0, len(hit_bool_sig)):
			self.map_delta[hit_bool_sig[i]] += 1.0

		H.mollview(self.map_delta)


	#### PLOTS ONE RAN. SKYMAP FROM SCRATCH ####                                   ######## NEW ##########
	def plotOneSkymap(self, nFigure=-1):
		plotControl(nFigure)

		ranThetaSig, ranPhiSig, sigNeutrinos  = self.getMapList(self.NUMBER_SIMULATED_EVENTS, self.curNSources, self.curMuSources, self.PSSpline)

		hit_bool_sig = H.ang2pix(self.nside,ranThetaSig,ranPhiSig)
	
		self.map_delta = self.hitBool2Map(zen2dec(ranThetaSig), ranPhiSig)

		H.mollview(self.map_delta)



	#### SETS curNSources ####                                                     ######## NEW ##########
	def setCurNSources(self, nCurNSources):
		self.curNSources = nCurNSources


	#### SETS curMuSources ####                                                    ######## NEW ##########
	def setCurMuSources(self, nCurMuSources):
		self.curMuSources = nCurMuSources



	#### GENERATES EVENTS FOR POINTSOURCES ####
	def getMapList(self, N_nu, N_sources, mu, Source_PS_Spline, atmZenith, signalZenith):
		map_list = []
		countMeasuredSources=0.
		##### SET SOURCE POSITION
		if self.fullSphereMode == True: 
			cutoffSimul = 1.0
			cutoffAccep = 1.0
		else: 
			cutoffSimul = self.ZEN_BAND[1] ####  former 1.0
			cutoffAccep = self.ZEN_BAND[1] ####  former 0.0 (!)
		
		startSimul    = self.ZEN_BAND[0] #### former -1
		countMeasuredSources = 0
		self.allSignalNeutrinos = []
		
		#if type(N_sources) == int and type(mu) != list:
			#N_sources = [N_sources]
			#mu   = [mu]
		#elif type(N_sources) == list and type(mu) == list:
			#print "NOTE: Simulate muli-mu-source!"
		#elif type(N_sources) == list or type(mu) == list:
			#print "ML-ERROR: N_sources and mu don't have both list character nor do both have non-list character!"
		#else: "ML-ERROR: Critical but not understood ..." 
		
		if self.MC_GEN_BOOL != True:    ### Standard is MC_GEN_BOOL=True
			for i in range(0,len(N_sources)):
				if not N_sources[i] == 1:
					for n in range(0, N_sources[i]):
						#~ muSim_p = np.random.poisson(muSim[n])
						if self.GalacticPlane == False:
							self.gen_one_pointsource_mu(mu[i], map_list, self.signalSpline, Source_PS_Spline, MuPrecise=self.MuPrecise)
						else:
							self.gen_one_galacticPlaneSource(mu[i],map_list,self.signalSpline, Source_PS_Spline, MuPrecise=self.MuPrecise)
						countMeasuredSources+=1
						
				else:
					#~ muSim_p = np.random.poisson(muSim[0])
					if self.GalacticPlane == False:
						self.gen_one_pointsource_mu(mu[i], map_list, self.signalSpline, Source_PS_Spline, MuPrecise=self.MuPrecise)
					elif self.GalacticPlane == True:
						self.gen_one_galacticPlaneSource(mu[i],map_list,self.signalSpline, Source_PS_Spline, MuPrecise=self.MuPrecise)
			###############Create Point Sources Here. Point Source Injection Above is not up to date anymore.############################################					
		else:   
			for i in range(0,len(N_sources)):
				if len(self.muSim)==0:
					theta=np.random.uniform(startSimul, cutoffSimul, N_sources[i])
					self.phi_sou = np.random.uniform(0., 2*np.pi, N_sources[i])
					self.theta_sou = np.arccos(theta)
					muSim = []
					R = 0.
					for t in theta:
						R += self.acc_spline(t)      # get the expected mean source strength before acceptance  checks = "real" source strength 
						#~ print "R="+str(R)
					if R>0:
						M = mu[i]*N_sources[i]*1./R 		#		M = mu[i]*self.slope*N_sources[i]*1./R
					for t in theta:
						muSim.append(self.acc_spline(t)*M)    ### calculate simulation values
					print "Simulated Signal Neutrinos per Source:"
					print muSim
					self.muSim=[muSim]
				
				if self.extended_source_angle!=0.0:
					print "Create Extended Sources...."
					theta_new=[]
					phi_new=[]
					mus_new=[]
					for j,val in enumerate(self.theta_sou[i]):
						ret=sample_from_circle_on_sphere(math.ceil(self.muSim[i][j]),self.extended_source_angle,np.degrees(val-np.pi/2),np.degrees(self.phi_sou[i][j]))
						phi_new.extend(ret[0])
						theta_new.extend([t+np.pi/2 for t in ret[1]]) ####Normal Theta Coordinates
						mus_new.extend([self.muSim[i][j]/len(ret[0])]*len(ret[0]))
					mu_poi=np.random.poisson(mus_new)
					N_fin=len(mu_poi)
				else:
					mus_new=self.muSim[i]
					theta_new=self.theta_sou[i]
					phi_new=self.phi_sou[i]
					mu_poi = np.random.poisson(self.muSim[i])
					N_fin=N_sources[i]
				#print "mu_poi is"
				#print mu_poi
				#print "N_sources[i]"
				#print N_sources[i]
				#print "self.theta_sou"
				#print self.theta_sou
				#print "self.phi_sou"
				#print self.phi_sou
				#print "self.MuPrecise"
				#print self.MuPrecise
				self.ps_sample = self.point_source_injection(self.mc_sample_full, mu_poi, N_fin, map_list, theta_new, phi_new, MuPrecise=self.MuPrecise, nReplace=self.nRep, rotation=self.psRotation)
				countMeasuredSources += N_fin
			
				
		retPhis=[i[1] for i in map_list]
		retThetas=[i[0] for i in map_list]		
		print "MAP-LIST-LEN: "+str(len(map_list)) #+" for mu="+str(mu[i])+", N="+str(N_sources[i])
	
		sigNeutrinos = len(map_list)
		self.allSignalNeutrinos = copy.deepcopy(map_list)
		
		###############Create Diffuse Astrophysical Background############################################
		
		if self.MC_GEN_BOOL != True:
			if self.useDiffBG:
				print "CURRENT MU: "+str(mu[0])
				print str(int(round(mu[0]*N_sources[0])))+" ("+str(mu[0]*N_sources[0])+")"
				nDif=self.NumSigEvents-int(round(mu[0]*N_sources[0]))
				print "nDif vorher: " +str(nDif)
				if nDif>0: nDif=np.random.poisson(lam=nDif)
				else : nDif=0
				print "nDif nachher: " +str(nDif)
				if nDif>0:
					newMap=self.getDiffBG(nDif, self.signalSpline, self.fullSphereMode)
					map_list=map_list+newMap
			
			if N_nu-len(map_list) > 0:
				nDif=0
				newMap = self.getAtmosNu(N_nu-len(map_list),self.atmosSpline, fullSphereMode=self.fullSphereMode, Milagro=self.MilagroAzimuth, RA_ACC=self.RA_ACC)    
				map_list = map_list+newMap
				
		else:
			if self.useDiffBG:
				print str(int(round(mu[0]*N_sources[0])))+" ("+str(round(mu[0]*N_sources[0],3))+")"
				nDif=self.NumSigEvents-int(round(mu[0]*N_sources[0]))
				print "nDif vorher: " +str(nDif)
				if nDif>0: nDif=np.random.poisson(lam=nDif)
				else : nDif=0
				print "nDif nachher: " +str(nDif)
				if nDif>0:		
					diffThetaPhi, ranThetaPhi = self.createBackground(self.mc_sample_full, N_nu-nDif-sigNeutrinos, nDif)
				else:
					diffThetaPhi, ranThetaPhi = self.createBackground(self.mc_sample_full, N_nu-sigNeutrinos, 0.)
				retPhis.extend(diffThetaPhi[1])
				retPhis.extend(ranThetaPhi[1])
				retThetas.extend(diffThetaPhi[0])
				retThetas.extend(ranThetaPhi[0])					
			else:
				nDif=0
				diffThetaPhi, ranThetaPhi = self.createBackground(self.mc_sample_full, N_nu-sigNeutrinos, 0.)
				retPhis.extend(diffThetaPhi[1])
				retPhis.extend(ranThetaPhi[1])
				retThetas.extend(diffThetaPhi[0])
				retThetas.extend(ranThetaPhi[0])

		if countMeasuredSources > 0:
			average = sigNeutrinos*1.0/sum(N_sources)
		else:
			average = 0.0
		
		print "Measured Sources: "+str(countMeasuredSources)
		print "Signal Neutrinos: " +str(sigNeutrinos)
		if self.useDiffBG:
			print "Astro Neutrinos: " +str(sigNeutrinos+nDif)
		print "Atmospheric Neutrinos: "+str(N_nu-nDif-sigNeutrinos)
		print "All: "+str(len(retThetas))
		print ""
		#~ retPhis = [float(i[1]) for i in map_list]
		#~ retThetas = [float(i[0]) for i in map_list]

		signalInfo = [int(sigNeutrinos), int(sum(N_sources)), int(countMeasuredSources), int(sum(mu)), float(average)]
		ret  = [np.array(retThetas), np.array(retPhis), np.array(signalInfo)]
		return ret


	#### RETURNS RANDOM THETA VALUES ####      #### DON'T EEEEEEVVEEER USE!!! (MUCH TOO SLOW)
	def get_ran_theta(self, Num,Spline):
		def get_dpsi_from_hitnmiss_linear(Anz,Sp,Yug,Yog,Xug,Xog):	
			count = 0
			ran_dpsi = np.zeros(Anz)
			while count < Anz:
				y_t = np.random.uniform(Yug,Yog)
				dpsi_temp = np.random.uniform(Xug,Xog)
				if y_t < Sp(dpsi_temp):
					ran_dpsi[count] = dpsi_temp
					count = count + 1
			return ran_dpsi
		return get_dpsi_from_hitnmiss_linear(Num,Spline,0.,1.,0., 45.)


	#### GENERATE nSOU SOURCES WITH DIFFERENT MU FROM MC ####
	def point_source_injection(self, mc_IC86, Mu, Nsou, Map_List, nTheta=[], nPhi=[], MuPrecise=False, nReplace=True, rotation=True):
		"""
		- Mu, nTheta, nPhi all have each #Nsou items
		"""

		inj = PointSourceInjector2(self.GAMMA)
		### GAMMA isnt used at the moment, weights have to be changed in createSkymap script
		### it is then used for both point sources and diffuse astro events
		inj.fill(0.0, mc_IC86)
		inj.sinDec_bandwidth = np.radians(1)

		inj.src_dec = np.radians(60) #"random" value for generating empty structured array = "sample"

		dec = np.array(nTheta)-np.pi/2.                                               #conversion from zenith to declination
		#~ print type(nPhi), nPhi
		
		sample = inj.sample(0, poisson=False, replace=False, rotate_this=rotation).next()                     # empty sample
		temp_sample = inj.sample(0, poisson=False, replace=False, rotate_this=rotation).next()                # empty temp sample
		temp_E = []
		for i in range(0, Nsou):                                                            # loop over sources
			#~ print "i_sou="+str(i)
			inj.src_dec = dec[i]                                              # set source position
			inj.src_ra = nPhi[i]
			temp_sample = inj.sample(Mu[i], poisson=MuPrecise, replace=nReplace, rotate_this=rotation).next()
			sample = np.concatenate((sample, temp_sample),axis=0) 
			temp_E = temp_sample["logE"]           # load energy
			#~ print "energy="+str(temp_E)
			#~ print len(temp_E)
			self.energyList.extend(temp_E)
			self.sigElist.extend(temp_E)        
			self.MuReal_all.append(len(temp_E))    # append number of neutrinos per source
		
		#~ print "shape dec "+str(np.shape(sample["dec"]))
		#~ print "dec "+str(sample["dec"])
    
		Map_List.extend(np.transpose([dec2zen(sample["dec"]), sample["ra"]])) #np.array(
		#~ print "shape/ transpose content:"
		#~ print np.shape(np.transpose([dec2zen(sample["dec"]), sample["ra"]]))
		#~ print np.transpose([dec2zen(sample["dec"]), sample["ra"]])
		print "Sources with zero neutrinos: "+str(len(np.where(Mu<1.)[0]))
		return sample
		
    
  ##### GENERATES A SINGLE POINT SOURCE
	def gen_one_pointsource(self, Mu,Map_List,Spline_Zenith,Spline_Pointspread, nTheta, Lifetime=1., MuPrecise=False):
		print "use gen_one_pointsource"
		if self.fullSphereMode == True: 
			cutoffSimul = 1.0
			cutoffAccep = 1.0
		else: 
			cutoffSimul = 0.0 ####  former 1.0
			cutoffAccep = np.cos(np.pi/2-np.radians(5)) ####  former 0.0 (!)
		
		theta = nTheta		
		phi = np.random.uniform(0., 2*np.pi)
		self.phi_sou.append(phi)

		##### GENERATE NEUTRINOS FROM SOURCE
		#~ if MuPrecise:
		allNeutrinosPerSource = Mu
		#~ else:
		#~ allNeutrinosPerSource = np.random.poisson(Mu*Lifetime) 
			
		thisSourceHas 	= 0
		wrongHemisphere = 0
		
		#### MAKE SURE THAT EVERY NEUTRINO IS ACCEPTED; GENERATE AS LONG AS UNTIL THIS IS FULFILLED ####
		
		while thisSourceHas < allNeutrinosPerSource:
			#~ print "len Map_List: "+str(len(Map_List))
			#### CALCULATE NEW SOURCE STRENGTH FROM DIFFERENCE BETWEEN "DEBIT AND CREDIT" ####
			NeutrinosPerSource = allNeutrinosPerSource-thisSourceHas

			##### GET RANDOM ARRIVAL DIRECTION, ROTATE TO RIGHT POSITION FOR EASIER USE #####
			ran_phi    = np.random.uniform(0, 2*np.pi, NeutrinosPerSource)
			ran_theta	 = []
			ran_sample = []
			
			
			for i in range(0, NeutrinosPerSource):
				##### DETERMINE DETECTOR FOR THE MEASUREMENT RANDOMLY ===> ran_sample[] #####
				ran_choice 		= np.random.uniform(0.0,1.0)
			
				if self.samAeffContr[0] > ran_choice: sample = 0 													#"IC86-I"
				elif self.samAeffContr[1]+self.samAeffContr[0] > ran_choice: sample = 1 	#-----
				elif sum(self.samAeffContr) > ran_choice: sample = 2 											#-----
				else: print "ML-ERROR: Could not find matching detector for point source generation!"
				
				ran_theta.extend(np.pi*1.0/180.0*np.array(quickRandomNumbers(self.PSSpline[sample], self.PSQuickRandomTables[sample], 1)))
				ran_sample.append(sample)

			##### STRETCH RECONSTRUCTION ERROR TO SIMULATE FULL SPHERE #####
			ran_theta 		= self.PS_STRETCHING*np.array(ran_theta)
			hit				= False
			

			##### ROTATE TO SOURCE POSITION AND CHECK DETECTOR ACCEPTANCE #####
			for ran_t,ran_p,ran_sam in zip(ran_theta,ran_phi,ran_sample):

				##### RA ACCEPTANCE (SOURCE RA) #####
				if self.RA_ACC == True:
					z = np.random.uniform(0.0,1.0)
					if z >= self.RA_ACC_spline(ran_p):
						print "SIGNAL event kicked by RA acceptance."
						self.RA_ACC_rejectSIG += 1
						continue
						
				##### GET EVENT POSITION ON MAP #####
				z=(np.cos(theta)*np.cos(ran_t))-(np.sin(theta)*np.sin(ran_t)*np.sin(ran_p))
				x=0
				y=0
				x =-(np.sin(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.cos(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.cos(phi)*np.sin(theta)*np.cos(ran_t))
				y = (np.cos(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.sin(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.sin(phi)*np.sin(theta)*np.cos(ran_t))
				cosFillT = z
				
				##### DETECTOR ACCEPTANCE CHECK #####
				if (np.random.uniform(0.,1.) < Spline_Zenith[ran_sam]((-1.0)*abs(cosFillT)) or MuPrecise==True) and cosFillT <= cutoffAccep:
					fill_t = np.arccos(z)
					fill_p = np.arctan2(y,x)
					
					if (fill_p < 0):
						fill_p = 2*np.pi+fill_p
					#~ if self.useE:	
					fill_E = self.getSignalEventEnergy()	
					self.energyList.append(fill_E)
					self.sigElist.append(fill_E)
						
					##### INSERT EVENT INTO MAP IN RADIANS(!) #####
					Map_List.append([fill_t,fill_p])
					hit = True
					thisSourceHas += 1
					#~ self.sigCheck.append(fill_t)
					self.evPerDetector[ran_sam] += 1
					
				#~ if not cosFillT <= cutoffAccep: wrongHemisphere +=1
			
		#~ print "Check: This source has "+str(thisSourceHas)+" Neutrinos! It should have "+str(allNeutrinosPerSource)
		#~ print ""
			
		self.MuReal_all.append(thisSourceHas)
		#~ self.wrongHemi_all.append(wrongHemisphere)
		return True

	##### GENERATES A SINGLE POINT SOURCE BASED ON MU
	def gen_one_pointsource_mu(self, Mu, Map_List, Spline_Zenith, Spline_Pointspread, Lifetime=1., MuPrecise=False):

		##### SET SOURCE POSITION
		if self.fullSphereMode == True: 
			cutoffSimul = 1.0
			cutoffAccep = 1.0
		else: 
			cutoffSimul = self.ZEN_BAND[1] ####  former 1.0
			cutoffAccep = np.cos(np.pi/2-np.radians(5))               ####  former 0.0 (!)
	
		startSimul    = self.ZEN_BAND[0]
		
		theta = np.arccos(np.random.uniform(startSimul, cutoffSimul)) ### cutoff 5deg above horizon
		phi = np.random.uniform(0., 2*np.pi)

		##### GENERATE NEUTRINOS FROM SOURCE
		if MuPrecise:
			NeutrinosPerSource = Mu
		else:
			#~ print Mu, Lifetime
			NeutrinosPerSource = np.random.poisson(Mu*Lifetime) 

		##### GET RANDOM ARRIVAL DIRECTION #####
		ran_phi   		= np.random.uniform(0, 2*np.pi,NeutrinosPerSource)
		ran_theta		= []
		ran_sample		= []
		
		for i in range(0, NeutrinosPerSource):
			##### DETERMINE DETECTOR FOR THE MEASUREMENT RANDOMLY #####
			ran_choice 		= np.random.uniform(0.0,1.0)
		
			if self.samAeffContr[0] > ran_choice: sample = 0 							            #"IC86-I"
			elif self.samAeffContr[1]+self.samAeffContr[0] > ran_choice: sample = 1 	#"---"
			elif sum(self.samAeffContr) > ran_choice: sample = 2 						          #"---"
			else: print "ML-ERROR: Could not find matching detector for point source generation!"
			
			ran_theta.extend(np.pi*1.0/180.0*np.array(quickRandomNumbers(self.PSSpline[sample], self.PSQuickRandomTables[sample], 1)))
			ran_sample.append(sample)

		##### STRETCH RECONSTRUCTION ERROR TO SIMULATE FULL SPHERE #####
		ran_theta 		= self.PS_STRETCHING*np.array(ran_theta)
		hit				= False
		thisSourceHas 	= 0

		##### ROTATE TO SOURCE POSITION AND CHECK DETECTOR ACCEPTANCE #####
		for ran_t,ran_p,ran_sam in zip(ran_theta,ran_phi,ran_sample):

			##### RA ACCEPTANCE (SOURCE RA) #####
			if self.RA_ACC == True:
				z = np.random.uniform(0.0,1.0)
				if z >= self.RA_ACC_spline(ran_p):
					print "SIGNAL event kicked by RA acceptance."
					self.RA_ACC_rejectSIG += 1
					continue
					
			##### GET EVENT POSITION ON MAP #####
			z=(np.cos(theta)*np.cos(ran_t))-(np.sin(theta)*np.sin(ran_t)*np.sin(ran_p))
			x=0
			y=0
			x =-(np.sin(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.cos(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.cos(phi)*np.sin(theta)*np.cos(ran_t))
			y = (np.cos(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.sin(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.sin(phi)*np.sin(theta)*np.cos(ran_t))
			cosFillT = z
			
			##### DETECTOR ACCEPTANCE CHECK #####
			if (np.random.uniform(0.,1.) < Spline_Zenith[ran_sam]((-1.0)*abs(cosFillT)) or MuPrecise==True) and cosFillT <= cutoffAccep:
				fill_t = np.arccos(z)
				fill_p = np.arctan2(y,x)
				
				if (fill_p < 0):
					fill_p = 2*np.pi+fill_p
				#~ if self.useE:	
				fill_E = self.getSignalEventEnergy()	
				self.energyList.append(fill_E)
				self.sigElist.append(fill_E)
					
				##### INSERT EVENT INTO MAP IN RADIANS(!) #####
				Map_List.append([fill_t,fill_p])
				hit = True
				thisSourceHas += 1
				#~ self.sigCheck.append(fill_t)
				self.evPerDetector[ran_sam] += 1
				
		self.MuReal_all.append(thisSourceHas)
		return hit

	##### GENERATES A SINGLE POINT SOURCE
	def gen_one_galacticPlaneSource(self, Mu,Map_List,Spline_Zenith,Spline_Pointspread,Lifetime=1., MuPrecise=False):

		##### SET SOURCE POSITION
		if self.fullSphereMode == True: 
			cutoffSimul = 1.0
			cutoffAccep = 1.0
		else: 
			cutoffSimul = 1.0 ####  former np.cos(85.0*np.pi/180.0)
			cutoffAccep = np.cos(np.pi/2-np.radians(5)) ####  former 0.0 (!)
		
		foundPlane = False
		while foundPlane == False:
			theta = np.arccos(np.random.uniform(-1., cutoffSimul)) ### cutoff 5deg above horizon
			phi = np.random.uniform(0., 2*np.pi)
			#if phi <= 0.0: RA = phi+2.0*np.pi
			#else: 
			RA = phi
			dec = np.pi-theta
			pix = H.ang2pix(self.nside,dec,RA)
			if self.GalacticPlaneSkymap[pix] >= np.random.uniform(0.0,1.0): foundPlane = True # = gauss(3sigma)

		##### GENERATE NEUTRINOS FROM SOURCE
		if MuPrecise:
			NeutrinosPerSource = Mu
		else:
			NeutrinosPerSource = np.random.poisson(Mu*Lifetime) 
		ran_phi   = np.random.uniform(0, 2*np.pi,NeutrinosPerSource)
		ran_theta = np.pi*1.0/180.0*np.array(quickRandomNumbers(self.PSSpline, self.PSQuickRandomTables, NeutrinosPerSource))
		
		##### STRETCH RECONSTRUCTION ERROR TO SIMULATE FULL SPHERE #####
		ran_theta = self.PS_STRETCHING*ran_theta
		hit=False
		thisSourceHas = 0

		##### ROTATE TO SOURCE POSITION AND CHECK DETECTOR ACCEPTANCE #####
		for ran_t,ran_p in zip(ran_theta,ran_phi):
			z=(np.cos(theta)*np.cos(ran_t))-(np.sin(theta)*np.sin(ran_t)*np.sin(ran_p))
			x=0
			y=0
			x =-(np.sin(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.cos(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.cos(phi)*np.sin(theta)*np.cos(ran_t))
			y = (np.cos(phi)*np.sin(ran_t)*np.cos(ran_p))+(np.sin(phi)*np.cos(theta)*np.sin(ran_t)*np.sin(ran_p))+(np.sin(phi)*np.sin(theta)*np.cos(ran_t))
			cosFillT = z
			##### DETECTOR ACCEPTANCE CHECK #####
			if (np.random.uniform(0.,1.) < Spline_Zenith((-1.0)*abs(cosFillT)) or MuPrecise==True) and cosFillT <= cutoffAccep:
				fill_t = np.arccos(z)
				fill_p = np.arctan2(y,x)
				if (fill_p < 0):
					fill_p = 2*np.pi+fill_p
				##### INSERT EVENT INTO MAP IN RADIANS(!) #####
				Map_List.append([fill_t,fill_p])
				hit = True
				thisSourceHas += 1
				#~ self.sigCheck.append(fill_t)
				
		self.MuReal_all.append(thisSourceHas)
		return hit
	
	def setSourcePos(self, otheta, theta_sou, phi_sou):
		self.theta=otheta
		self.phi_sou=phi_sou
		self.theta_sou=theta_sou
			

	#### CREATES PURE SIGNAL SKYMAP                                      ##### NEW #####
	def getPureSigMapList(self, N_nu, curN, curMu, Source_PS_Spline, signalSpline, useN=True):
		map_list = []
		countMeasuredSources = 0
		
		print "curN: "+str(curN)
		#~ curMu = self.NumSigEvents*1./curN
		#~ print "CurMu_real: " +str(curMu)
		#~ curMu = (curMu-self.ax)/self.slope
		#~ print "CurMu: " +str(curMu)
		
		##### SET SOURCE POSITION
		if self.fullSphereMode == True: 
			cutoffSimul = 1.0
			cutoffAccep = 1.0
		else: 
			cutoffSimul = self.ZEN_BAND[1] ####  former 1.0
			cutoffAccep = np.cos(np.pi/2-np.radians(6)) ####  former 0.0 (!)
	
		startSimul    = self.ZEN_BAND[0]
		
		
		if useN:
			if (len(self.phi_sou)==0) and (len(self.theta_sou)==0):
				theta = np.random.uniform(startSimul, cutoffSimul, curN)
				self.phi_sou = np.random.uniform(0., 2*np.pi, curN)
				self.theta_sou = np.arccos(theta)
			else:
				print "Position Variables are set globally"
				theta=self.theta
				
			muSim = []
			
			if self.SPEC_M==True:
				R = 0.
				for t in theta:
					R += self.acc_spline(t)      # get the expected mean source strength before acceptance checks = "real" source strength 
					#~ print "R="+str(R)
				M = self.NumSigEvents*1./R		
				#~ print "R="+str(R)
				#~ print "M="+str(M)	
			else:
				M = N_nu*1./(curN)	#(curN*self.slope) equals R in that case
			#~ check=0.	
			for t in theta:
				muSim.append(self.acc_spline(t)*M)        # calculate mu for simulation depending on single source position	
				#~ print "R Ratio "+str(self.acc_spline(t)/R)
				#~ check += self.acc_spline(t)/R
			#~ 
			#~ print "check (1)="+str(check)
			#~ print "MuSim: "+str(muSim)+" sum="+str(sum(muSim))
			
			
			
			if self.MC_GEN_BOOL != True:
				for i in range(0,curN): 
					mu = np.random.poisson(muSim[i])
					#~ print "Mu for Simulation is "+str(mu)
					if self.GalacticPlane == False:
						self.gen_one_pointsource(mu,map_list,signalSpline, Source_PS_Spline, self.theta_sou[i], MuPrecise=self.MuPrecise)
					elif self.GalacticPlane == True:
						self.gen_one_galacticPlaneSource(mu,map_list,signalSpline, Source_PS_Spline, self.theta_sou[i], MuPrecise=self.MuPrecise)						
					countMeasuredSources += 1
					
			else:
				### THIS IS THE NEW POINT SOURCE MC GENERATION ###
				mu = np.random.poisson(muSim)
				self.ps_sample = self.point_source_injection(self.mc_sample_full, mu, curN, map_list, self.theta_sou, self.phi_sou, 
																										 MuPrecise=self.MuPrecise, nReplace=self.nRep, rotation=self.psRotation)
				#print "map list in gpsml: "
				#print map_list
				countMeasuredSources = curN
				
		else:
			print "N_nu = "+str(N_nu) ###### = self.numSigEvents_Poi
			while(N_nu > len(map_list)):
				countMeasuredSources += 1
				allHitsAccepted = False
				while(not allHitsAccepted): 
					if self.GalacticPlane == False:
						if self.gen_one_pointsource_mu(curMu, map_list, signalSpline, Source_PS_Spline, MuPrecise=self.MuPrecise):
							allHitsAccepted = True
					elif self.GalacticPlane == True:
						print "PROBABLY NOT WORKING"
						if self.gen_one_galacticPlaneSource(curMu, map_list,signalSpline, Source_PS_Spline,MuPrecise=self.MuPrecise):
							allHitsAccepted = True
			#~ print "countMeasuredSources: "+str(countMeasuredSources)	
				
				if N_nu < len(map_list):
					while(N_nu < len(map_list)):
						ranIndex = np.random.random_integers(0,len(map_list)-1)
						del map_list[ranIndex]
						del self.energyList[ranIndex]
						del self.sigElist[ranIndex]
						mu_cum = np.cumsum(self.MuReal_all)
						it = np.where(mu_cum>ranIndex)[0][0]
						self.MuReal_all[it] -= 1
				
		sys.stdout.write("\n")
		#~ print "shape map list "+str(np.shape(map_list))
		print " => total number events: "+str(len(map_list))+" (mean "+str(self.NumSigEvents)+")"
		sigNeutrinos = len(map_list)
		self.allSignalNeutrinos = copy.deepcopy(map_list)
		
		retPhis = [i[1] for i in map_list]   # Phis of generated signal neutrinos
		retThetas = [i[0] for i in map_list] #Thetas of generated signal neutrinos

		
		if countMeasuredSources > 0:
			average = sigNeutrinos*1.0/countMeasuredSources
		else:
			average = 0.0
			
		#~ print "Measured Sources: "+str(countMeasuredSources)
		#~ print "Average number of nu per source: " +str(average)+ " in comparison to expectation "+str(mean_muReal)
		#~ print ""
		#~ signalInfo = [int(sigNeutrinos), int(countMeasuredSources), float(average)]
		signalInfo = [int(sigNeutrinos), int(-1), int(countMeasuredSources), int(curMu), float(average)]
		ret  = [np.array(retThetas), np.array(retPhis), np.array(signalInfo)]
		return ret


	#### SAVES LIST OF SIGNAL NEUTRINOS ####
	def saveSignalNeutrinos(self, nPath):
		np.savetxt(nPath+"_signalHits.txt", self.allSignalNeutrinos)
		print "'Signal Hits' data successfully saved to... "+nPath+"_signalHits.txt"
	
	
	#### GENERATE NEW ZENITH SPEC. BY CHANING ATM BY self.curChange ####
	def genNewSpec(self):
		print "CurChange Percentage: "+str(self.curPerc)

		self.histoZenithDistrAtm = normHist(self.histoZenithDistrAtm, 0.95)
		self.histoZenithDistrAtm = changeDistrPerc(self.histoZenithDistrAtm, self.curPerc)
		self.histoZenithDistrAtm = normHist(self.histoZenithDistrAtm, 0.95)

		self.atmosSpline = interpolate.InterpolatedUnivariateSpline(self.edgesZenithDistrAtm, self.histoZenithDistrAtm)
				

	#### GENERATE NEW ZENITH SPEC. BY CHANING ATM BY self.curChange ####
	def reformSpec(self):
		self.histoZenithDistrAtm = self.histoZenithDistrAtm_ORIGINAL[:]


	#### PLOT ZENITH DISTR. ####
	def plotZenithSpec(self,nFigure=-1, changed=False):		
		if changed==True:
			multiPoleAnalysis.plotZenithSpec(self, nFigure, nLabel="atm. varied "+str(round(self.curPerc,3)*100)+"%")
			plt.title("atm. zenith distr., Variation="+str(round(self.curPerc,3)*100)+"%")
		else:
			multiPoleAnalysis.plotZenithSpec(self, nFigure)

	#### SETS CHANGE PERCENTAGE OF SPECTRUM ####
	def setNewPerc(self,nCurPerc):
		self.curPerc = nCurPerc
		return True



###########################################
##########    CLASS FOR    ################
########## MERGING 2 ANAS. ################
###########################################

class compareAnas:
	#### CONSTRUCTOR ####
	def __init__(self, nSignal, nOriginal, nReadDir, nSaveDir, weights=[]):
		self.ORG = copy.copy(nOriginal)	
		self.SIG = copy.copy(nSignal)
		self.pltColor= "k"
		self.usedLogForD2 = False

		self.ls = self.ORG.ls
		self.lsEff = self.ORG.lsEff
		self.lsLog = self.ORG.lsLog

		self.SAVE_DIR = nSaveDir
		self.READ_DIR = nReadDir

		if weights !=[]:
			self.clChangedWeight = weights[0]
			self.effClChangedWeight = weights[1]
			if self.SIG.useAlm == True:
				self.almWeights = weights[2]


	#### DECLARES TYPE OF COMPARISON FOR PLOT LABELLING ####
	def setCompType(self, nType):
		if(nType == "PureAtm"):
			self.compType = "PureAtm"
			self.infoString = " pure bg., histogram"
			self.saveString = ""
		if(nType == "PureAtm_fixZenith"):
			self.compType = "PureAtm"
			self.infoString = " pure bg., fix zenith"
			self.saveString = ""
		if(nType == "ATM_var1"):
			self.compType = "ATM_var1"
			self.infoString = " changed: "+str(round(self.SIG.curPerc,3))+" RUNS: "+str(self.SIG.NUMBER_RAN_RUNS)
			self.saveString = "varAtm_"+str(round(self.SIG.curPerc,3))+"RUNS"+str(self.SIG.NUMBER_RAN_RUNS)
		if(nType == "SIG_NSources"):
			self.compType = "SIG_NSources"
			self.infoString = " $ N_{Sources} $: "+str(self.SIG.curNSources)#+" $ \mu $: "+str(self.SIG.curMuSources) #+" RUNS: "+str(self.SIG.NUMBER_RAN_RUNS)
			self.saveString = "varSig_NSou"+str(self.SIG.curNSources)+"Mu"+str(self.SIG.curMuSources)+"RUNS"+str(self.SIG.NUMBER_RAN_RUNS)
		if(nType == "SIG_MuSources"):
			self.compType = "SIG_MuSources"
			self.infoString = r" $ N_{Sources} $: "+str(self.SIG.curNSources)+r" $ \mu $: "+str(self.SIG.curMuSources)+" RUNS: "+str(self.SIG.NUMBER_RAN_RUNS)
			self.saveString = "varMu_NSou"+str(self.SIG.curNSources)+"Mu"+str(self.SIG.curMuSources)+"RUNS"+str(self.SIG.NUMBER_RAN_RUNS)
		if(nType == "PURE_SIG"):
			self.compType = "PURE_SIG"
			self.infoString = "Pure signal, $N_{\mathrm{Sou}}= $" +str(self.SIG.curNSources)#+", RUNS: "+str(self.SIG.NUMBER_RAN_RUNS) $ \mu="+str(self.SIG.curMuSources)+", 
			self.saveString = "pureSig_Mu"+str(self.SIG.curMuSources)+"RUNS"+str(self.SIG.NUMBER_RAN_RUNS)
		if(nType == "EXPERIMENT"):
			self.compType = "EXPERIMENT"
			self.infoString = "Experimental data" 
			self.saveString = "ExperimentalRUNS"+str(self.SIG.NUMBER_RAN_RUNS)
		if(nType == "none"):
			self.compType = "none"
			self.infoString = "" 
			self.saveString = ""
			 

	#### CALCULATES ALL DIFFERENT WEIGHTS ####
	def calcWeights(self, log, opt="Cl"):
		if log == True:
			print "ML-WARNING: LOGARITHMIC WEIGHTS ARE BEING CALCULATED!"
			if opt=="Cl":
				self.clChangedWeight = calcWeights(self.SIG.logCl_means , self.ORG.logCl_means , self.SIG.logCl_errors, self.ORG.logCl_errors)		
			if opt=="effCl":
				self.effClChangedWeight = calcWeights(self.SIG.logEffCl_means, self.ORG.logEffCl_means, self.SIG.logEffCl_errors, self.ORG.logEffCl_errors)	
		else:
			print "Note: Non logarithmic weights chosen!"
			if opt=="Cl":
				self.clChangedWeight = calcWeights(self.SIG.cl_means, self.ORG.cl_means, self.SIG.cl_errors, self.ORG.cl_errors)	
	
			if opt=="effCl":
				self.effClChangedWeight = calcWeights(self.SIG.effCl_means, self.ORG.effCl_means, self.SIG.effCl_errors, self.ORG.effCl_errors)	

			if opt=="sqrtEffCl":
				self.sqrtEffClChangedWeight = calcWeights(self.SIG.sqrtEffCl_means, self.ORG.sqrtEffCl_means, self.SIG.sqrtEffCl_errors, self.ORG.sqrtEffCl_errors)	
				
			if opt=="ClLog":
				self.ClLogChangedWeight = calcWeights(self.SIG.cl_log_means, self.ORG.cl_log_means, self.SIG.cl_log_errors, self.ORG.cl_log_errors)	
				
			if opt=="almCl":
				self.almWeights = calcWeights(self.SIG.alm_means, self.ORG.alm_means, self.SIG.alm_errors, self.ORG.alm_errors)


	#### CALCULATES ALL DIFFERENT WEIGHTS ####
	def calcAllWeights(self, log):
			self.calcWeights(log=log, opt="Cl" )
			self.calcWeights(log=log, opt="effCl")
			self.calcWeights(opt="almCl")


	#### RETURNS ALL WEIGHTS NICELY PACKED ####
	def packWeights(self, packAlm=False):
		if packAlm == True:
			pack = [self.clChangedWeight, self.effClChangedWeight, self.almWeights]
		else:
			pack = [self.clChangedWeight, self.effClChangedWeight]
		return pack


	#### PLOTS ALL DIFFERENT WEIGHTS ####
	def plotWeights(self, nFigure=-1 ,opt="Cl", info="", label="", marker="v", markersize=10.0, markeredgewidth=0.0, xlim=600):
		plotControl(nFigure)
		plt.subplots_adjust(left=0.2)
		s = plt.subplot(111)
		s.title.set_position([0.5,1.01])
		
		if(self.compType == "ATM_var1"):
			nLabel = "atm. changed: "+info
		elif(self.compType == "SIG_NSources"):
			nLabel = "var. Sig.: "+info
		elif(self.compType == "PURE_SIG"):
			nLabel = "Pure signal, "+info
		else:
			nLabel = "- - - -"
			
		if label != "":
			nLabel = label
		
		if opt=="Cl":
			plt.errorbar(self.ls, self.clChangedWeight, fmt="o", color=str(self.pltColor), label=nLabel, markeredgewidth=markeredgewidth)# "+str(-round(curPerc,3))+".."+str(round(curPerc,3)))
			plt.title(str(nLabel)+", lmax="+str(self.SIG.l_max)+", ev.: "+str(self.ORG.NUMBER_SIMULATED_EVENTS)+", Runs: "+str(self.ORG.NUMBER_RAN_RUNS))
			plt.xlabel("$ \ell $")
			plt.ylabel(r"$  w_{\ell} = (\langle C_{l,\mathrm{sig}}\rangle-\langle C_{l,\mathrm{bg}}\rangle) / \sigma_{l,\mathrm{bg}} $") #\sqrt{\sigma_{l}^{2}+\sigma_{l,atm}^{2}} 
			plt.yscale("linear")
			plt.xlim([0,xlim])

		if opt=="effCl":
			plt.errorbar(self.lsEff, self.effClChangedWeight, fmt=marker, color=str(self.pltColor), label=nLabel, markeredgewidth=markeredgewidth, markersize=markersize, markeredgecolor="k") 
			plt.title("TS weights, $\ell_\mathrm{max}$="+str(self.SIG.l_max))
			plt.xlabel("$ \ell $")
			plt.ylabel(r"$  w_{\ell} = (\langle C_{\ell, \mathrm{sig}}^{\mathrm{eff}}\rangle -\langle C_{\ell,\mathrm{bg}}^{\mathrm{eff}}\rangle) / \sigma_{C_{\ell,\mathrm{bg}}^{\mathrm{eff}}} $") #\sqrt{\sigma_{l,eff}^{2}+\sigma_{l,eff,atm}^{2}}
			plt.yscale("linear")
			plt.xlim([0,xlim])
			
		if opt=="sqrtEffCl":
			plt.errorbar(self.lsEff, self.sqrtEffClChangedWeight, fmt="o", color=str(self.pltColor), label=nLabel, markeredgewidth=0.0) 
			plt.title(nLabel+", lmax="+str(self.SIG.l_max)+", ev.: "+str(self.SIG.NUMBER_SIMULATED_EVENTS)+", Runs: "+str(self.SIG.NUMBER_RAN_RUNS))
			plt.xlabel("$ \ell $")
			plt.ylabel(r"$  (\langle \sqrt{C_{l}^{\mathrm{eff}}}\rangle -\langle \sqrt{C_{l,atm}^{\mathrm{eff}}}\rangle) / \sigma_{l,eff,sqrt,atm} $") #\sqrt{\sigma_{l,eff}^{2}+\sigma_{l,eff,atm}^{2}}
			plt.yscale("linear")
			
		if opt=="ClLog":
			plt.errorbar(self.lsLog , self.clLogChangedWeight, fmt="o", color=str(self.pltColor), label=nLabel)# "+str(-round(curPerc,3))+".."+str(round(curPerc,3)))
			plt.title(str(nLabel)+", lmax="+str(self.SIG.l_max)+", ev.: "+str(self.ORG.NUMBER_SIMULATED_EVENTS)+", Runs: "+str(self.ORG.NUMBER_RAN_RUNS))
			plt.xlabel("$ \ell $")
			plt.ylabel(r"$  (\langle C_{l}^{Log}\rangle-\langle C_{l,atm}^{Log}\rangle) / \sigma_{l,atm}^{Log} $") #\sqrt{\sigma_{l}^{2}+\sigma_{l,atm}^{2}} 
			plt.yscale("linear")
			
		if opt=="almCl":
			self.plot_alm_weights = getAlmPlane(self.almWeights, False)
			plt.imshow(self.plot_alm_weights , interpolation='nearest')
			plt.xlabel("m")
			plt.ylabel("$ \ell $")
			cb = plt.colorbar()
			cb.set_label(r"$ ( \langle \| a_{l}^{m}\| \rangle - \langle \| a_{l,atm}^{m}\| \rangle ) /(\sqrt{\sigma_{l,m,eff}^{2}+\sigma_{l,m,eff,atm}^{2}}) $"+infoString)
			plt.title(nLabel+", "+self.infoString)
 

	#### CALC. D2s ####
	def calcD2(self, opt="Cl", norm=True, log=False, calcLimits=[], cummulative=False, lokalCummulative=False):
		#print "D2 are beeing calculated, log-mode is: "+str(log)
		if log == False:
			self.usedLogForD2=False
			if opt=="Cl":
				self.DsquaredDistr = []
				for i in range(0, len(self.SIG.cl_all)):
					curDSquared = calcD2(self.SIG.cl_all[i], self.ORG.cl_means, self.ORG.cl_errors, self.clChangedWeight, self.SIG.l_max, norm, calcLimits)
					self.DsquaredDistr.append(curDSquared)

			if opt=="effCl":
				self.effDsquaredDistr = []
				for i in range(0, len(self.SIG.effCl_all)):
					curDSquared = calcEffD2(self.SIG.effCl_all[i], self.ORG.effCl_means, self.ORG.effCl_errors, self.effClChangedWeight, self.SIG.l_max, norm, calcLimits)
					self.effDsquaredDistr.append(curDSquared)
				
				if cummulative == True:
					self.cummulativeEffD2		= []
					self.cummulativeEffD2_err 	= []
					curLmaxCummulativeAll		= [0.0 for i in range(0, len(self.SIG.effCl_all))]
								
					for thisLmax in range(0, self.SIG.l_max):
						print "This lmax: "+str(thisLmax)
						for i in range(0, len(self.SIG.effCl_all)):
											
							#curLmaxCummulativeAll[i] += (1.0/(self.SIG.l_max)*
							#		(np.power((self.SIG.effCl_all[i][thisLmax]-self.ORG.effCl_means[thisLmax]),2)*1.0/np.power(self.ORG.effCl_errors[thisLmax],2))*self.effClChangedWeight[thisLmax]*returnSign(self.SIG.effCl_all[i][thisLmax]-self.ORG.effCl_means[thisLmax]))*1.0/sumFromTo(self.effClChangedWeight, 0, self.SIG.l_max)
							
							curLmaxCummulativeAll[i] += (1.0/(self.SIG.l_max)*
									(np.power((self.SIG.effCl_all[i][thisLmax]-self.ORG.effCl_means[thisLmax]),2)*1.0/np.power(self.ORG.effCl_errors[thisLmax],2))*self.effClChangedWeight[thisLmax]*returnSign(self.SIG.effCl_all[i][thisLmax]-self.ORG.effCl_means[thisLmax]))*1.0/sumFromTo(self.effClChangedWeight, 0, self.SIG.l_max)
									
						self.cummulativeEffD2.append(np.mean(curLmaxCummulativeAll))
						self.cummulativeEffD2_err.append(np.std(curLmaxCummulativeAll))
						
						#for i in range(0,lmax):
						#	curEffDSquared += 1.0/lmax*np.power(cl_new[i]-cl_original[i],2)*1.0/(np.power(cl_original_errors[i],2))*weights[i]*returnSign(cl_new[i]-cl_original[i])
						
				elif lokalCummulative == True:
					self.cummulativeEffD2		= []
					self.cummulativeEffD2_err 	= []
					self.singleL_D2				= []
					
					print "Calculate single l deviations ..."
					for i in range(0, len(self.SIG.effCl_all)):
						self.singleL_D2.append([])
						for l in range(0, self.SIG.l_max):
							self.singleL_D2[i].append(float(  np.power(self.SIG.effCl_all[i][l] - self.ORG.effCl_means[l],2)*1.0/np.power(self.ORG.effCl_errors[l],2)*self.effClChangedWeight[l]*returnSign(self.SIG.effCl_all[i][l]-self.ORG.effCl_means[l])*1.0/(self.SIG.l_max)))
							
					print "Calculate accumulated deviations ..."
					for curLCenter in range(0,self.SIG.l_max):
						minL = curLCenter-10
						maxL = curLCenter+10
						if minL < 1: minL = 0
						if maxL > self.SIG.l_max: maxL = self.SIG.l_max
						
						summedL_D2	= []

						for i in range(0,len(self.SIG.effCl_all)):
							summedL_D2.append(sumFromTo(self.singleL_D2[i], minL, maxL)*1.0/sumFromTo(self.effClChangedWeight, minL, maxL))
							
						self.cummulativeEffD2.append(np.mean(summedL_D2))
						self.cummulativeEffD2_err.append(np.std(summedL_D2))
					print "... done."
						
			if opt=="sqrtEffCl":
				self.sqrtEffDsquaredDistr = []
				for i in range(0, len(self.SIG.sqrtEffCl_all)):
					curDSquared = calcEffD2(self.SIG.sqrtEffCl_all[i], self.ORG.sqrtEffCl_means, self.ORG.sqrtEffCl_errors, self.sqrtEffClChangedWeight, self.SIG.l_max, norm, calcLimits)
					self.sqrtEffDsquaredDistr.append(curDSquared)
							
			if opt=="ClLog":
				self.DsquaredLogDistr = []
				for i in range(0, len(self.SIG.cl_log_all)):
					curDSquaredLog = calcEffD2(self.SIG.cl_log_all[i], self.ORG.cl_log_means, self.ORG.cl_log_errors, self.clLogChangedWeight, self.SIG.l_max, norm, calcLimits)
					self.DsquaredLogDistr.append(curDSquaredLog)
					
			if opt=="almCl":
				self.almDsquaredDistr =[]
				for i in range(0, self.SIG.NUMBER_RAN_RUNS_CHANGED):
					almD2 = calcD2Alm(self.ORG.alm_means, self.SIG.alm_all_abs[i], self.ORG.alm_errors, self.almWeights, lmax=self.l_max)
					self.almDsquaredDistr.append(almD2)

			if opt=="effAlmCl":
				self.almEffDsquaredDistr =[]
				for i in range(0, self.SIG.NUMBER_RAN_RUNS_CHANGED):
					almEffD2 = calcEffD2Alm(self.ORG.alm_means, self.SIG.alm_all_abs[i], self.ORG.alm_errors, self.almWeights, lmax=self.l_max)
					self.almEffDsquaredDistr.append(almEffD2)
		elif log == True:
			self.usedLogForD2=True
			if opt=="Cl":
				self.DsquaredDistr = []
				for i in range(0, len(self.SIG.cl_all)):
					curDSquared = calcD2(np.log(self.SIG.cl_all[i]), self.ORG.logCl_means, self.ORG.logCl_errors, self.clChangedWeight, self.SIG.l_max, norm, calcLimits)
					self.DsquaredDistr.append(curDSquared)
			if opt=="effCl":
				self.effDsquaredDistr = []
				for i in range(0, len(self.SIG.effCl_all)):
					curDSquared = calcEffD2(np.log(self.SIG.effCl_all[i]), self.ORG.logEffCl_means, self.ORG.logEffCl_errors, self.effClChangedWeight, self.SIG.l_max, norm, calcLimits)
					self.effDsquaredDistr.append(curDSquared)


	#### CREATE HISTOS ####
	def createD2Histos(self, minD2, maxD2, opt="Cl", resolution=0, fitfunc="gauss", shift=0.01, fitting=False):
		plotFits = False
		if resolution==0: resolution = 500
		if opt=="Cl":
			self.histoDSquaredDistr, self.edgesDSquaredDistr = np.histogram(self.DsquaredDistr, bins=resolution, range=(minD2, maxD2))
			if len(self.DsquaredDistr) > 1:
					N, mu, sigma, errN, errMu, errSigma = fitGaussian(self.DsquaredDistr, self.edgesDSquaredDistr, self.histoDSquaredDistr, plot=plotFits)
			else:
				N 		= 0.0
				mu 		= self.DsquaredDistr[0]
				sigma	= 0.0
				errN 	= 0.0
				errMu 	= 0.0
				errSigma= 0.0
			self.D2GausFit = [N, mu, sigma, errN, errMu, errSigma]
			
			
		elif opt=="effCl":
			self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=resolution, range=(minD2, maxD2))
			self.histoEffDSquaredDistr = np.array(self.histoEffDSquaredDistr)*1.0/sum(self.histoEffDSquaredDistr)
			DSquared_temp = []
			for i in range(0, len(self.edgesEffDSquaredDistr)):
				self.edgesEffDSquaredDistr[i]+= shift
			for i in range(0, len(self.effDsquaredDistr)):
				DSquared_temp.append(self.effDsquaredDistr[i]+ shift)
			if len(DSquared_temp) > 1:
				if fitfunc=="gamma":
					if fitting:
						print "Fitting enabled!"
						k, theta, s, nrm, errk, errtheta, errs, errnrm = fitGammaDistr(DSquared_temp, self.edgesEffDSquaredDistr, self.histoEffDSquaredDistr, plot=plotFits, shift=shift)
						N 		= 0.0
						mu 		= DSquared_temp[0]
						sigma	= 0.0
						errN 	= 0.0
						errMu 	= 0.0
						errSigma= 0.0
					else:
						ndof     = len(DSquared_temp)
						N        = 0.0 #### NEW DUE TO ASYMETRIC D2 DISTRIBUTION
						errN     = 0.0
						mu       = np.median(DSquared_temp)
						errQ     = 0.5/np.sqrt(len(DSquared_temp))
						errMu    = (np.percentile(DSquared_temp, 0.5+errQ)-np.percentile(DSquared_temp, 0.5-errQ))/2. ### "symmetrified" error
						sigma    = np.std(DSquared_temp, ddof=1)
						errSigma = np.sqrt(1./ndof*(moment(DSquared_temp,4)-(ndof-3.)*1./(ndof-1.)*sigma**4))/(2.*sigma)
						k			= 0.0
						theta = 0.0
						s			= 0.0
						nrm		= 0.0
						errk		= 0.0
						errtheta= 0.0
						errs 		= 0.0
						errnrm 	= 0.0
				

				else:
					if fitting:
						print "Fitting enabled!"
						N, mu, sigma, errN, errMu, errSigma = fitGaussian(DSquared_temp, self.edgesEffDSquaredDistr, self.histoEffDSquaredDistr, plot=plotFits)
					else:
						ndof     = len(DSquared_temp)
						N        = 0.0 #### NEW DUE TO ASYMETRIC D2 DISTRIBUTION
						errN     = 0.0
						mu       = np.median(DSquared_temp)
						errQ     = 0.5/np.sqrt(len(DSquared_temp))
						errMu    = (np.percentile(DSquared_temp, 0.5+errQ)-np.percentile(DSquared_temp, 0.5-errQ))/2. ### "symmetrified" error
						sigma    = np.std(DSquared_temp, ddof=1)
						errSigma = np.sqrt(1./ndof*(moment(DSquared_temp,4)-(ndof-3.)*1./(ndof-1.)*sigma**4))/(2.*sigma)
						
					k			= 0.0
					theta = 0.0
					s			= 0.0
					nrm		= 0.0
					errk		= 0.0
					errtheta= 0.0
					errs 		= 0.0
					errnrm 	= 0.0
					
				self.effD2GausFit  = [N, mu, sigma, errN, errMu, errSigma]
				self.effD2GammaFit = [k, theta, s, nrm, errk, errtheta, errs, errnrm]
						
			else: 
				N 		= 0.0
				mu 		= DSquared_temp[0]
				sigma	= 0.0
				k			= 0.0
				theta = 0.0
				s			= 0.0
				nrm		= 0.0
				errN 	= 0.0
				errMu 	= 0.0
				errSigma= 0.0
				errk		= 0.0
				errtheta= 0.0
				errs 		= 0.0
				errnrm 	= 0.0
				self.effD2GammaFit = [k, theta, s, nrm, errk, errtheta, errs, errnrm]
				self.effD2GausFit = [N, mu, sigma, errN, errMu, errSigma]
			
		elif opt=="sqrtEffCl":
			self.histoSqrtEffDSquaredDistr, self.edgesSqrtEffDSquaredDistr = np.histogram(self.sqrtEffDsquaredDistr, bins=resolution, range=(minD2, maxD2))
			if len(self.sqrtEffDsquaredDistr) > 1:
				N, mu, sigma, errN, errMu, errSigma = fitGaussian(self.sqrtEffDsquaredDistr, self.edgesSqrtEffDSquaredDistr, self.histoSqrtEffDSquaredDistr, plot=plotFits)
			else:
				N 		= 0.0
				mu 		= DSquared_temp[0]
				sigma	= 0.0
				errN 	= 0.0
				errMu 	= 0.0
				errSigma= 0.0
			self.sqrtEffD2GausFit = [N, mu, sigma, errN, errMu, errSigma]
		elif opt=="ClLog":
			self.histoDSquaredLogDistr, self.edgesDSquaredLogDistr = np.histogram(self.DsquaredLogDistr, bins=resolution, range=(minD2, maxD2))
			N, mu, sigma, errN, errMu, errSigma = fitGaussian(self.DsquaredLogDistr, self.edgesDSquaredLogDistr, self.histoDSquaredLogDistr, plot=plotFits)
			self.D2LogGausFit = [N, mu, sigma, errN, errMu, errSigma]
		elif opt=="effAlmCl":
			self.histoAlmDSquaredDistr, self.edgesAlmDSquaredDistr = np.histogram(self.almDsquaredDistr, bins=resolution, range=(minD2, maxD2))
			N, mu, sigma, errN, errMu, errSigma = fitGaussian(self.almDsquaredDistr, self.edgesAlmDSquaredDistr, self.histoAlmDSquaredDistr, plot=plotFits)
			self.almD2GausFit = [N, mu, sigma, errN, errMu, errSigma]
		elif opt=="effAlmCl":
			self.histoEffAlmDSquaredDistr, self.edgesEffAlmDSquaredDistr = np.histogram(self.almEffDsquaredDistr, bins=resolution, range=(minD2, maxD2))
			N, mu, sigma, errN, errMu, errSigma = fitGaussian(self.almEffDsquaredDistr, self.edgesEffAlmDSquaredDistr, self.histoEffAlmDSquaredDistr, plot=plotFits)
			self.almEffD2GausFit = [N, mu, sigma, errN, errMu, errSigma]
		if fitfunc=="gamma" and fitting ==True:
			return [k, theta, s, nrm]
		else:
			return [N, mu, sigma]

	def getCauchyFit(self, minD2, maxD2, resolution=100, plotting=False):
		self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=resolution, range=(minD2, maxD2), normed=True)
		n = sum(self.histoEffDSquaredDistr)
		self.histoEffDSquaredDistr = np.array(self.histoEffDSquaredDistr)*1./n
		#~ self.effDsquaredDistr = np.array(self.effDsquaredDistr)*1./n
		coeff, err = fitCauchy(self.effDsquaredDistr, self.edgesEffDSquaredDistr, self.histoEffDSquaredDistr, plot=plotting)
		self.effD2Cauchy=[coeff, err]
		return coeff
		
	def getStudFit(self, minD2, maxD2, resolution=100, plotting=False, nFigure=-1):
		self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=resolution, range=(minD2, maxD2), normed=True)
		n = sum(self.histoEffDSquaredDistr)
		self.histoEffDSquaredDistr = np.array(self.histoEffDSquaredDistr)*1./n
		result = t.fit(self.effDsquaredDistr)
		self.effD2Stud = result
		if plotting:
			plt.figure(nFigure)
			if len(self.edgesEffDSquaredDistr) == len(self.histoEffDSquaredDistr) +1:
				ed = setNewEdges(self.edgesEffDSquaredDistr)
			else:
				ed = self.edgesEffDSquaredDistrS
			bin_w = ed[1]-ed[0]
			plt.plot(ed, t.pdf(ed, *result)*bin_w, color="teal", label="Student t Fit")
		return result
		
		
	def getExpFit(self, minD2, maxD2, resolution=100, plotting=False, percentile=0.7, nFigure=-1):
		self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=resolution, range=(minD2, maxD2)) #, normed=True)
		n = sum(self.histoEffDSquaredDistr)
		self.histoEffDSquaredDistr = np.array(self.histoEffDSquaredDistr)*1./n
		
		#~ if len(self.edgesEffDSquaredDistr)==len(self.histoEffDSquaredDistr)+1:
			#~ self.edgesEffDSquaredDistr=setNewEdges(self.edgesEffDSquaredDistr)
			
		p70=np.percentile(self.effDsquaredDistr, percentile*100.)
		p50=np.median(self.effDsquaredDistr)
		bin_w=self.edgesEffDSquaredDistr[1]-self.edgesEffDSquaredDistr[0]
		err_histoEff = np.sqrt(self.histoEffDSquaredDistr/len(self.histoEffDSquaredDistr))
		
		
		effD2_p70= []
		for i in self.effDsquaredDistr:
				if i>p70:
						effD2_p70.append(i)			
		histo_p70, edges_p70 = np.histogram(effD2_p70, bins=resolution, range=(minD2, maxD2)) #, normed=True)
		if len(edges_p70)==len(histo_p70)+1:
				edges_p70=setNewEdges(edges_p70)

		histo_p70 = np.array(histo_p70)*1./n    
		err_histo_p70 = np.sqrt(histo_p70/len(self.effDsquaredDistr))
		#~print sum(histo_p70)
		#~print sum(histo)

		#find first bin >0 and add 1#
		starting_index = np.where(histo_p70>0)[0][0]+1		 

		#calculate cummulative histogram #
		histo_cummulative = np.zeros(len(self.histoEffDSquaredDistr)+1)
		for f in range(0, len(self.histoEffDSquaredDistr)):
				histo_cummulative[f+1]=histo_cummulative[f]+self.histoEffDSquaredDistr[f]
		self.effD2_cummulative = histo_cummulative[1:]
		#~print histo_cummulative[-1]
		#~print sum(histo)
		#cut histo to right size x>p70#
		edges_cut=np.array(edges_p70[starting_index:])
		histo_cut=np.array(histo_p70[starting_index:])
		err_histo_cut=np.array(err_histo_p70[starting_index:])
		
		#make an educated guess for fit parameters#
		a_guess=self.histoEffDSquaredDistr[starting_index-1]/(1.-percentile)/bin_w
		#~ print a_guess
		result = fitExp(edges_cut, histo_cut, err_histo_cut, a=a_guess, perc=percentile, x0=p70, plot=plotting, N=len(self.effDsquaredDistr))
		#~ print "Fit Result and Parameters: "+str(result)
		self.effD2Exp = result
		if plotting:
			plt.figure(nFigure)
			#~ plt.errorbar(edges_cut, histo_cut, yerr=err_histo_cut, drawstyle="steps-mid", label="D2>p70") #yerr=err_histo_cut, 
			#~ plt.vlines(p50, 0.0001, max(self.histoEffDSquaredDistr), color="#339966", linestyle="--", label="median="+str(round(p50, 5)))
			plt.vlines(p70, 0.0001, max(self.histoEffDSquaredDistr), color="#663399", linestyle="--", label=str(percentile)+"quantile="+str(round(p70, 5)))
		return result
    
    
	def getGausExpConvFit(self, minD2, maxD2, tolerance=0.2, resolution=50, maxNruns=1000):
			
		while_count=0
		diff=[1.,1.]
		fail_count=0
		condition_fail=0
		fit_params=[]
		gof=[]
		rel_err_array=[]
		fit_points=[]
		res_array=[]	
		maxD2_array=[]
		
		while while_count<=maxNruns:
			infodict=[]
			pcov=[]
			rannumber2=np.random.uniform(0.,1.)
			if rannumber2<0.5 or len(res_array)==0:
				if while_count<maxNruns/2:
					cresolution=np.random.randint(resolution,resolution+resolution/2.)
				else:
					cresolution=np.random.randint(resolution-resolution/2.,resolution)
			else:
				val=np.random.choice(res_array)
				maxD2val=val+np.random.uniform(-0.1,0.1)*val

			shiftval=0.
			#shiftval=
			rannumber=np.random.uniform(0.,1.)
			if rannumber<0.5 or len(maxD2_array)==0:
				maxD2val=maxD2+np.std(self.effDsquaredDistr)*np.random.uniform(-1,1)
			else:
				val=np.random.choice(maxD2_array)
				maxD2val=val+np.random.uniform(-0.1,0.1)*val
			ya, xa = np.histogram(self.effDsquaredDistr, bins=cresolution, range=(minD2, maxD2val))
			xa=np.array(setNewEdges(xa))
			ya=np.array(ya)
			ya_err = np.sqrt(ya)
			medEff=np.median(self.effDsquaredDistr)
			maxvalind=np.where(ya==np.max(ya))[0][0]
			#maxx=xa[maxvalind]
			maxx=medEff
			sigma=np.std(self.effDsquaredDistr)
			xcut=maxx+(np.random.uniform(1.,3.))*sigma
			tau=0.5*(xcut-maxx)/sigma**2	
					
			####################Gaus + Exp Convolution Fit############################	
			try:	
				p0=[np.max(ya),sigma,1./tau,maxx]	
				vfunc=np.vectorize(gaussexpconvolv)
				popt, pcov, infodict, errmsg, ier=curve_fit(lambda x,a,b,c,d: vfunc(x,a,b,c,d),xa, ya, p0=p0,sigma=ya_err, full_output = True)
				chi2_conv= (infodict['fvec']**2).sum()/ (len(ya)-len(p0))
				if pcov[0][0]!=np.inf and chi2_conv<2.5 and chi2_conv>0.5: 
					fit_params.append(popt)
					gof.append(chi2_conv)
					rel_err_array.append([k/popt[i] for i,k in enumerate(np.sqrt(np.diag(pcov)))])
					fit_points.append(cresolution)
					res_array.append(cresolution)
					maxD2_array.append(maxD2val)
				else:
					condition_fail+=1
			except:
				fail_count=fail_count+1	
			  
			#########Break Conditions################################
			 
			if len(fit_params)>50 or while_count==maxNruns: 
				print "Succesfull Fits for Gauss+Exp Convolution :{}".format(len(fit_params))
				fit_params=np.array(fit_params)
				gof=np.array(gof)
				rel_err_array=np.array(rel_err_array)	
								  
				if len(fit_params)>0:
					ind0_bf=np.where(np.absolute(np.array(gof)-1)==np.min(np.absolute(np.array(gof)-1)))[0][0]
					self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=res_array[ind0_bf], range=(minD2, maxD2_array[ind0_bf]))
					print "Best Chi2 Gauss+Exp Convolution:  {}".format(gof[ind0_bf])
					print "Amount of Fit Points: {}".format(fit_points[ind0_bf])
					return [fit_params[ind0_bf],gof[ind0_bf],rel_err_array[ind0_bf] ]

				if len(fit_params)==0:
					print "FIT has FAILED for {} runs".format(fail_count)
					print "Condition was not fullfilled for {} runs".format(condition_fail)
					return [[-1],[-1], [-1]]
			
				break
									
			else:
				while_count=while_count+1
		
	def getGausExpFit(self, minD2, maxD2, tolerance=0.2, resolution=50, maxNruns=1000):
			
		while_count=0
		diff=[1.,1.]
		fail_count=0
		condition_fail=0
		fit_params=[]
		gof=[]
		rel_err_array=[]
		fit_points=[]
		res_array=[]	
		
		while while_count<=maxNruns:
			infodict=[]
			pcov=[]
			if while_count<maxNruns/2:
				cresolution=np.random.randint(resolution,resolution+resolution/2.)
			else:
				cresolution=np.random.randint(resolution-resolution/2.,resolution)
			#shiftval=np.std(self.effDsquaredDistr)*np.random.uniform(0,2)
			shiftval=0.0
			ya, xa = np.histogram(self.effDsquaredDistr, bins=cresolution, range=(minD2, maxD2-shiftval))
			xa=np.array(setNewEdges(xa))
			ya=np.array(ya)
			ya_err = np.sqrt(ya)
			medEff=np.median(self.effDsquaredDistr)
			maxvalind=np.where(ya==np.max(ya))[0][0]
			#maxx=xa[maxvalind]
			maxx=medEff
			sigma=np.std(self.effDsquaredDistr)
			xcut=maxx+(np.random.uniform(1.,3.))*sigma
			tau=0.5*(xcut-maxx)/sigma**2			
			####################Gaus + Exp Fit############################		
			p0=[sigma,maxx,np.max(ya),tau,xcut]
			try:
				popt, pcov, infodict, errmsg, ier=curve_fit(lambda x,a,b,c,d,e: gaussexpfunc(x,a,b,c,d,e), xa, ya, p0=p0, sigma=ya_err, full_output = True)
				chi2= (infodict['fvec']**2).sum()/ (len(ya)-len(p0))
				xcut_rel_differential=2*(popt[4]-popt[1])/(popt[0])**2/popt[3]
				if pcov[0][0]!=np.inf and chi2<2.5 and chi2>0.5 and xcut_rel_differential>0.99 and xcut_rel_differential<1.01: 
					rel_err_array.append([k/popt[i] for i,k in enumerate(np.sqrt(np.diag(pcov)))])
					fit_params.append(popt)
					gof.append(chi2)
					fit_points.append(cresolution)
					res_array.append(cresolution)
				else:
					condition_fail+=1
			except:
				fail_count=fail_count+1
			  
			#########Break Conditions################################
			 
			if len(fit_params)>50 or while_count==maxNruns: 
				print "Succesfull Fits for Gaus+Exp :{}".format(len(fit_params))
				fit_params=np.array(fit_params)
				gof=np.array(gof)
				rel_err_array=np.array(rel_err_array)	
								  
				if len(fit_params)>0:
					ind0_bf=np.where(np.absolute(np.array(gof)-1)==np.min(np.absolute(np.array(gof)-1)))[0][0]
					self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=res_array[ind0_bf], range=(minD2, maxD2))
					print "Best Chi2 Exp+Gaus:  {}".format(gof[ind0_bf])
					print "Amount of Fit Points: {}".format(fit_points[ind0_bf])
					return [fit_params[ind0_bf],gof[ind0_bf]]

						
				if len(fit_params)==0:
					print "FIT has FAILED for {} runs".format(fail_count)
					print "Condition was not fullfilled for {} runs".format(condition_fail)
					#print "Min Chi2 :{}".format(np.min(gof))
					return [[-1],[-1]]
			
				break
									
			else:
				while_count=while_count+1		
    
    
	def getGausExpsimultaneousFit(self, minD2, maxD2, tolerance=0.2, resolution=50, maxNruns=1000):
		
		########## Fit Function for (not-normalized) TS Distribution. 2 Possible Fit Functions are used #################################################
		#				Gaus Function + Exponential Tail (result function is constructed in a way that it is continous and (nearly) differentiable)		#
		#				Convolution of a Gauss Function and and Exponential Function (smooth function, but integral can just be performed numerically)	#
		##################################################################################################################################################
		
		while_count=0
		diff=[1.,1.]
		fail_count=0
		condition_fail=0
		fit_params=[[],[]]
		gof=[[],[],[]]
		rel_err_array=[[],[]]
		fit_points=[[],[]]
		res_array=[[],[]]	
		
		while while_count<=maxNruns:
			infodict=[]
			infodict2=[]
			pcov=[]
			pcov2=[]
			if while_count<maxNruns/2:
				cresolution=np.random.randint(resolution,resolution+resolution/2.)
			else:
				cresolution=np.random.randint(resolution-resolution/2.,resolution)
			ya, xa = np.histogram(self.effDsquaredDistr, bins=cresolution, range=(minD2, maxD2))
			#n = sum(self.histoEffDSquaredDistr)
			#self.histoEffDSquaredDistr = np.array(self.histoEffDSquaredDistr)*1./n
			xa=np.array(setNewEdges(xa))
			ya=np.array(ya)
			ya_err = np.sqrt(ya)
			medEff=np.median(self.effDsquaredDistr)
			maxvalind=np.where(ya==np.max(ya))[0][0]
			#maxx=xa[maxvalind]
			maxx=medEff
			sigma=np.std(self.effDsquaredDistr)
			xcut=maxx+(np.random.uniform(1.,3.))*sigma
			tau=0.5*(xcut-maxx)/sigma**2

			if maxvalind==0:
					ya=np.append([0.0],ya)
					xa=np.append([np.min(self.effDsquaredDistr)],xa)
			
			####################Gaus + Exp Fit############################		
			p0=[sigma,maxx,np.max(ya),tau,xcut]
			popt, pcov, infodict, errmsg, ier=curve_fit(lambda x,a,b,c,d,e: gaussexpfunc(x,a,b,c,d,e), xa, ya, p0=p0, sigma=ya_err, full_output = True)
			try:
				chi2= (infodict['fvec']**2).sum()/ (len(ya)-len(p0))
				xcut_rel_differential=2*(popt[4]-popt[1])/(popt[0])**2/popt[3]
				#dist=init_sens_function(popt)
				#norm=dist(np.inf)
				#vfunc_med= np.vectorize(lambda x:0.5-dist(x)/norm)
				#median, = fsolve(vfunc_med, medEff)
				if pcov[0][0]!=np.inf and chi2<2.5 and chi2>0.5 and xcut_rel_differential>0.99 and xcut_rel_differential<1.01: 
					rel_err_array[0].append([k/popt[i] for i,k in enumerate(np.sqrt(np.diag(pcov)))])
					fit_params[0].append(popt)
					gof[0].append(chi2)
					gof[1].append(xcut_rel_differential)
					fit_points[0].append(cresolution)
					res_array[0].append(cresolution)
				else:
					condition_fail+=1
			except:
				fail_count=fail_count+1
				
			
				
			#############Convolution Fit ######################################
				
			p02=[np.max(ya),sigma,1./tau,maxx]	
			vfunc=np.vectorize(gaussexpconvolv)
			popt2, pcov2, infodict2, errmsg2, ier2=curve_fit(lambda x,a,b,c,d: vfunc(x,a,b,c,d),xa, ya,p0=p02,sigma=ya_err, full_output = True)
			try:
				chi2_conv= (infodict2['fvec']**2).sum()/ (len(ya)-len(p02))
				#dist=init_sens_function(popt2)
				#norm=dist(np.inf)
				#vfunc_med= np.vectorize(lambda x:0.5-dist(x)/norm)
				#median, = fsolve(vfunc_med, medEff)
				if pcov2[0][0]!=np.inf and chi2_conv<2.5 and chi2_conv>0.5: 
					fit_params[1].append(popt2)
					gof[2].append(chi2_conv)
					rel_err_array[1].append([k/popt2[i] for i,k in enumerate(np.sqrt(np.diag(pcov2)))])
					fit_points[1].append(cresolution)
					res_array[1].append(cresolution)
				else:
					condition_fail+=1
			except:
				fail_count=fail_count+1			
			  
			#########Break Conditions################################
			 
			if (len(fit_params[0])+len(fit_params[1]))>100 or while_count==maxNruns: 
				print "Succesfull Fits for Gamma+Exp :{}".format(len(fit_params[0]))
				print "Succesfull Fits for Convolution of Gamma and Exp :{}".format(len(fit_params[1]))
				fit_params=np.array(fit_params)
				gof=np.array(gof)
				rel_err_array=np.array(rel_err_array)	
								  
				if len(fit_params[0])>0:
					ind0_bf=np.where(np.absolute(np.array(gof[0])-1)==np.min(np.absolute(np.array(gof[0])-1)))[0][0]
					print "Best Chi2 Exp+Gaus:  {}".format(gof[0][ind0_bf])
					if len(fit_params[1])==0:
						self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=res_array[0][ind0_bf], range=(minD2, maxD2))
						print "Amount of Fit Points: {}".format(fit_points[0][ind0_bf])
						print "Quotient of Differential: {}".format(gof[1][ind0_bf])
						print "Relative Error: {}".format(rel_err_array[0][ind0_bf])
						return fit_params[0][ind0_bf]
						
				if len(fit_params[1])>0:
					ind1_bf=np.where(np.absolute(np.array(gof[2])-1)==np.min(np.absolute(np.array(gof[2])-1)))[0][0]
					print "Best Chi2 Convolution: {}".format(gof[2][ind1_bf])
					if len(fit_params[0])==0:
						self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=res_array[1][ind1_bf], range=(minD2, maxD2))
						print "Amount of Fit Points: {}".format(fit_points[1][ind1_bf])
						print "Relative Error: {}".format(rel_err_array[1][ind1_bf])
						return fit_params[1][ind1_bf]
						
				if len(fit_params[0])>0 and len(fit_params[0])>0:
					if abs(1-gof[0][ind0_bf])/abs(1-gof[2][ind1_bf])>120.: ### schliesst conv im prinzip aus
						self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=res_array[1][ind1_bf], range=(minD2, maxD2))						
						print "Amount of Fit Points: {}".format(fit_points[1][ind1_bf])
						print "Relative Error: {}".format(rel_err_array[1][ind1_bf])
						print "Convolution, Fit Params : {}".format(len(fit_params[1][ind1_bf]))
						return fit_params[1][ind1_bf]
					else:
						self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=res_array[0][ind0_bf], range=(minD2, maxD2))
						print "Quotient of Differential: {}".format(gof[1][ind0_bf])
						print "Amount of Fit Points: {}".format(fit_points[0][ind0_bf])
						print "Relative Error: {}".format(rel_err_array[0][ind0_bf])
						print "Gaus+Exp, Fit Params : {}".format(len(fit_params[0][ind0_bf]))
						return fit_params[0][ind0_bf]
						
				if len(fit_params[0])==0 and len(fit_params[0])==0:
					print "FIT has FAILED for {} runs".format(fail_count)
					print "Condition was not fullfilled for {} runs".format(condition_fail)
					return [-1]
			
				break
									
			else:
				while_count=while_count+1
		

	#### PLOTS D2 DISTR. ####
	def plotD2Distr(self, minD2, maxD2, nFigure=-1, opt="Cl", maxNruns=1000, tolerance=0.5 , linestyle="-", dirLabel="", resolution=0, log=False, norm=True, fitfunc="exponential", shift=0.0, fitting=False, plotting=False, nPercentile=0.9, nPerc2=0.1, real_data=False, isBG=False):
		if resolution==0: resolution = 500
		plt.figure(nFigure)
		#s = plotControl([nFigure,1,1,0], legend=1, title=dirLabel)
		if log: s.set_yscale("log", nonposy='clip')    
		    
		if(self.compType == "ATM_var1"):
			nLabel = "atm."
		elif(self.compType == "SIG_NSources"):
			nLabel = "var $ N_{Sources} $"
		elif(self.compType == "PURE_SIG"):
			nLabel = "pure-signal spec."
		elif(self.compType == "SIG_MuSources"):
			nLabel = "various MuSources"
		elif(self.compType == "PureAtm"):
			nLabel = "atmos."
		else:
			nLabel = ""
			
		nLabel = nLabel+" "+self.infoString

		if not dirLabel == "": nLabel = dirLabel
			
		if opt=="Cl":
			fit = self.createD2Histos(minD2, maxD2, "Cl", resolution=resolution)
			if norm:
				fit[0] = fit[0]*1.0/sum(self.histoDSquaredDistr)
				self.histoDSquaredDistr = np.array(self.histoDSquaredDistr)*1.0/sum(self.histoDSquaredDistr)
			plt.plot(self.edgesDSquaredDistr, np.concatenate([[self.histoDSquaredDistr[0]],self.histoDSquaredDistr]), drawstyle='steps-pre', linestyle=linestyle, color=self.pltColor, label=nLabel)
			plt.title("")#r" $ D^{2} $-Distribution for $ C_{l} $")
			plt.xlabel("$ D^{2} $")
			plt.grid(True)
		fit=[]	
		if opt=="effCl":
			if fitfunc=="cauchy":
				fit = self.getCauchyFit(minD2, maxD2, resolution=resolution, plotting=plotting)
			elif fitfunc=="exponential":
				fit = self.getExpFit(minD2, maxD2, resolution=resolution, plotting=plotting, nFigure=nFigure, percentile=nPercentile)
			elif fitfunc=="studentT":
				fit = self.getStudFit(minD2, maxD2, resolution=resolution, plotting=plotting, nFigure=nFigure)
			elif fitfunc=="gamma":
				fit = self.createD2Histos(minD2, maxD2, "effCl",resolution=resolution, fitfunc=fitfunc, shift=shift, fitting=fitting)
			elif fitfunc=="gamma+exp":
				fit = self.createD2Histos(minD2, maxD2, "effCl",resolution=resolution, fitfunc=fitfunc, shift=shift, fitting=fitting)
			elif fitfunc=="gaussexp":
				fit = self.getGausExpFit(minD2, maxD2, resolution=resolution, tolerance=tolerance, maxNruns=maxNruns)
				self.GausExpFitParams=fit[0]
			elif fitfunc =="gaussexpconvolv":
				fit= self.getGausExpConvFit(minD2, maxD2, resolution=resolution, tolerance=tolerance, maxNruns=maxNruns)
				self.GausExpFitParams=fit[0]
				self.FitErr=fit[2]
			else:
				fit = self.createD2Histos(minD2, maxD2, "effCl",resolution=resolution, fitfunc=fitfunc, shift=shift, fitting=fitting)
				if norm: 
					fit[0] = fit[0]*1.0/sum(self.histoEffDSquaredDistr)
					self.histoEffDSquaredDistr = np.array(self.histoEffDSquaredDistr)*1.0/sum(self.histoEffDSquaredDistr)
			print len(fit[0])
			if not real_data:
				if len(fit[0])==1:
					self.histoEffDSquaredDistr, self.edgesEffDSquaredDistr = np.histogram(self.effDsquaredDistr, bins=50, range=(minD2, maxD2))
				plt.plot(self.edgesEffDSquaredDistr, np.concatenate([[self.histoEffDSquaredDistr[0]],self.histoEffDSquaredDistr]), drawstyle='steps-pre', linestyle=linestyle, color=self.pltColor) #label=nLabel+", "+self.infoString
				plt.vlines(np.median(self.effDsquaredDistr), 0.0001, 1, color=self.pltColor, linestyle="-", label=nLabel)
				plt.vlines(np.percentile(self.effDsquaredDistr, nPercentile*100.), 0.0001, 1, color=self.pltColor, linestyle="--")
				plt.vlines(np.percentile(self.effDsquaredDistr, nPerc2*100.), 0.0001, 1, color=self.pltColor, linestyle="--")
				plt.title("")#r" $ D^{2}_{\mathrm{eff}} $-Distribution for $ C_{l}^{\mathrm{eff}} $")
				plt.xlabel("$ D^{2}_{\mathrm{eff}} $")	
				plt.ylim(0,np.max(np.concatenate([[self.histoEffDSquaredDistr[0]],self.histoEffDSquaredDistr])))
				plt.grid(True)
			
		if opt=="sqrtEffCl":
			fit = self.createD2Histos(minD2, maxD2, "sqrtEffCl", resolution=resolution)
			plt.plot(self.edgesSqrtEffDSquaredDistr, np.concatenate([[self.histoSqrtEffDSquaredDistr[0]],self.histoSqrtEffDSquaredDistr]), drawstyle='steps-pre', linestyle=linestyle, color=self.pltColor, label = nLabel) #label=nLabel+", "+self.infoString
			plt.title(r" $ D^{2}_{eff, sqrt} $-Distribution for $ \sqrt{C_{l}^{\mathrm{eff}}} $")
			plt.xlabel("$ D^{2}_{eff, sqrt} $")
		if opt=="ClLog":
			fit = self.createD2Histos(minD2, maxD2, "ClLog", resolution=resolution)
			plt.plot(self.edgesDSquaredLogDistr, np.concatenate([[self.histoDSquaredLogDistr[0]],self.histoDSquaredLogDistr]), drawstyle='steps-pre', linestyle=linestyle, color=self.pltColor, label = nLabel) #label=nLabel+", "+self.infoString
			plt.title(r" $ D^{2}_{Log} $-Distribution for $ C_{l}^{Log} $")
			plt.xlabel("$ D^{2}_{Log} $")
		if opt=="almCl":
			fit = self.createD2Histos(minD2, maxD2, "almCl", resolution=resolution)
			plt.plot(self.edgesAlmDSquaredDistr, np.concatenate([[self.histoAlmDSquaredDistr[0]],self.histoAlmDSquaredDistr]), drawstyle='steps-pre', color=self.pltColor, label = self.infoString) #label=nLabel+", "+self.infoString
			plt.title("$ D^{2}_{alm} $ -Distribution")
			plt.xlabel("$ D^{2}_{alm} $")
		if opt=="effAlmCl":
			fit = self.createD2Histos(minD2, maxD2, "effAlmCl", resolution=resolution)
			plt.plot(self.edgesEffAlmDSquaredDistr, np.concatenate([[self.histoEffAlmDSquaredDistr[0]],self.histoEffAlmDSquaredDistr]), drawstyle='steps-pre', color=self.pltColor, label = self.infoString) #label=nLabel+", "+self.infoString
			plt.title("$ D^{2}_{alm,eff} $ -Distribution")
			plt.xlabel("$ D^{2}_{alm,eff} $")
		plt.ylabel("probability")#"counts")
		plt.grid(True)
		plt.legend(prop={"size":30.0}, loc="best", fancybox=True) 
		if fitfunc=="gamma":
			if fitting:	plotGammaDistr(np.linspace(minD2,maxD2, 5000), fit[0], fit[1], fit[2], fit[3], color=self.pltColor, linestyle=linestyle)
			
		if fitfunc=="gauss":
			if fitting: plotGaussian(np.linspace(minD2,maxD2, 5000) ,fit[0],fit[1], fit[2], color=self.pltColor, linestyle=linestyle)
			
		if fitfunc=="gaussexp" or fitfunc=="gaussexpconvolv":
			fig2=plt.figure(51)
			fig=plt.figure(50)
			ax = fig.add_subplot(111)
			popt=self.GausExpFitParams
			if fitting and len(popt)!=1: 
				xa=setNewEdges(self.edgesEffDSquaredDistr)
				ya=self.histoEffDSquaredDistr
				normTS=(xa[-1]-xa[0])/len(xa)*len(self.effDsquaredDistr)
				ya_err = np.sqrt(self.histoEffDSquaredDistr)
				if len(popt)==5:
					x=np.linspace(np.min(xa)-abs(np.max(xa))/4,3*np.max(xa), 10000)
					dist=init_sens_function(popt)
					norm=dist(np.inf)
					plt.plot(x, gaussexpfunc(np.array(x), popt[0],popt[1], popt[2],popt[3],popt[4])/norm, color=self.pltColor, label=self.infoString+" $\chi^2$"+"={:0.2f}".format(fit[1]))
				if len(popt)==4:
					if isBG==True:
						plt.errorbar(np.array(xa)*10**4,ya/normTS/10**4, ya_err/normTS/10**4, color=self.pltColor, linestyle="None")
						dist=init_sens_function(popt)
						norm=dist(np.inf)		
						quantiles=[0.9987,0.999997] #3 and 4 sigma quantile
						maxTS=0.
						for i in quantiles:
							quantile=fsolve(lambda x: dist(x)/norm-i, 0.001)
							print "pValue {} at TS {}".format(i, quantile)
							plt.figure(50)
							plt.vlines(quantile*10**4, 1e-6, 0.35 , linestyles='--',  linewidth=2.0, color="k")
							plt.figure(51)
							plt.vlines(quantile*10**4, 1e-6, 1. , linestyles='--',  linewidth=2.0, color="k")
							if quantile>maxTS:
								maxTS=quantile
						x=np.linspace(np.min(xa)-abs(np.max(xa))/4,maxTS*1.02, 10000)
						#plt.xlim(np.min(self.effDsquaredDistr),maxTS*1.02)
					else:
						plt.scatter(np.array(xa)*10**4,ya/normTS/10**4, color=self.pltColor)
						x=np.linspace(np.min(xa)-abs(np.max(xa))/4, 2*np.max(xa), 10000)
						dist=init_sens_function(popt)
						norm=dist(np.inf)
					plt.figure(50)
					plt.plot(x*10**4, gaussexpconvolv(np.array(x), popt[0],popt[1], popt[2],popt[3])/norm/10**4, color=self.pltColor,label=self.infoString+" $\chi^2/\mathrm{doF}$"+"={:0.2f}".format(fit[1]) )#
					plt.figure(51)
					plt.errorbar(np.array(xa)*10**4,ya/normTS/10**4, ya_err/normTS/10**4, color=self.pltColor, linestyle="None")
					plt.semilogy(x*10**4, gaussexpconvolv(np.array(x), popt[0],popt[1], popt[2],popt[3])/norm/10**4, color=self.pltColor, label=self.infoString+" $\chi^2/\mathrm{doF}$"+"={:0.2f}".format(fit[1]))

		return fit

		


	### CALC alm NormDiff PLANE
	def calcNormDiffPlane(self, nOverErr=True, nShowLZero=True):
		if nOverErr == True:
			self.summedErr_SigOrg = getDifferenceError(self.ORG.alm_errors, self.SIG.alm_errors)
			self.plot_alm_normDiffOverErr = getAlmNormDiffPlane(self.SIG.alm_means, self.ORG.alm_means, self.summedErr_SigOrg, nShowLZero)
		else:
			self.plot_alm_normDiff = getAlmNormDiffPlane(self.SIG.alm_means, self.ORG.alm_means, nShowLZero)


	### PLOT alm NormDiff PLANE
	def plotNormDiffPlane(self, nFigure=-1, nOverErr=True, nVmin=-1.5, nVmax=1.5):
		plotControl(nFigure)
		if(self.compType == "ATM_var1"):
			nLabel = "atm. changed"
		if(self.compType == "SIG_NSources"):
			nLabel = "various Sig."
		elif(self.compType == "PURE_SIG"):
			nLabel = "pure-signal spec."

		if nOverErr == True:	
			if nVmin==0 and nVmax == 0:
				plt.imshow(self.plot_alm_normDiffOverErr , interpolation='nearest')
			else:
				plt.imshow(self.plot_alm_normDiffOverErr , interpolation='nearest', vmin=nVmin, vmax=nVmax)
			cb = plt.colorbar()
			cb.set_label(r'$ (\langle a_{l}^{m}\rangle - \langle a_{l,atm}^{m}\rangle )/\sigma_{l,atm}^{m} $')
		else:
			if nVmin==0 and nVmax == 0:
				plt.imshow(self.plot_alm_normDiff , interpolation='nearest')
			else:
				plt.imshow(self.plot_alm_normDiff , interpolation='nearest', vmin=nVmin, vmax=nVmax)
			cb = plt.colorbar()
			cb.set_label(r'$ (\langle a_{l}^{m}\rangle - \langle a_{l,atm}^{m}\rangle) $')
		plt.title(nLabel+", "+self.infoString)
		plt.ylabel("$ \ell $")
		plt.xlabel("m")

	
	### PLOT Al0 DISTR.
	def plotNormAl0DiffDistr(self, nFigure=-1,nOverErr=True):
		plotControl(nFigure)
		if(self.compType == "ATM_var1"):
			nLabel = "atm. changed"
		if(self.compType == "SIG_NSources"):
			nLabel = "various Sig."
		elif(self.compType == "PURE_SIG"):
			nLabel = "pure-signal spec."

		if nOverErr == True:
			self.Al0NormList = getAl0NormDiffList(self.SIG.alm_means, self.ORG.alm_means, self.summedErr_SigOrg)
			plt.errorbar(self.ls, self.Al0NormList, fmt='o', color=str(self.pltColor), label=nLabel+", "+self.infoString)
			plt.ylabel(r'$ (\langle a_{l}^{0}\rangle -\langle a_{l, atm}^{0}\rangle )/ \sigma_{abs,atm} $')
			plt.title(r" $ \langle \| a_{l}^{0} \| \rangle $ differences, "+nLabel+", "+self.infoString)
		else:
			self.Al0NormList = getAl0NormDiffList(self.SIG.alm_means, self.ORG.alm_means)
			plt.errorbar(self.ls, self.Al0NormList, fmt='o', color=str(self.pltColor), label=nLabel+", "+self.infoString) #" $ \langle \| a_{l}^{0} \| \rangle $ differences, "+
			plt.ylabel(r'$ (\langle a_{l}^{0}\rangle-\langle a_{l, atm}^{0}\rangle)$')
			plt.title(r" $ \langle \| a_{l}^{0} \| \rangle $ differences, "+nLabel+", RUNS: "+str(NUMBER_RAN_RUNS))
		plt.xlabel("$ \ell $")


	#### SET NEW PLOTTING COLOR	
	def setPltColor(self, nColor):
		self.pltColor = nColor


	#### SAVE WEIGHTS TO FILE
	def saveWeights(self, Opt="Cl", nPath=""):
		if nPath == "":
			nPath = self.SAVE_DIR

		if(self.compType == "ATM_var1"):
			nLabel = "varAtm"+str(round(self.SIG.curPerc,3))
		if(self.compType == "SIG_NSources"):
			nLabel = "varSigNSources"+str(self.SIG.curNSources)+"Mu"+str(self.SIG.curMuSources)		
		if(self.compType == "PURE_SIG"):
			nLabel = "pureSigNEvents"+str(self.SIG.NUMBER_SIMULATED_EVENTS)+"N"+str(self.SIG.curNSources)
		if(self.compType == "PureAtm"):
			nLabel = "pureAtmNEvents"+str(self.SIG.NUMBER_SIMULATED_EVENTS)

		if nPath == "":
			nPath = self.SAVE_DIR
		if Opt=="Cl":
			np.savetxt(nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsCl_hist.txt", self.clChangedWeight)
			print "data A(weights Cl) saved to "+nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsCl_hist.txt"
		if Opt=="effCl":
			np.savetxt(nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsEffCl_hist.txt", self.effClChangedWeight)
			print "data B(weights effCl) saved to"+nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsEffCl_hist.txt"
		if Opt=="sqrtEffCl":
			np.savetxt(nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsSqrtEffCl_hist.txt", self.sqrtEffClChangedWeight)
			print "data D(weights sqrtEffCl) saved to"+nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsSqrtEffCl_hist.txt"
		if Opt=="ClLog":
			np.savetxt(nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsClLog_hist.txt", self.ClLogChangedWeight)
			print "data C(weights ClLog) saved to"+nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsClLog_hist.txt"
		if Opt=="almCl":
			np.savetxt(nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsAlmCl_hist.txt", self.almWeights)
			print "data D(weights almCl) saved."
		return nPath+nLabel+"RUNS"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+"_weightsEffCl_hist.txt"


	#### SAVE ALL WEIGHTS
	def saveAllWeights(self, nPath="", optCl=False):
		if nPath == "":
			nPath = self.SAVE_DIR+"data/"
		if optCl==True:
			self.saveWeights("Cl",nPath)

		if self.SIG.useAlm == True:
			self.saveWeights("almCl", nPath)
		if self.SIG.useClLog == True:
			self.saveWeights("ClLog", nPath)
		if self.SIG.useSqrtEffCl == True:
			self.saveWeights("sqrtEffCl", nPath) 
		ret = self.saveWeights("effCl",nPath)
		return ret


	#### READ Cl WEIGHTS																###### NEW ########
	def readClWeights(self, nPath):
		self.clChangedWeight = np.genfromtxt(str(nPath))
		print r"reading weights from '"+str(nPath)+"'"
		if self.SIG.l_max+1 < len(self.clChangedWeight):
			self.clChangedWeight = self.clChangedWeight[0:self.SIG.l_max+1]
			print "ML-WARNING: Length of weight-list was reduced to fit to l_max (cl-weights)"
			
	#### READ ClLog WEIGHTS
	def readClLogWeights(self, nPath):
		self.clLogChangedWeight = np.genfromtxt(str(nPath))
		print r"reading weights from '"+str(nPath)+"'"
		if self.SIG.l_max+1 < len(self.clLogChangedWeight):
			self.clLogChangedWeight = self.clLogChangedWeight[0:self.SIG.l_max+1]
			print "ML-WARNING: Length of weight-list was reduced to fit to l_max (clLog-weights)"


	#### READ effCl WEIGHTS																###### NEW ########
	def readEffClWeights(self, nPath):
		self.effClChangedWeight = np.genfromtxt(str(nPath))
		#print r"reading weights from '"+str(nPath)+"'"
		if self.SIG.l_max < len(self.effClChangedWeight):
			self.effClChangedWeight = self.effClChangedWeight[0:self.SIG.l_max]
			print "ML-WARNING: Length of weight-list was reduced to fit to l_max (effCl-weights)"


	#### READ sqrtEffCl WEIGHTS																###### NEW ########
	def readSqrtEffClWeights(self, nPath):
		self.sqrtEffClChangedWeight = np.genfromtxt(str(nPath))
		print r"reading weights from '"+str(nPath)+"'"
		if self.SIG.l_max < len(self.sqrtEffClChangedWeight):
			self.sqrtEffClChangedWeight = self.sqrtEffClChangedWeight[0:self.SIG.l_max]
			print "ML-WARNING: Length of weight-list was reduced to fit to l_max (sqrtEffCl-weights)"
			

	#### READ WEIGHTS FROM FILE
	def readAllWeights(self, nPath="", almToo=True):
		if(self.compType == "ATM_var1"):
			nLabel = "varAtm"
		if(self.compType == "SIG_NSources"):
			nLabel = "varSig"

		if nPath == "":
			nPath = self.READ_DIR
		nClWeightFile = nLabel+str(round(self.SIG.curPerc,3))+"WeightsSameBins_hist"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+".txt"
		nEffClWeightFile = nLabel+str(round(self.SIG.curPerc,3))+"EffWeightsSameBins_hist"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+".txt"
		nAlmClWeightFile = nLabel+str(round(self.SIG.curPerc,3))+"AlmWeightsSameBins_hist"+str(int(self.SIG.NUMBER_RAN_RUNS_CHANGED))+".txt"

		self.clChangedWeight = np.genfromtxt(str(nPath)+str(nClWeightFile))
		self.effClChangedWeight = np.genfromtxt(str(nPath)+str(nEffClWeightFile))
		if (almToo==True):
			self.almWeights = np.genfromtxt(str(nPath)+str(nAlmClWeightFile))
	
		print r"reading weights from '"+str(nPath)+str(nClWeightFile)+"'"
		print r"reading weights from '"+str(nPath)+str(nEffClWeightFile)+"'"
		print r"reading weights from '"+str(nPath)+str(nAlmClWeightFile)+"'"
				
		if self.SIG.l_max< len(self.clChangedWeight):
			self.clChangedWeight = self.clChangedWeight[0:self.SIG.l_max+1]
			print "ML-WARNING: Length of weight-list was reduced to fit to l_max (cl-weights)"
		if self.SIG.l_max < len(self.effClChangedWeight):
			self.effClChangedWeight = self.effClChangedWeight[0:self.SIG.l_max]
			print "ML-WARNING: Length of weight-list was reduced to fit to l_max (effCl-weights)"



	#### PLOTS Chi2 DISTR. ####													##### NEW #####
	def calcChi2(self, Opt="Cl", calcLimits=[], cummulative=False):
		if Opt == "Cl":
			self.Chi2Distr = []
			for i in range(0, len(self.SIG.cl_all)):
				curChi2 = calcChi2(self.SIG.cl_all[i], self.ORG.cl_means, self.ORG.cl_errors, calcLimits)
				self.Chi2Distr.append(curChi2)

		elif Opt == "effCl":
			self.effChi2Distr = []
			for i in range(0, len(self.SIG.effCl_all)):
				curChi2 = calcChi2(self.SIG.effCl_all[i], self.ORG.effCl_means, self.ORG.effCl_errors, calcLimits)
				self.effChi2Distr.append(curChi2)
			if cummulative == True:
				self.cummulativeEffChi2 		= []
				self.cummulativeEffChi2_err 	= []
				curLmaxCummulativeAll	= [0.0 for i in range(0, len(self.SIG.effCl_all))]
								
				for thisLmax in range(0, self.SIG.l_max):
					
					for i in range(0, len(self.SIG.effCl_all)):
						curLmaxCummulativeAll[i] += (np.power((self.SIG.effCl_all[i][thisLmax]-self.ORG.effCl_means[thisLmax])*
															1.0/self.ORG.effCl_errors[thisLmax],2))
					self.cummulativeEffChi2.append(np.mean(curLmaxCummulativeAll))
					self.cummulativeEffChi2_err.append(np.std(curLmaxCummulativeAll))
					#print np.mean(curLmaxCummulativeAll)
					
					


	#### PLOTS Chi2 DISTR. ####													##### NEW #####
	def plotChi2Distr(self, minChi, maxChi, nFigure=-1, Opt="Cl", dirLabel=""):
		plotControl(nFigure)
		nLabel = ", "+self.infoString
		if not dirLabel == "": nLabel=dirLabel
		
		if Opt=="Cl":
			self.histoChi2Distr, self.edgesChi2Distr = np.histogram(self.Chi2Distr, bins=100, range=(minChi, maxChi))
			plt.plot(self.edgesChi2Distr, np.concatenate([[self.histoChi2Distr[0]],self.histoChi2Distr]), drawstyle='steps-pre', color=self.pltColor, label=nLabel)
			plt.title(r" $ \chi^{2} $-Distribution for $ C_{l} $")
			plt.xlabel("$ \chi^{2} $")
			plt.ylabel("counts")
		elif Opt=="effCl":
			self.histoEffChi2Distr, self.edgesEffChi2Distr = np.histogram(self.effChi2Distr, bins=100, range=(minChi, maxChi))
			plt.plot(self.edgesEffChi2Distr, np.concatenate([[self.histoEffChi2Distr[0]],self.histoEffChi2Distr]), drawstyle='steps-pre', color=self.pltColor, label=nLabel)
			plt.title(r" $ \chi^{2} $-Distribution for $ C_{l}^{\mathrm{eff}} $")
			plt.xlabel("$ \chi^{2} $")
			plt.ylabel("counts")
			self.histoEffChi2Distr, self.edgesEffChi2Distr = np.histogram(self.effChi2Distr, bins=100, range=(minChi, maxChi))
			
			
	
	#### COUNTS DEVIATIONS IN CORRECT AND IN WRONG DIRECTION FOR (eff)Cl ####
	def calcDeviationSigns(self, opt="Cl"):
		if opt=="Cl":
			self.deviationsCl_counts = np.zeros(self.SIG.l_max+1)
			self.ClDifference_counts= np.zeros(self.SIG.l_max+1)
			for i in range(0, len(self.SIG.cl_all)):
				for l in range(0, self.SIG.l_max+1):
					self.deviationsCl_counts[l] += returnSign(self.SIG.cl_all[i][l]-self.ORG.cl_all[i][l])*returnSign(self.clChangedWeight[l])
					self.ClDifference_counts[l] += returnSign(self.SIG.cl_all[i][l]-self.ORG.cl_all[i][l])
			return self.deviationsCl_counts
		elif opt=="effCl":
			self.deviationsEffCl_counts = np.zeros(self.SIG.l_max)
			self.effClDifference_counts = np.zeros(self.SIG.l_max)
			for i in range(0, len(self.SIG.effCl_all)):
				for l in range(0, self.SIG.l_max):
					self.deviationsEffCl_counts[l] += returnSign(self.SIG.effCl_all[i][l]-self.ORG.effCl_all[i][l])*returnSign(self.effClChangedWeight[l])
					self.effClDifference_counts[l] += returnSign(self.SIG.effCl_all[i][l]-self.ORG.effCl_all[i][l])
			return self.deviationsEffCl_counts

		
		
	#### PLOT DEVIATIONS IN CORRECT AN WRONG DIRECTION FOR Cl ####
	def plotDeviationSigns(self, nFigure=-1, opt="Cl", plotDirectSign=False, linestyle="-"):
		plotControl(nFigure)
		if opt=="Cl":
			plt.plot(self.ls, self.deviationsCl_counts, color=self.pltColor, label=self.infoString, linestyle=linestyle)
			plt.title(r" # of deviations in expected direction ")
			plt.xlabel("$ \ell $")
			plt.ylabel("counts")
			if plotDirectSign == True:
				plt.plot(self.ls, self.ClDifference_counts, linestyle="--", color=self.pltColor, label=self.infoString)
		elif opt=="effCl":
			plt.plot(self.lsEff, self.deviationsEffCl_counts, color=self.pltColor, label=self.infoString, linestyle=linestyle)
			plt.title(r" # of deviations in expected direction ")
			plt.xlabel("$ \ell $")
			plt.ylabel("counts")
			if plotDirectSign == True:
				plt.plot(self.lsEff, self.effClDifference_counts, linestyle="--", color=self.pltColor, label=self.infoString)	
				
				
	#### SAVE D2 AND effD2 DISTRIBUTION ####
	def saveD2Distr(self, path, opt="Cl"):
		if self.usedLogForD2 == True:
			extension = "log"
		else:
			extension = ""
		
		if opt == "Cl":
			extension = extension+"Cl_"
			fullPath = path+"_D2_Distr_"+extension+".txt"
			np.savetxt(fullPath, self.DsquaredDistr)
			print "D2 have been saved to: "+str(fullPath)
		elif opt == "effCl":
			extension = extension+"EffCl_"
			fullPath = path+"_D2_Distr_"+extension+".txt"
			np.savetxt(fullPath, self.effDsquaredDistr)
			print "effD2 have been saved to: "+str(fullPath)
			
			
	#### SAVE D2 AND effD2 DISTRIBUTION ####
	def readD2Distr(self, path, opt="Cl"):
		if self.usedLogForD2 == True:
			extension = "log"
		else:
			extension = ""
		
		if opt == "Cl":
			extension = extension+"Cl_"
			fullPath = path+"_D2_Distr_"+extension+".txt"
			self.DsquaredDistr = np.genfromtxt(fullPath)
			print "D2 have been loaded from: "+str(fullPath)
		elif opt == "effCl":
			extension = extension+"EffCl_"
			fullPath = path+"_D2_Distr_"+extension+".txt"
			self.effDsquaredDistr = np.genfromtxt(fullPath)
			print "effD2 has been loaded from: "+str(fullPath)
			
		
	##### SQUARES ALL WEIGHTS #####
	def squareWeights(self):
		self.clChangedWeight = np.power(self.clChangedWeight,2)
		self.effClChangedWeight = np.power(self.effClChangedWeight,2)
		print "ML-WARNING: ALL WEIGHTS HAVE BEEN SQUARED!!!"
				
		
	#### PLOT effCl DISTRIBUTION
	def pltWeightedClDiff(self, nFigure=-1, opt="Cl",  nLabel=""):
		plotControl(nFigure)

		if opt == "Cl":
			self.weightedClDiff	= [returnSign(self.SIG.cl_means[l]-self.ORG.cl_means[l]) *
									self.clChangedWeight[l]*1.0/sum(self.clChangedWeight)*
									np.power(  (self.SIG.cl_means[l]-self.ORG.cl_means[l])*1.0/
									self.ORG.cl_errors[l],2 ) for l in range(0,len(self.SIG.cl_means))]
			plt.errorbar(self.ls, self.weightedClDiff, fmt='o', color=str(self.pltColor), label=nLabel)
			plt.xlabel("$ \ell $")
			plt.ylabel("$ sign_{l} \cdot w_{\ell} \cdot (C_{\ell, exp} - C_{\ell, bg})^{2} / \sigma^{2}_{C_{\ell, bg}} $")
			plt.title(r" weighted $ C_{\ell} $ - deviations") 
			#plt.yscale("log")
			
		if opt == "effCl":
			#self.weightedEffClDiff = [1.0/self.SIG.l_max*returnSign(self.SIG.effCl_means[l]-self.ORG.effCl_means[l])*
			#							self.effClChangedWeight[l]*1.0/sum(self.effClChangedWeight)
			#							*np.power(  #(self.SIG.effCl_means[l]-self.ORG.effCl_means[l])*1.0/self.ORG.effCl_errors[l],2) for l in range(0,(self.SIG.l_max))]
			self.weightedEffClDiff = []
			for i in range(0, self.SIG.l_max):
				this = np.mean([calcEffD2(self.SIG.effCl_all[k], self.ORG.effCl_means,
								self.ORG.effCl_errors, self.effClChangedWeight, 
								self.SIG.l_max, norm=True, limits=[0, i]) for k in range(0, len(self.SIG.effCl_all))])
				print this
				self.weightedEffClDiff.append(this)
						
			#self.weightedEffClDiff = [np.mean([calcEffD2(self.SIG.effCl_all[k], self.ORG.effCl_all[k],
			#							self.ORG.effCl_errors, self.effClChangedWeight, 
			#							self.SIG.l_max, norm=True, limits=[0, l]) for k in range(0, len(self.ORG.effCl_all))]) #for l in range(0, self.SIG.l_max)]
										
			plt.errorbar(self.lsEff, self.weightedEffClDiff, fmt='o', color=str(self.pltColor), label=nLabel)
			plt.xlabel("$ \ell $")
			plt.ylabel("$ 1/l_{max} \cdot sign_{l} \cdot (w_{\ell}^{\mathrm{eff}} / \Sigma_{\ell'} w_{\ell'}^{\mathrm{eff}})\cdot (C_{\ell, exp}^{\mathrm{eff}} - C_{\ell, bg}^{\mathrm{eff}})^{2} / \sigma_{C_{\ell, bg}^{\mathrm{eff}}}^{2} $")
			plt.title(r" weighted $ C_{\ell}^{\mathrm{eff}} $ - deviations, ev.") 
			#plt.yscale("log")	
			self.cummulativeWeightedEffClDiff = [ sum(self.weightedEffClDiff[0:l]) for l in range(0, len(self.weightedEffClDiff))]
				
