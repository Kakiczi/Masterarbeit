#!/usr/bin/env python
import matplotlib.pyplot as plt
# test lolololo
############################ IMPORTS ############################################################
from functions4 import *
import healpy as H
from classes4 import *
import copy
from random import shuffle
import sys, os, cPickle, ConfigParser

allParam = getParameters()
##### GENERAL QUANTITIES 
pathOption=""
NUMBER_RAN_RUNS 	= 1 										### do not change!
l_max				= 1000

##### MAP CONFIGURATION                         
SMEARING_METHOD = "none"
RENORMALIZATION = True
USE_GALACTIC_PLANE = True
CORRECT_FOR_AEFF = True
HIT_CENTERING 	= True
USE_ALM 			  = True
USE_CL_LOG 			= False
SHOW_PLOTS 			= True
SAVE_MAPS 			= False

SAVE_ENERGY 		= True
SAVE_MU         = True
SAVE_ANG        = True


if socket.gethostname()=='theo-UL50VT':
        localPath="/home/theo/fusessh/"
else:
        localPath = "/net/scratch_icecube4/user/glauch/"
        
datapath = localPath+"DATA/"

GAMMA_BF, _ =get_best_fit_values()  #Best -Fit Values

if "UseGalPlane" in allParam:
	USE_GALACTIC_PLANE = bool(allParam["UseGalPlane"] == "True")
else: USE_GALACTIC_PLANE = True

##### GAMMA
if "gamma" in allParam: 
	GAMMA = float(allParam["gamma"])
else:
	GAMMA=GAMMA_BF

if "experiment" in allParam:
	EXPERIMENT = bool(allParam["experiment"]=="True")
	if EXPERIMENT==True:
		pathOption +="_Experiment"
else: EXPERIMENT = False
DETECTOR_config=allParam["detector"]
DETECTOR=LoadDETECTOR(DETECTOR_config)	
#if os.path.exists("/net/scratch_icecube4/user/glauch/MCode/data/NumberNeutrinos"+str(GAMMA)+".npy"):
#	ndict= np.load("/net/scratch_icecube4/user/glauch/MCode/data/NumberNeutrinos"+str(GAMMA)+".npy")
#else:
#	print "There is no Number of Neutrinos File for Gamma: "+str(GAMMA)
print "DETECTOR Configuration:"
print DETECTOR
samples=dict()
ndict=dict()
##Load all MC samples in DETECTOR		
for j in DETECTOR:  
	nneutrinos=NumberofSimulatedEvents(2.13, j, ind_mode=False)
	if "gamma" in allParam:
		samples[j]=simulationLoad(j, datapath, GAMMA, exp_bool=EXPERIMENT)
		ndict[j]=[sum(samples[j][0]["astro"]), nneutrinos-sum(samples[j][0]["astro"]), nneutrinos]
	else:
		samples[j]=simulationLoad(j, datapath, exp_bool=EXPERIMENT)	
		ndict[j]=[sum(samples[j][0]["astro"]), nneutrinos-sum(samples[j][0]["astro"]), nneutrinos]


if not os.path.exists(localPath+"MCode/data/NumberNeutrinos"+str(GAMMA)+".npy"):
	np.save(localPath+"MCode/data/NumberNeutrinos"+str(GAMMA)+".npy",ndict)
	print "SAVED number of neutrinos dict"
##### PARAMETERS 


PROJECT_ID 	 = str(sys.argv[1])
ADD_TO_RUN_ID= int(sys.argv[2])
RUN_NUMBER   = int(sys.argv[3])
SIGNAL       = str(sys.argv[4])   				# OPTIONS: PureAtm, PureSig, VarAtm, VarSig
if SIGNAL == "VarSig" or SIGNAL == "PureSig":
	MU_SOURCES = sys.argv[5]
	if MU_SOURCES[0]!="[" or MU_SOURCES[-1]!="]":
		raise SyntaxError('Syntax incorrect. First and last character have to be square bracket')
	else:
		MU_SOURCES=[float(i) for i in MU_SOURCES[1:-1].split(",")]

	N_SOURCES  = sys.argv[6]		
	if N_SOURCES[0]!="[" or N_SOURCES[-1]!="]":
		raise SyntaxError('Syntax incorrect. First and last character have to be square bracket')
	else:
		N_SOURCES=[int(i) for i in N_SOURCES[1:-1].split(",")]		
		
	if len(MU_SOURCES)!=len(N_SOURCES):
		raise ValueError("The Mu and N Array have to be of same length")
        
if SIGNAL == "VarAtm":  ###What is that even? VarAtm...Sounds strange to me :-)
	CUR_PERC   = float(sys.argv[5])
	
##### OPTIONS
if "nside" in allParam: 
	nside = np.power(2, int(allParam["nside"]))
	npix=H.nside2npix(nside)
else: print "nside as usual: "+str(nside)

if "poslist" in allParam:
	usePosList=allPAram["poslist"]
	print "Using Position List {} for creation of Skymap".format(usePosList)
else: usePosList=""

if "catalog" in allParam:
	catalog=allParam["catalog"]
else: catalog="" 

if "fullSphere" in allParam: 
	FULL_SPHERE_MODE = bool(allParam["fullSphere"] == "True")
	if FULL_SPHERE_MODE: pathOption+="_fullSphere"
else:  FULL_SPHERE_MODE = False

NUMBER_EV_PER_SAMPLE		= [int(ndict[i][2]) for i in DETECTOR] 
print "Number of Events per Chosen Sample:"
print NUMBER_EV_PER_SAMPLE
NUMBER_SIMULATED_EVENTS 	= sum(NUMBER_EV_PER_SAMPLE)


if FULL_SPHERE_MODE == True:
	NUMBER_SIMULATED_EVENTS = 2*NUMBER_SIMULATED_EVENTS


###########################ADD other Source Catalogs for Cross-Correlations if necesarry################
if catalog=="NVSS":
	print "Reading Source Catalog" 
	NVSS=readNVSS(skip_line=16)
	catalog_map=createNVSSSkymap(NVSS, npix, USE_GALACTIC_PLANE, RENORMALIZATION, CORRECT_FOR_AEFF, NUMBER_SIMULATED_EVENTS)

########################################################################################################

if "useCatalogPos" in allParam:
	try:
		useCatalogPos=int(allParam["useCatalogPos"])
		print "Using {} Sources from Source Catalog for creation of Skymap".format(useCatalogPos)
	except ValueError:
		"Please Set useCatalogPos as an Integer representing the Number of strongest Sources you want to include from the Source Catalog"
else: useCatalogPos=-1


if "useweakestsource" in allParam:
	try:
		useweakestsource=int(allParam["useweakestsource"])
	 	print "Using {} Sources from Source Catalog for creation of Skymap".format(useCatalogPos)
	except ValueError:
		"Please Set useCatalogPos as an Integer representing the Number of strongest Sources you want to include from the Source Catalog"
else: useweakestsource=-1

if "extended_source" in allParam:
	extended_source_angle=float(allParam["extended_source"])
	print "Use Extended Source with Angle of {}".format(extended_source_angle)
else:
	extended_source_angle=0.

if "stretchPS" in allParam: 
	PS_STRETCHING = float(allParam["stretchPS"])
	pathOption+="_PSStretching"+str(PS_STRETCHING)
else: PS_STRETCHING = 1.0
	
if "fixMu" in allParam: 
	MU_PRECISE = bool(allParam["fixMu"])
	if MU_PRECISE: pathOption+="_fixedMu"	
else: MU_PRECISE = False


if "milagro" in allParam: 
	MILAGRO_ANISO = bool(allParam["milagro"] == "True")
	if MILAGRO_ANISO: pathOption+="_Milagro"
else:  MILAGRO_ANISO = False

if "fixZenith" in allParam: 
	FIX_ZENITH = bool(allParam["fixZenith"] == "True")
	pathOption+="_fixZenith"
else:  FIX_ZENITH = False

if "galacticPlane" in allParam: 
	GALACTIC_PLANE = True
	print "ONLY Simulate Events in the GALATIC PLANE"
else:
	GALACTIC_PLANE = False
	
if "mcGen" in allParam:
	MC_GEN = bool(allParam["mcGen"] != "False")
else:
	MC_GEN = True
#~ print "MC_GEN="+str(MC_GEN)
if MC_GEN == True:
	pathOption+="_MCgen"
	
if "replaceMode" in allParam:
	REP = bool(allParam["replaceMode"] != "False")
else:
	REP = True
#~ print "replaceMode="+str(REP)

if "PSRotation" in allParam:
	PS_ROT = bool(allParam["PSRotation"] != "False")
else:
	PS_ROT = True
#~ print "PS_ROT="+str(PS_ROT)

if "useE" in allParam: 
	USE_ENERGY		= bool(allParam["useE"] == "True")
else:	
	USE_ENERGY		= True
if USE_ENERGY == True:
	pathOption+="_useE"
#~ print "USE_ENERGY="+str(USE_ENERGY)
	
#if "onlyIC79" in allParam:
	#ONLY_IC79 = bool(allParam["onlyIC79"]=="True")
pathOption +="_"+DETECTOR_config 
NSIG_EVENTS = [int(ndict[i][0]) for i in DETECTOR]
print "Based on a flux of Gamma="+str(GAMMA)+" there are " +str(NSIG_EVENTS)+ " Signal Events"

pathOption+="_E"+str(GAMMA)

if "useDiffBG" in allParam:
	USE_DIFF_BG = bool(allParam["useDiffBG"] == "True")
else: 
	USE_DIFF_BG = True
if USE_DIFF_BG ==True:	
	pathOption+="_useDiff"
#~ print "USE_DIFF_BG="+str(USE_DIFF_BG)
	
if "useN" in allParam:
	USE_N = bool(allParam["useN"]=="True")
else:
	USE_N = True
#~ print "USE_N="+str(USE_N)
	
if "allData" in allParam:
	ALL_DATA = bool(allParam["allData"]=="True")
#	SAVE_ANG = bool(allParam["allData"]=="True")
else:
	ALL_DATA = False
#	SAVE_ANG = False
#print "ALL_DATA="+str(ALL_DATA)
#print "SAVE_ANG="+str(SAVE_ANG)
	
if "specificM" in allParam:
	SPEC_M = bool(allParam["specificM"]=="True")
else:
	SPEC_M = True
#~ print "SPEC_M="+str(SPEC_M)
	
if ("zenBand1" and "zenBand2") in allParam:
	ZEN_INT = [float(allParam["zenBand1"]), float(allParam["zenBand2"])]
else:
	ZEN_INT = [-1., np.cos(np.pi/2-np.radians(5))]

if "varZenith" in allParam: 
	VAR_ZENITH		= float(allParam["varZenith"])
	pathOption+="_varZen"+str(VAR_ZENITH)
else:	VAR_ZENITH		= 0.0

if "multiMu" in allParam: 
	MULTI_MU_SOURCES = bool(allParam["multiMu"] == "True")
	if MULTI_MU_SOURCES: 
		MULTI_MU_EXPANSION = [[10],[100]]
		pathOption+="_multiMu"
		for i in range(0,len(MULTI_MU_EXPANSION[0])):
			pathOption += "_mu"+str(MULTI_MU_EXPANSION[0][i])+"N"+str(MULTI_MU_EXPANSION[1][i])
	print MULTI_MU_EXPANSION
else:  
	MULTI_MU_SOURCES = False
	MULTI_MU_EXPANSION = [[],[]]
	
if "cutOff" in allParam: 
	CUTOFF = True
	if CUTOFF: 
		pathOption+="_cutOff"+str(allParam["cutOff"])
		CUTOFF_VALUE = int(allParam["cutOff"])
else:  CUTOFF = False

if "mcEXP" in allParam:
	mcEXP = bool(allParam["mcEXP"]=="True")
	if "mcSET" in allParam:
		MC_SET = int(allParam["mcSET"])
	else:
		MC_SET = 0
else:
	mcEXP = False

 

GENEVA		  = False
GEN_OHNE_MU = False
ADD_SOME_MU = False
GEN_AND_MU	= False
	
if "shuffle" in allParam:
	### USE ONLY WITH EXPERIMENT ###
	SHUFFLE = bool(allParam["shuffle"]=="True")
else: 
	SHUFFLE = False
if EXPERIMENT and SHUFFLE==False:
	print "*** WARNING!!! EXPERIMENTAL MAPS ARE USED UN-SHUFFLED!!! ***"
	pathOption +="_realData"

if "RAAcceptance" in allParam:
	RA_ACC = bool(allParam["RAAcceptance"] == "True")
	pathOption +="_RAAcceptance"
else: RA_ACC = False

if not CUTOFF==False: 
	addFile = "_CutOff"+str(CUTOFF_VALUE)+"TeV"
	addDict = "_cutOff10"
else: 
	addFile = ""
	addDict = ""
print "ADDFILE = "+str(addFile)

	
##### GENERAL FILES NEEDED #####/net/scratch_icecube4/user/glauch/MCode/data

prePath 				= localPath+"MCode/data/"
file_acceptance = prePath+"IC79_mu_ratio"
#prePath         = prePath+"IC86-I/"


						
########### THIS WILL BE RELEVANT WHEN JOINING MULTIPLE DETECTOR CONFIGURATIONS ############
fileAeffContributions	= "IC79_Aeff_Contribution.dict"


##### RELICS #####							
fileGPUSmearingPS 	= "" #prePath+"IC79_GPUSmearing_sig_histEdges_100bins.txt"
fileMilagro 			  = prePath+"MilagroAnisotropyMap.txt"
fileFixZenith 			= prePath+"exp/IC40IC59IC79_allZenithsList.txt" #prePath+"IC79_allZenithsList.txt"
fileRAAcceptance		= prePath+"exp/IC40+59+79_EXP_3Y_RA_Acceptance_360bins.txt"
fileGalacticPlane		= "" #prePath+"galacticPlane_sourceDensity.txt"
#########################################

##### SAVE DIRECTORY

if not 'physik.rwth-aachen.de' in socket.gethostname():
	SAVE_DIR =  "/home/ls280163/data/batch/"
	print "Assume being on a BATCH machine..."
else:
	SAVE_DIR = "/net/scratch_icecube4/user/kalaczynski/Analysis_stuff/"
	print "Assume being on a CONDOR machine..."

#if EXPERIMENT:
	#SAVE_DIR = SAVE_DIR+DETECTOR_config+"_Skymaps"+pathOption+"/smearedExperiment/CPU_unsmearedExperiment_Ev"+str(NUMBER_SIMULATED_EVENTS)+"LMAX"+str(l_max)+"NSIDE"+str(nside)	
	
#else:
	#SAVE_DIR = SAVE_DIR+"1Y_Skymaps"+pathOption+"/smeared"+SIGNAL+"/CPU_unsmeared"+SIGNAL+"_Ev"+str(NUMBER_SIMULATED_EVENTS)+"LMAX"+str(l_max)+"NSIDE"+str(nside)
	
	
#if not EXPERIMENT or not mcEXP:
	#if SIGNAL == "PureSig":
		#N_STRING=""
		#for i, n in enumerate(N_SOURCES):	
			#N_STRING += str(n)
			#if i!=len(N_SOURCES)-1:
				#N_STRING+="-"	
		#SAVE_DIR = SAVE_DIR+"NSou"+str(N_STRING)
	#elif SIGNAL == "VarSig":
		#MU_STRING=""
		#N_STRING=""
		
		#for i, mu in enumerate(MU_SOURCES):
			#if mu == int(mu): 
				#MU_STRING += str(int(mu))
			#else:
				#MU_STRING += str(mu).replace(".","_")
			#if i!=len(MU_SOURCES)-1:
				#MU_STRING+="-"
				
		#for i, n in enumerate(N_SOURCES):	
			#N_STRING += str(n)
			#if i!=len(N_SOURCES)-1:
				#N_STRING+="-"					
		#SAVE_DIR = SAVE_DIR+"Mu"+MU_STRING+"NSou"+N_STRING
	#elif SIGNAL == "VarAtm":
		#SAVE_DIR = SAVE_DIR+"changePerc"+str(CUR_PERC)

#SAVE_DIR = SAVE_DIR+"/PROJECT_"+str(PROJECT_ID)+"/"

#### CREATE SAVE DIRECTORY
prepareDirectory(SAVE_DIR, subs = True)


############################
##### START ALL RUNS #######
#####    DEMANDED    #######
############################


### Create Neutrino Positions
for i in range(0, RUN_NUMBER):
	theta=[]
	phi_source=[]
	theta_source=[]
	if SIGNAL == "PureSig" or SIGNAL == "VarSig":
		if len(N_SOURCES)==1:
			if "Phi" in allParam and "Theta" in allParam:
				temp1=allParam["Theta"][1:-1].split(',')
				temp2=allParam["Phi"][1:-1].split(',')
				print temp1
				print temp2
				if len(temp1)==len(temp2) and len(temp1)==N_SOURCES[0]:
					theta_source.append([np.radians(float(l)) for l in temp1])
					phi_source.append([np.radians(float(l)) for l in temp2])
					theta=np.cos(theta_source)
				else:
					print "Mismatch between lenght of Phi and Theta or N_Sources"
			elif useCatalogPos!=-1:
				theta_source, phi_source=getstrongestsources(NVSS, N=useCatalogPos)
				theta.append(np.cos(theta_source)) 
				theta_source=[theta_source]
				phi_source=[phi_source]
				print "Use the {} strongest sources from the given Source Catalog".format(useCatalogPos)
			elif useweakestsource!=-1:
				theta_source, phi_source=getweakestsources(NVSS, N=useweakestsource)
				theta.append(np.cos(theta_source))
				theta_source=[theta_source]
				phi_source=[phi_source] 
				print "Use the {} strongest sources from the given Source Catalog".format(useweakestsource)	
			elif GALACTIC_PLANE:
				positions=np.array(sample_in_galactic_plane(N_SOURCES[0]))
				phi_source.append(np.radians(np.remainder(np.concatenate(positions[:,0:1]),360)))
				theta_source=[np.radians(np.concatenate(positions[:,1:2]+90))]
				theta=np.cos(theta_source)
			else:
				theta.append(np.random.uniform(-1.0, np.cos(np.pi/2-np.radians(6)), N_SOURCES[0]))  ##icecube coordinates
				phi_source.append(np.random.uniform(0., 2*np.pi, N_SOURCES[0]))
				theta_source=np.arccos(np.array(theta))
		else:
			print "Be careful: Multi-Mu Skymaps are going to be created."
			for n in N_SOURCES:
				theta_temp=np.random.uniform(-1.0, np.cos(np.pi/2-np.radians(6)), n)
				theta.append(theta_temp)
				phi_source.append(np.random.uniform(0., 2*np.pi, n))
				theta_source.append(np.arccos(theta_temp))
		
	print "Theta Positions of Simulated Events (deg):"
	print np.degrees(theta_source)
	print "Theta Positions of Simulated Events (deg):"
	print theta_source
	print "Phi Positions of Simulated Events (deg):"
	print np.degrees(phi_source)
		
	if SIGNAL == "VarSig":		
		muSim=[]
		allyear_MU_SOURCES=[]
		for mu_i in range(len(MU_SOURCES)):              
			x=calc_posmu_from_Aeff_ratio(theta[mu_i], theta_source[mu_i], N_SOURCES[mu_i], MU_SOURCES[mu_i], NSIG_EVENTS, samples,  DETECTOR, GAMMA, prePath)         
			muSim.append(x[0])
			allyear_MU_SOURCES.append(x[1])
		print muSim
		#print allyear_MU_SOURCES
		
	#PREFIX = str(PROJECT_ID)+"_"+str(int(ADD_TO_RUN_ID)+int(i))+"_"
	PREFIX = str(int(ADD_TO_RUN_ID)+int(i))
	print "\n Current ID prefix: "+PREFIX
	atmGPU=[]
			
	for ob_id, j in enumerate(DETECTOR):
		print "\n"+ "############################################################" + "\n"
		print "Class Object is going to be created for " + j +"\n"
		print "------------------Current RAM Usage-----------------------------" 
		print str(memory_usage_psutil())+ "MB"
		print "----------------------------------------------------------------"
		print "\n"+ "############################################################" + "\n"
		
		mc_folder_path=prePath+j+"/sim/gamma"+str(GAMMA_BF)+"/" ## Path for Point Spread Functions, as well as Energy Distribution and Zenith Distribution Splines // !!!always use BF functions
		
		fileAtmZenith 	= [ mc_folder_path+"BG_zenithDistr_"+j+".txt"]
								
		fileSigZenith		= [ mc_folder_path+"SIG_E-2.07_zenithDistr_"+j+addFile+".txt"]
	
		fileSignalPS 		= [ mc_folder_path+j+"_PointSpread_sig_E-2.07"+addFile+"_histEdges_100bins.txt"]
	
		fileExpData			= [	prePath+j+"/exp/IC40EXP_DECrRArMuEx.txt"] #prePath+"EXP_DECrRArMuEx.txt"
		fileMuExAtm			=  mc_folder_path+j+"_BG_E-2.07_MuEx_recEDistr.txt" #BOTH SIMULATED FOR NOW
		fileMuExSig			=  mc_folder_path+j+"_MC_E-2.07_MuEx_recEDistr.txt"
		
		
		if SIGNAL == "PureAtm":
			atmGPU.append(multiPoleAnalysis(NUMBER_EV_PER_SAMPLE, "", SAVE_DIR, detector=j, nNEvents=NUMBER_EV_PER_SAMPLE[ob_id], nlmax=l_max, nPSSmearingPath=fileGPUSmearingPS,nNRanRuns=NUMBER_RAN_RUNS))
			atmGPU[ob_id].setAtmZenithFile(fileAtmZenith,NUMBER_EV_PER_SAMPLE[ob_id])
			atmGPU[ob_id].setSigZenith(fileSigZenith)
		elif SIGNAL == "PureSig":	
			atmGPU.append(sigAnalysis(NUMBER_EV_PER_SAMPLE, "", SAVE_DIR, detector=j, nNEvents=NUMBER_EV_PER_SAMPLE[ob_id],  nlmax=l_max, nNSources="dep.", nMuSources=MU_SOURCES[0], nVarNSources=0, nPSSmearingPath=fileGPUSmearingPS, nNRanRuns=NUMBER_RAN_RUNS))
			atmGPU[ob_id].setAtmZenithFile(fileAtmZenith,NUMBER_EV_PER_SAMPLE[ob_id])
			atmGPU[ob_id].setsigZenith(fileSigZenith)
			atmGPU[ob_id].setCurNSources(N_SOURCES[0]) 			
			atmGPU[ob_id].setCurMuSources(MU_SOURCES[0])
			atmGPU[ob_id].getPSSpline(fileSignalPS)
			atmGPU[ob_id].setPSStretching(PS_STRETCHING)
			#atmGPU[ob_id].setGalacticPlane(GALACTIC_PLANE, fileGalacticPlane)###
		elif SIGNAL == "VarSig":
			atmGPU.append(sigAnalysis(NUMBER_EV_PER_SAMPLE, "", SAVE_DIR, detector=j, nNEvents=NUMBER_EV_PER_SAMPLE[ob_id], nlmax=l_max, nNSources=N_SOURCES, nMuSources=MU_SOURCES, nVarNSources=0, nPSSmearingPath=fileGPUSmearingPS, nNRanRuns=NUMBER_RAN_RUNS))
			atmGPU[ob_id].setAtmZenithFile(fileAtmZenith,NUMBER_EV_PER_SAMPLE[ob_id])
			atmGPU[ob_id].setsigZenith(fileSigZenith)
			atmGPU[ob_id].getPSSpline(fileSignalPS)
			atmGPU[ob_id].extended_source_angle=extended_source_angle
			atmGPU[ob_id].setmusim([mu[j] for mu in muSim])
			atmGPU[ob_id].setCurNSources(N_SOURCES)
			atmGPU[ob_id].setCurMuSources([x[ob_id] for x in allyear_MU_SOURCES])
			atmGPU[ob_id].setPSStretching(PS_STRETCHING)
			#atmGPU[ob_id].setGalacticPlane(GALACTIC_PLANE, fileGalacticPlane)###
			atmGPU[ob_id].setMultiMuSources(MULTI_MU_SOURCES, MULTI_MU_EXPANSION[0], MULTI_MU_EXPANSION[1])
		elif SIGNAL == "VarAtm":
			atmGPU.append(varZenithAnalysis(NUMBER_EV_PER_SAMPLE,"", SAVE_DIR, nNEvents=NUMBER_EV_PER_SAMPLE[ob_id], nlmax=l_max,nNZenPercMods=1, nZenTotMod=CUR_PERC, nPSSmearingPath=fileGPUSmearingPS, nNRanRuns=NUMBER_RAN_RUNS,detector=j))
			atmGPU[ob_id].setAtmZenithFile(fileAtmZenith,NUMBER_EV_PER_SAMPLE[ob_id])
			atmGPU[ob_id].setNewPerc(CUR_PERC)
			atmGPU[ob_id].genNewSpec()
		print "Current run option: "+str(SIGNAL)
			
		##### GPU SETTINGS
		atmGPU[ob_id].setCenterHits(HIT_CENTERING)
		atmGPU[ob_id].setRenormalization(RENORMALIZATION)
		atmGPU[ob_id].setUseAlm(USE_ALM)
		atmGPU[ob_id].setNside(nside)
		atmGPU[ob_id].setSmearingMethod(SMEARING_METHOD)
		atmGPU[ob_id].setFullSphere(FULL_SPHERE_MODE)
		atmGPU[ob_id].setUseClLog(USE_CL_LOG)
		atmGPU[ob_id].setMilagroAzimuth(MILAGRO_ANISO, fileMilagro)
		atmGPU[ob_id].setFixedZenith(FIX_ZENITH, pathName=fileFixZenith)
		atmGPU[ob_id].setRAAcc(RA_ACC, pathName=fileRAAcceptance)
		atmGPU[ob_id].setGAMMA(GAMMA)
		atmGPU[ob_id].setGalPlaneSwitch(USE_GALACTIC_PLANE)
	####ADDITIONAL DIFFUSE SETTINGS ######
		atmGPU[ob_id].setUseDiffBG(USE_DIFF_BG)
	
	##### SET ENERGY SPECTRUM
		atmGPU[ob_id].setAtmEnergyDistr(fileMuExAtm) #ENERGY dependencies
		atmGPU[ob_id].setSignalEnergyDistr(fileMuExSig)
		atmGPU[ob_id].setUseE(USE_ENERGY)
		atmGPU[ob_id].setNumSigEvents(NSIG_EVENTS[ob_id]) #Number of Signal Events for Energy weighting LJS<--LOL...self named function
		atmGPU[ob_id].setColor(['#33cc99', '#3399cc','#9966cc']) #Colors for Energy Distr. Plots
		
	####	ADDITIONAL STUFF ####	
		atmGPU[ob_id].setAllData(ALL_DATA)
		atmGPU[ob_id].chooseZenBand(ZEN_INT)
		atmGPU[ob_id].setSpecM(SPEC_M)
		
		
	#### NEW MC CHOICE STUFF ####
	
		if MC_GEN == True and not mcEXP==True and not EXPERIMENT:
			weight_astro = samples[j][0]["astro"]
			weight_conv  = samples[j][0]["conv"]
			atmGPU[ob_id].setMCSample(samples[j][0], weight_astro, weight_conv, MC_GEN, gamma=GAMMA)
		
	##### PARAMETERS FOR NEW SKYMAP SIMULATION #####
		#~ if SIGNAL == "VarSig":
			#~ atmGPU[ob_id].calcMuReal(MU_SOURCES, "realMuTables_1samples_E"+str(GAMMA)+"_hemisphere.dict")
			#~ atmGPU[ob_id].setNMax()
	
		if SIGNAL == "PureSig" or SIGNAL == "VarSig":
			atmGPU[ob_id].setMuPreciseBool(MU_PRECISE)
			atmGPU[ob_id].setAcceptanceSpline()
			atmGPU[ob_id].setPSReplaceMode(REP)
			atmGPU[ob_id].setRotationMode(PS_ROT)
	
		
		#~ if not VAR_ZENITH == 0.0:
			#~ atmGPU[ob_id].setNewPerc(VAR_ZENITH)
			#~ atmGPU[ob_id].genNewSpec()
			#~ print "... generated new bg spectrum."
		
		if EXPERIMENT:
			print "Note: Creating EXPERIMENTAL maps!"
			atmGPU[ob_id].ranTheta	= dec2zen(samples[j][1]["dec"])
			print "Length: "+str(len(atmGPU[ob_id].ranTheta))
			if SHUFFLE:
				atmGPU[ob_id].ranPhi = np.random.uniform(0., 2*np.pi, len(atmGPU[ob_id].ranTheta))
				print "NOTE: SHUFFLED EXPERIMENTAL AZIMUTH VALUES!!!"
			else:
				atmGPU[ob_id].ranPhi	= samples[j][1]["ra"]
			atmGPU[ob_id].energyList=samples[j][1]["logE"]
			
			atmGPU[ob_id].hitBool2Map(zen2dec(atmGPU[ob_id].ranTheta), atmGPU[ob_id].ranPhi, atmGPU[ob_id].energyList)
			atmGPU[ob_id].analyseMap()
			
		elif mcEXP:
			#read theta 
			atmGPU[ob_id].ranTheta = np.loadtxt(thispath + "mcBG_for_reshuffling_"+str(MC_SET)+"_smearedPureAtm_Ev35557_allTheta.txt")
			#read energy 
			atmGPU[ob_id].energyList = np.loadtxt(thispath + "mcBG_for_reshuffling_"+str(MC_SET)+"_smearedPureAtm_Ev35557_allEnergy.txt")
			atmGPU[ob_id].ranPhi = np.random.uniform(0., 2*np.pi, len(atmGPU[ob_id].ranTheta))
			atmGPU[ob_id].hitBool2Map(zen2dec(atmGPU[ob_id].ranTheta), atmGPU[ob_id].ranPhi, atmGPU[ob_id].energyList)
			atmGPU[ob_id].analyseMap()
		
		else:		# here the map_delta's are created (I think so at least lol)
			##### RUN GPU CALCULATIONS
			if SIGNAL == "PureAtm":
				atmGPU[ob_id].createRanSkymaps()
			elif SIGNAL == "PureSig":
				# and USE_DIFF_BG:					
				atmGPU[ob_id].setSourcePos(theta[0], theta_source[0], phi_source[0])
				atmGPU[ob_id].createSigBGSkymaps(useNsou=USE_N, shuff=SHUFFLE, analyse=False)			
			#~ elif SIGNAL == "PureSig" and not USE_DIFF_BG: ########ACHTUNG#######
				#~ print "***** Warning!! Maybe using wrong Skymap Creation!! *****"
				#~ atmGPU[ob_id].createPureSigSkymaps()
			elif SIGNAL == "VarSig":
				atmGPU[ob_id].setSourcePos(theta, theta_source, phi_source)
				atmGPU[ob_id].createRanSigSkymaps()
			#~ elif SIGNAL == "VarAtm":
				#~ atmGPU[ob_id].createRanSkymaps()
				
		print "------------------Current RAM Usage-----------------------------" 
		print str(memory_usage_psutil()) + "MB"
		print "----------------------------------------------------------------"
		
		if not VAR_ZENITH == 0.0:
			atmGPU[ob_id].reformSpec()
	
		##### SAVING RESULTS
		if EXPERIMENT:
			#saveTitle = SAVE_DIR+"data/"+PREFIX+"smearedExperiment_Ev"+str(atmGPU[ob_id].NUMBER_SIMULATED_EVENTS)
			saveTitle = SAVE_DIR
		else:
			#saveTitle = SAVE_DIR+"data/"+PREFIX+"smeared"+SIGNAL+"_Ev"+str(atmGPU[ob_id].NUMBER_SIMULATED_EVENTS)
			saveTitle = SAVE_DIR
	
		if SAVE_MAPS == True:  ###Better be False ;-)
			atmGPU[ob_id].saveLastSkymap(saveTitle+"_"+SMEARING_METHOD)
			
		if SAVE_ENERGY:
			if "Rene" not in PROJECT_ID:
				atmGPU[ob_id].saveEventEnergy(saveTitle,PREFIX)
			else:
				print "Save Full Energy Event List for Rene (not histogrammed)"
				atmGPU[ob_id].nTitlesaveEventEnergy(saveTitle, histogrammsave=False)
		
		if SAVE_MU and not SIGNAL == "PureAtm":
			atmGPU[ob_id].saveMuInfo(saveTitle,PREFIX)
			
		if SAVE_ANG:
			if "Rene" not in PROJECT_ID:
				atmGPU[ob_id].saveAngular(saveTitle,PREFIX)
				if not SIGNAL == "PureAtm":
					atmGPU[ob_id].saveSouAngular(saveTitle,PREFIX)
			else:
				print "Save Full Angular Event List for Rene (not histogrammed)"
				atmGPU[ob_id].saveAngular(saveTitle,PREFIX,histogrammsave=False)				
		
		############################End of Loop############################################
		
	if EXPERIMENT:
		#saveTitle = SAVE_DIR+"data/"+PREFIX+"smearedExperiment_Ev"+str(NUMBER_SIMULATED_EVENTS)
		saveTitle = SAVE_DIR
	else:
		#saveTitle = SAVE_DIR+"data/"+PREFIX+"smeared"+SIGNAL+"_Ev"+str(NUMBER_SIMULATED_EVENTS)
		saveTitle = SAVE_DIR
	
	print "Merge Maps..."	
	if SIGNAL == "PureSig" or SIGNAL == "VarSig":
		theta_source_commulated=[]
		phi_source_commulated=[]
		for i in range(len(theta_source)):
			theta_source_commulated.extend(theta_source[i])
			phi_source_commulated.extend(phi_source[i])
		mergeGPU=MergeAnalysis(theta_source_commulated,phi_source_commulated, atmGPU)
	else:
		mergeGPU=MergeAnalysis(skymaps=atmGPU)
	mergeGPU.set_catalog(catalog)
	if catalog!="":
		mergeGPU.set_catalogmap(catalog_map)
	#atmGPU[ob_id].plotFirstAlms(l_max)
	mergeGPU.analyseMap()
	mergeGPU.calcMeans(alm=USE_ALM)
	mergeGPU.saveCleffClList(saveTitle,PREFIX)
	mergeGPU.saveAlms(saveTitle,PREFIX)

	#~ atmGPU[ob_id].saveClLogList(saveTitle)
	#if not EXPERIMENT and (SIGNAL == "PureSig" or SIGNAL == "VarSig"):
	#	atmGPU[ob_id].saveSignalNeutrinos(saveTitle)
	#if USE_ALM and ALL_DATA:
	#	mergeGPU.saveAlms(saveTitle)
	#	print "Alm Length is: "+str(len(atmGPU[ob_id].alm_all))
	#	BLA = copy.copy(atmGPU[ob_id].alm_all)
	
	##### PLOTTING SKYMAPS
	#if SHOW_PLOTS == True and i==0:
	if False:
		plt.figure(1)
		atmGPU[ob_id].plotLastSkymap(title=SIGNAL+", "+SMEARING_METHOD, nFigure=1, logz=True) #Possibility for logz option, only use for visualization! See classes4.py
		atmGPU[ob_id].plotEnergyDistr(2)
		atmGPU[ob_id].plotSingleAlm(0,0)
		plt.show()
		##### SAVING PLOTS 
		#if not EXPERIMENT:
		#	saveAll([1], )
			#saveAll([1], SAVE_DIR+"plots/"+PREFIX+"smeared"+SIGNAL+"_")
		#else: 
		#	saveAll([1], )
			#saveAll([1], SAVE_DIR+"plots/"+PREFIX+"smeared"+SIGNAL+"_")
	else:
		print "i is: "+str(i)
		
	##### REFORM SPECTRUM
	if SIGNAL == "VarAtm":
		atmGPU[ob_id].reformSpec()

#########################
##### FINISHED ALL ######
#####     RUNS     ######
#########################
