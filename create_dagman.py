import pydag
import itertools
import sys
import os
from functions4 import *

##########################################-<<-------------------------
Resc=False############## DO NOT FORGET TO SET THIS VARIABLE. IMPORTANT TO MAKE SURE YOU ARE RUNNING IN THE RIGHT MODE<<------------------
##################################<<--------------------------------

gamma=2.0


allParam=getParameters()
print sys.argv
PROJECT_ID		= str(sys.argv[1])	#sets project name, can be anything
SIGNAL 			= str(sys.argv[2])	#PureAtm, PureSig, VarSig ((VarAtm))
NSou=str(sys.argv[4])
if SIGNAL=="PureAtm" or SIGNAL=="PureSig" or NSou=="[0]":
	RUNS_PER_JOB 	= 40	#100
	JOBS 		= 25	#100
elif SIGNAL=="VarSig":
	RUNS_PER_JOB 	= 40	#40
	JOBS 		= 25	#25 for VarSig, 100 for Pure*
PAR1 			= str(sys.argv[3])	#MU: mean number of nu per source
PAR2 			= str(sys.argv[4])	#N_SOURCE, 0 for PureSig
PAR3 			= getParameters(connect=True)

PROCESS_DIR		= "/net/scratch_icecube4/user/kalaczynski/Analysis_stuff/condor/1Y_Skymaps_MCgen_useE_"+allParam["detector"]+"_E"+str(gamma)+"_useDiff"
WORKDIR= PROCESS_DIR+"/jobs/unsmeared"+str(SIGNAL)+"/"
script = "/net/scratch_icecube4/user/kalaczynski/createSmearedSkymap_multipleyears.py"
dagFile = WORKDIR+"job_PROJECT_"+PROJECT_ID+"_"+str(PAR1)+".dag"
submitFile = WORKDIR+"job_PROJECT_"+PROJECT_ID+"_"+str(PAR1)+".sub"

if not Resc==True:
	if os.path.exists(WORKDIR)==False:
		os.makedirs(WORKDIR)
		print "Created New Folder in:  "+ WORKDIR
	
	path=PROCESS_DIR+"/logs/unsmeared"+SIGNAL+"/"+SIGNAL+"_PROJECT_"+PROJECT_ID+"_MU"+str(PAR1)+"/"
	
	if os.path.exists(path)==False:
			os.makedirs(path)
			print "Created New Folder in:  "+ path
	print "Write Dagman Files to: "+submitFile
	
	arguments = " {2} $(ARG3) {4} {3} {6} {7} {8}".format(script, PROCESS_DIR, PROJECT_ID, SIGNAL, RUNS_PER_JOB, JOBS, PAR1, PAR2, PAR3)
	submitFileContent = {"getenv": True,
	                     "universe": "vanilla",
	                     "notification": "Error",
	                     "log": "$(LOGFILE).log",
	                     "output": "$(LOGFILE).out",
	                     "error": "$(LOGFILE).err",
	                     "request_memory": "1.5GB",
	                     "arguments": arguments}
	
	submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
	                                           script,
	                                           **submitFileContent)
	submitFile.dump()
	
	args1=range(JOBS)
	nodes = []
	for i, a1 in enumerate(args1): #for i, a1, a2 in enumerate(itertools.product(args1, args2))
	    logfile = path+"/"+str(i*RUNS_PER_JOB)
	    dagArgs = pydag.dagman.Macros(
	        LOGFILE=logfile,
			ARG3=i*RUNS_PER_JOB)
	    node = pydag.dagman.DAGManNode(i, submitFile)
	    node.keywords["VARS"] = dagArgs
	    nodes.append(node)
	
	dag = pydag.dagman.DAGManJob(dagFile, nodes)
	dag.dump()

os.system("condor_submit_dag -f "+dagFile)
time.sleep(1)
