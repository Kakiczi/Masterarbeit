import os

toDo = """
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [0.15] [0] detector="1111" gamma=2.0 useAlm=True
"""



print toDo
print "start list..."
os.system(toDo)
print "finished list..."
print toDo
print "Finnishing list successfully, without any fatal error found."

