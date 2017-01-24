import os

toDo = """
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=False Renorm=True UseGalPlane=False OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=False Renorm=True UseGalPlane=True OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=False Renorm=True UseGalPlane=True OnlyGalPlane=True
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=False OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=True

python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=False Renorm=True UseGalPlane=False OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=False Renorm=True UseGalPlane=True OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=False Renorm=True UseGalPlane=True OnlyGalPlane=True
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=False OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=False
python create_dagman.py "Sig_N1_2010_gamma2_noE" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=True
"""




print toDo
print "start list..."
os.system(toDo)
print "finished list..."
print toDo
print "Finnishing list successfully, without any fatal error found."

