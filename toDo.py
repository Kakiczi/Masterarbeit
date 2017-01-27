import os

						#Remember to give DIFFERENT job names!
						#Aaand putting empty space in the name is not such a brilliant idea ...
toDo = """
python create_dagman.py "BGD1" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=True UseGalPlane=True OnlyGalPlane=False
python create_dagman.py "BGD2" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=True UseGalPlane=True OnlyGalPlane=True
python create_dagman.py "BGD3" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=False OnlyGalPlane=False
python create_dagman.py "SIG1" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=True UseGalPlane=True OnlyGalPlane=False
python create_dagman.py "SIG2" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=True UseGalPlane=True OnlyGalPlane=True
python create_dagman.py "SIG3" "VarSig" [30] [5] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=False OnlyGalPlane=False
"""
#python create_dagman.py "BGD4" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=False
#python create_dagman.py "BGD5" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=True
#python create_dagman.py "BGD6" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useAlm=True Renorm=False UseGalPlane=False OnlyGalPlane=False
#python create_dagman.py "SIG4" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=False
#python create_dagman.py "SIG5" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=True OnlyGalPlane=True
#python create_dagman.py "SIG6" "VarSig" [30] [0] detector="1111" gamma=2.13 catalog="NVSS" useCatalogPos=5 useAlm=True Renorm=False UseGalPlane=False OnlyGalPlane=False
#useCatalogPos
print toDo
print "start list..."
os.system(toDo)
print "finished list..."
print toDo
print "Finnishing list successfully, without any fatal error found."

