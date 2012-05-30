# ROOTLIBS := $(shell root-config --libs)
# ROOTCFLAGS := $(shell root-config --cflags)
# ROOTLIBS += lMinuit # Not included by default
ROOTLIBS := -I/home/herten/fairsoft/extpkg/tools/root/include -L/home/herten/fairsoft/extpkg/tools/root/lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lMinuit -lThread -lm -ldl

all:	lab6b

lab6b:	lab6b.cu 
		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $< 

