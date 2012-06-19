# # ROOTLIBS := $(shell root-config --libs)
# # ROOTCFLAGS := $(shell root-config --cflags)
# # ROOTLIBS += lMinuit # Not included by default
# CFLAGS = $(shell root-config --cflags)
# LIBS   = $(shell root-config --libs)
# GLIBS  = $(shell root-config --glibs)
# GLIBS  += -lMinuit

PATHTOROOT = /home/herten/fairsoft/extpkg/tools/root

ROOTLIBS := -m64 -I$(PATHTOROOT)/include -L$(PATHTOROOT)/lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -L$(PATHTOROOT)/lib -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -lMinuit

all:	lab8a test lab8b

lab8a:	lab8a.cu
		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $< 
# 		nvcc -arch=sm_20 --compiler-options "$(CFLAGS)" $(LIBS) --compiler-options "$(GLIBS)" -o $@ $< 
test: lab8btest.cu
		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $<

lab8b: lab8b.cu
		nvcc -arch=sm_20 $(ROOTLIBS) -o $@ $<