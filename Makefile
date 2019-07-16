RMATPATH = GTgraph/R-MAT
SPRNPATH = GTgraph/sprng2.0-lite
INCLUDE = -I$(SPRNPATH)/include 
COMPILER = mpicxx -std=c++11
sprng:	
	(cd $(SPRNPATH); $(MAKE); cd ../..)

rmat:	sprng
	(cd $(RMATPATH); $(MAKE); cd ../..)

#TOCOMPILE = $(RMATPATH)/graph.o $(RMATPATH)/utils.o $(RMATPATH)/init.o $(RMATPATH)/globals.o MPIType.o
TOCOMPILE = MPIType.o

MPIType.o:	MPIType.cpp MPIType.h
	$(COMPILER) $(INCLUDE) $(FLAGS) -c -o MPIType.o MPIType.cpp

#triangles: triangles.cpp CSC.h CSC.cpp CSR.h CSR.cpp multiply.h rmat MPIType.o
triangles: triangles.cpp CSC.h CSC.cpp CSR.h CSR.cpp multiply.h MPIType.o
	$(COMPILER) $(FLAGS) $(INCLUDE) -fopenmp -o triangles triangles.cpp ${TOCOMPILE} ${LIBS} 

clean:
	(cd GTgraph; make clean; cd ../..)
	rm -rf *.o triangles
