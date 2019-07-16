# triangle
triangle counting in graph using linear algebra operations

### How to build the code
To build for NEC machine:

    ./compile-nec.sh

To build for Intel CPU machine:

    make clean; make triangles

### How to run
    mpirun triangles-nec <input_file.data>
    
A small test dataset has been included together with the source: hep-th_mtx.data

To change the OpenMP threads number, export this before run:

    export OMP_NUM_THREADS=1
 
For both build and run of the code, make sure the installed NEC PATHes are set to $PATH, something like this:

    export PATH=$PATH:/opt/nec/ve/bin/:/opt/nec/ve/mpi/2.1.0/bin/
    
### Where are the testing dataset?
The original dataset were downloaded from The SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection), which is used by the triangle counting paper in our reference. The datasets have been preprocessed and converted to binary format to facilitate the reading from NEC machine.

The datasets are tempararily hosted at:

    http://129.79.186.230/graphdata/triangle
