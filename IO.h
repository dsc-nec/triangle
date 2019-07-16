#ifndef _IO_SPGEMM_H
#define _IO_SPGEMM_H

#include "Triple.h"
#include "CSC.h"
#include "MPIType.h"

#define READBUFFER (512 * 1024 * 1024)  // in MB

template <typename IT, typename NT>
int ReadBinary(string filename, CSC<IT,NT> * & csc)
{
    FILE * f = fopen(filename.c_str(), "r");
    if(!f)
    {
        cerr << "Problem reading binary input file" << filename << endl;
        return -1;
    }
    IT m,n,nnz;
    fread(&m, sizeof(IT), 1, f);
    fread(&n, sizeof(IT), 1, f);
    fread(&nnz, sizeof(IT), 1, f);
    
    if (m <= 0 || n <= 0 || nnz <= 0)
    {
        cerr << "Problem with matrix size in binary input file" << filename << endl;
        return -1;
    }
    double start = omp_get_wtime( );
    cout << "Reading matrix with dimensions: "<< m << "-by-" << n <<" having "<< nnz << " nonzeros" << endl;
    
    IT * rowindices = new IT[nnz];
    IT * colindices = new IT[nnz];
    NT * vals = new NT[nnz];
    
    size_t rows = fread(rowindices, sizeof(IT), nnz, f);
    size_t cols = fread(colindices, sizeof(IT), nnz, f);
    size_t nums = fread(vals, sizeof(NT), nnz, f);
    
    if(rows != nnz || cols != nnz || nums != nnz)
    {
        cerr << "Problem with FREAD, aborting... " << endl;
        return -1;
    }
    double end = omp_get_wtime( );
    printf("File IO time: = %.16g seconds\n", end - start);

    fclose(f);
    
    csc = new CSC<IT,NT>(rowindices, colindices, vals , nnz, m, n);
    
    delete [] rowindices;
    delete [] colindices;
    delete [] vals;
    return 1;
}

template <typename IT, typename NT>
int ReadASCII(string filename, CSC<IT,NT> * & csc)
{
    double start = omp_get_wtime( );
    ifstream infile(filename.c_str());
    char line[256];
    char c = infile.get();
    while(c == '%')
    {
        infile.getline(line,256);
        c = infile.get();
    }
    infile.unget();
    IT m,n,nnz;
    infile >> m >> n >> nnz;	// #{rows}-#{cols}-#{nonzeros}
    //cout << m << " " << n << " " << nnz << endl;
    
    Triple<IT,NT> * triples = new Triple<IT,NT>[nnz];
    if (infile.is_open())
    {
        IT cnz = 0;	// current number of nonzeros
        while (! infile.eof() && cnz < nnz)
        {
            infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;	// row-col-value
            triples[cnz].row--;
            triples[cnz].col--;
            ++cnz;
        }
        assert(cnz == nnz);
    }
    
    double end = omp_get_wtime( );
    printf("File IO time: = %.16g seconds\n", end - start);
    cout << "converting to csc ... " << endl;
    csc= new CSC<IT,NT>(triples, nnz, m, n);
    csc->totalcols = n;
    delete [] triples;
    return 1;
}


template <typename IT, typename NT>
int ReadASCIIDist(string filename, CSC<IT,NT> * & csc, bool removeselfloops = false)
{
    double start = omp_get_wtime( );
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    MPI_Datatype triptype;
    MPI_Type_contiguous(sizeof(Triple<IT,NT>), MPI_CHAR, &triptype );
    MPI_Type_commit(&triptype);
    vector< Triple<IT,NT> > loctrips;
    IT m,n,nnz;
    
    if(myrank == 0)
    {
        ifstream infile(filename.c_str());
        char line[256];
        char c = infile.get();
        while(c == '%')
        {
            infile.getline(line,256);
            c = infile.get();
        }
        infile.unget();
        infile >> m >> n >> nnz;	// #{rows}-#{cols}-#{nonzeros}
        //cout << m << " " << n << " " << nnz << endl;
    
        size_t max2read = std::min((size_t) nnz, READBUFFER / sizeof(Triple<IT,NT>));
        int * counts = new int[nprocs]();
        int * displs = new int[nprocs+1]();
        IT perproc = n / nprocs;
        
        IT temprow, tempcol;
        NT tempnum;
	IT skipped = 0;

        while (infile.is_open())
        {

       	    Triple<IT,NT> * triples = new Triple<IT,NT>[max2read];
            Triple<IT,NT> * trip2go = new Triple<IT,NT>[max2read];
            IT cnz = 0;	// current number of nonzeros
            while ((!infile.eof()) && cnz < max2read)
            {
                infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;	// row-col-value
		if((triples[cnz].row == triples[cnz].col) && removeselfloops)
		{
			skipped += 1;
			continue;
		}
		
                triples[cnz].row--;
                triples[cnz].col--;

		size_t cindex = std::min(triples[cnz].col / perproc, nprocs-1);	
		(counts[cindex])++;	// if you don't get the index to a temporary, it will cursh
                ++cnz;
            }
            std::partial_sum(counts, counts+nprocs, displs+1);  // like CSR/CSC ptrs array
	    //std::copy(counts, counts+nprocs, ostream_iterator<int>(cout, " ")); cout << endl;
            for(size_t i=0; i< cnz; ++i)    // reshuffle to destinations
            {
		size_t dindex = std::min(triples[i].col / perproc, nprocs-1);
                int tindex = displs[dindex]++;   // post increment
                trip2go[tindex] = triples[i];
            }

	    //std::copy(trip2go, trip2go+cnz, ostream_iterator<Triple<IT,NT>>(cout, " ")); cout << endl;
	    delete [] triples;
	    displs[0] = 0;
            std::partial_sum(counts, counts+nprocs, displs+1);  // revalidate
            
            int myshare;
            // int MPI_Scatter(void *sendbuf,int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt,
            //                MPI_Datatype recvtype, int root, MPI_Comm comm);
            MPI_Scatter(counts, 1, MPI_INT, &myshare, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            Triple<IT,NT> * mynonzeros = new Triple<IT,NT>[myshare];
            // int MPI_Scatterv(void *sendbuf,int *sendcnts,int *displs,MPI_Datatype sendtype,
            //                 void *recvbuf,int recvcnt,MPI_Datatype recvtype,int root,MPI_Comm comm);
            MPI_Scatterv(trip2go, counts, displs, triptype, mynonzeros, myshare, triptype, 0, MPI_COMM_WORLD);
	    delete [] trip2go;
            std::copy (mynonzeros, mynonzeros+myshare,back_inserter(loctrips));
            delete [] mynonzeros;
            if (infile.eof() || (cnz+skipped) == nnz)	
            {
                std::fill(counts, counts+nprocs, -1);
                MPI_Scatter(counts, 1, MPI_INT, &myshare, 1, MPI_INT, 0, MPI_COMM_WORLD);
		delete [] counts;
		delete [] displs;
                break;
            }
	    else
	    {
		std::fill(counts, counts+nprocs, 0);
		std::fill(displs, displs+nprocs+1, 0);
	    }
        }
	if(skipped > 0)
	{
		cout << skipped << " self loops are removed" << endl;
		nnz -= skipped;	// update
	}
    }
    else    // receive the triples
    {
        int myshare = 0;
        MPI_Scatter(NULL, 0, MPI_INT, &myshare, 1, MPI_INT, 0, MPI_COMM_WORLD);
        while(myshare >=0)
        {
            Triple<IT,NT> * mynonzeros = new Triple<IT,NT>[myshare];
            MPI_Scatterv(NULL, NULL, NULL, triptype, mynonzeros, myshare, triptype, 0, MPI_COMM_WORLD);
            std::copy (mynonzeros, mynonzeros+myshare,back_inserter(loctrips));
            delete [] mynonzeros;
            MPI_Scatter(NULL, 0, MPI_INT, &myshare, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast(&m, 1, MPIType<IT>(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPIType<IT>(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPIType<IT>(), 0, MPI_COMM_WORLD);
    
    IT mycol;
    if(myrank != nprocs-1)  mycol = n/nprocs;
    else    mycol = n- (n/nprocs)*myrank;
    IT mycolstart = (n/nprocs) * myrank;

    std::for_each(loctrips.begin(), loctrips.end(), [&mycolstart](Triple<IT,NT> & trip) { trip.col -= mycolstart; });
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(myrank == 0)
    {
    	double end = omp_get_wtime( );
        printf("File IO time: = %.16g seconds\n", end - start);
    	cout << "converting to csc ... " << endl;
    }
    csc= new CSC<IT,NT>(loctrips.data(), loctrips.size(), m, mycol);
    csc->totalcols = n;
    return 1;
}


#endif
