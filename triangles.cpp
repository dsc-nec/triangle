#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include "utility.h"
#include "CSC.h"
#include "CSR.h"
#include "IO.h"
#include "multiply.h"
#include "bloom_filter.hpp"
using namespace std;


extern "C" {
//#include <mkl_spblas.h>
#include "GTgraph/R-MAT/defs.h"
#include "GTgraph/R-MAT/init.h"
#include "GTgraph/R-MAT/graph.h"
}

#define VALUETYPE int
#define INDEXTYPE int
//#define TRIDEBUG
#define ITERS 10

enum generator_type
{
	rmat_graph,
	er_graph,
};

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}



// create bloom filter for rows with at least one nonzero
template <class IT, class NT>
bloom_filter rowFilter(const CSC<IT,NT>& A)
{
    unsigned int random_seed = 0xA57EC3B2;
    const double desired_probability_of_false_positive = 0.05;
    
    bloom_parameters parameters;
    parameters.projected_element_count    = A.rows;
    parameters.false_positive_probability = desired_probability_of_false_positive;
    parameters.random_seed                = random_seed++;
    parameters.maximum_number_of_hashes   = 7;
    if (!parameters) 
    {
        std::cout << "Error - Invalid set of bloom filter parameters!\n";
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    parameters.compute_optimal_parameters();
    bloom_filter bm(parameters);
    
    
    // first, count nnz in the result matrix
    for(int i=0; i<A.cols; i++)
    {
        IT endIdx = A.colptr[i+1];
        for(IT j=A.colptr[i]; j<endIdx; j++)
        {
            bm.insert(A.rowids[j]);
        }
    }
    
    return bm;
}


template <typename IT, typename NT>
bloom_filter SPARowIds_bloom(const CSC<IT,NT> & A)
{
    IT uniqri = 0;
    vector<IT> ri;
    vector<bool> spabools(A.rows, false); // default false
    for(int i=0; i < A.cols; ++i)
    {
        for(IT j=A.colptr[i]; j < A.colptr[i+1]; ++j)
        {
            if(!spabools[A.rowids[j]])
            {
                uniqri++;
                spabools[A.rowids[j]] = true;
                ri.push_back(A.rowids[j]);
            }
            
        }
    }
    
    unsigned int random_seed = 0xA57EC3B2;
    const double desired_probability_of_false_positive = 0.05;
    
    bloom_parameters parameters;
    parameters.projected_element_count    = uniqri;
    parameters.false_positive_probability = desired_probability_of_false_positive;
    parameters.random_seed                = random_seed++;
    parameters.maximum_number_of_hashes   = 7;
    if (!parameters)
    {
        std::cout << "Error - Invalid set of bloom filter parameters!\n";
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    parameters.compute_optimal_parameters();
    bloom_filter bm(parameters);
    
    
    for(auto itr=ri.begin(); itr != ri.end(); ++itr)
    {
        bm.insert(*itr);
    }
    
    return bm;
}



template <typename IT, typename NT>
vector<IT> SPARowIds(const CSC<IT,NT> & A)
{

    vector<IT> ri;
    vector<bool> spabools(A.rows, false);
    for(int i=0; i < A.cols; ++i)
    {
        for(IT j=A.colptr[i]; j < A.colptr[i+1]; ++j)
        {
            if(!spabools[A.rowids[j]])
            {
                spabools[A.rowids[j]] = true;
                ri.push_back(A.rowids[j]);
            }
        }
    }
    
    std::sort(ri.begin(), ri.end(), std::less<IT>());
    return ri;
}


// identify the row indices of A with at least one nonzero
vector<INDEXTYPE> RowIds(const CSC<INDEXTYPE,VALUETYPE> & A)
{
    INDEXTYPE hsize = A.cols;
    HeapEntry<INDEXTYPE,VALUETYPE> * mergeheap = new HeapEntry<INDEXTYPE,VALUETYPE>[hsize];
    vector<INDEXTYPE> RowIds;
    

    int k = 0;
    for(int i=0; i < A.cols; ++i)        // for all columns of A
    {
        if((A.colptr[i+1] - A.colptr[i]) > 0)
        {
            mergeheap[k].loc = 1;
            mergeheap[k].runr = i; //column
            mergeheap[k++].key = A.rowids[A.colptr[i]];	// A's first rowid is the first key
        }
    }
    
    hsize = k;
    make_heap(mergeheap, mergeheap + hsize);
    
    while(hsize > 0)
    {
        pop_heap(mergeheap, mergeheap + hsize);         // result is stored in mergeheap[hsize-1]
        HeapEntry<INDEXTYPE,VALUETYPE> hentry = mergeheap[hsize-1];
        
        if( RowIds.empty() || RowIds.back() != hentry.key)
        {
            RowIds.push_back( hentry.key );
        }

        INDEXTYPE index = A.colptr[hentry.runr] + hentry.loc;
        // If still unused nonzeros exists in A(:,i), insert the next nonzero to the heap
        if( index < A.colptr[hentry.runr+1] )
        {
            mergeheap[hsize-1].loc	 = hentry.loc +1;
            mergeheap[hsize-1].key	 = A.rowids[index];
            mergeheap[hsize-1].runr = hentry.runr;
            push_heap(mergeheap, mergeheap + hsize);
        }
        else
        {
            --hsize;
        }
    }
    
    delete [] mergeheap;
    return RowIds;
    
}


template <class IT, class NT>
void SpRef (const CSC<IT,NT>& A, const IT* ci, int cilen, const bloom_filter& rbm, CSC<IT,NT>* & B)
{
    if( (cilen>0) && (ci[cilen-1] > A.cols))
    {
        cerr << "Col indices out of bounds" << endl;
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    
    IT bnnz = 0;
    for(int i=0; i<cilen; i++)
    {
        IT endIdx = A.colptr[ci[i]+1];
        for(IT j=A.colptr[ci[i]]; j<endIdx; j++)
        {
            if(rbm.contains(A.rowids[j]))
                bnnz ++;
        }
    }
    
    // can nrows be predicted from bloom_filter ?
    // receiver does not care about nrows
    // also rowids of B remains in rowids from original matrix
    B = new CSC<IT,NT>;
    B->nnz = bnnz;
    B->rows = rbm.element_count();
    B->cols = cilen;
    B->colptr = new IT[B->cols+1];
    B->rowids = new IT[B->nnz];
    B->values = new NT[B->nnz];
    B->colptr[0] = 0;
    IT idx=0;
    
    for(int i=0; i<cilen; i++)
    {
        IT endIdx = A.colptr[ci[i]+1];
        for(IT j=A.colptr[ci[i]]; j<endIdx; j++)
        {
            if(rbm.contains(A.rowids[j]))
            {
                B->values[idx] = A.values[j];
                B->rowids[idx++] = A.rowids[j];
            }
        }
        B->colptr[i+1] = idx;
    }
}


// if allrows==true then use all rows and ignore ri
template <class IT, class NT>
void SpRef_basic (const CSC<IT,NT>& A, const IT* ci, int cilen, const IT* ri, int rilen, bool allrows, CSC<IT,NT>* & B)
{
    if( (cilen>0) && (ci[cilen-1] > A.cols))
    {
        cerr << "Col indices out of bounds" << endl;
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    if( (rilen>0) && (ri[rilen-1] > A.rows))
    {
        cerr << "Row indices out of bounds" << endl;
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    
    IT bnnz = 0;
    for(int i=0; i<cilen; i++)
    {
        IT startIdx = A.colptr[ci[i]];
        IT endIdx = A.colptr[ci[i]+1];
        if(allrows)
            bnnz += endIdx - startIdx;
        else
        {
            IT j=startIdx, k=0;
            while(j<endIdx && k < rilen)
            {
                if(ri[k]<A.rowids[j]) k++;
                else if(ri[k]>A.rowids[j]) j++;
                else //(ri[k]==rowids[j])
                {
                    bnnz++;
                    k++;
                    j++;
                }
            }
        }
    }
    
    // can nrows be predicted from bloom_filter ?
    // receiver does not care about nrows
    // also rowids of B remains in rowids from original matrix
    B = new CSC<IT,NT>;
    B->nnz = bnnz;
    B->rows = A.rows;  // not like actual SpRef
    B->cols = cilen;
    B->colptr = new IT[B->cols+1];
    B->rowids = new IT[B->nnz];
    B->values = new NT[B->nnz];
    B->colptr[0] = 0;
    IT idx=0;
    
    
    for(int i=0; i<cilen; i++)
    {
        IT startIdx = A.colptr[ci[i]];
        IT endIdx = A.colptr[ci[i]+1];
        if(allrows)
        {
            B->colptr[i+1] = B->colptr[i] + endIdx - startIdx;
            copy(A.rowids+startIdx, A.rowids+endIdx, B->rowids+B->colptr[i]);
            copy(A.values+startIdx, A.values+endIdx, B->values+B->colptr[i]);
        }
        else
        {
            IT j=startIdx, k=0;
            while(j<endIdx && k < rilen)
            {
                if(ri[k]<A.rowids[j]) k++;
                else if(ri[k]>A.rowids[j]) j++;
                else //(ri[k]==rowids[j])
                {
                    B->values[idx] = A.values[j];
                    B->rowids[idx++] = A.rowids[j];
                    k++;
                    j++;
                }
            }
            B->colptr[i+1] = idx;
        }
    }
}


// finish and run
// if allrows==true then use all rows and ignore ri
template <class IT, class NT>
void SpRef_finger (const CSC<IT,NT>& A, const IT* ci, int cilen, IT* ri, int rilen, bool allrows, CSC<IT,NT>* & B)
{
    
    if( (cilen>0) && (ci[cilen-1] > A.cols))
    {
        cerr << "Col indices out of bounds" << endl;
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    if( (rilen>0) && (ri[rilen-1] > A.rows))
    {
        cerr << "Row indices out of bounds" << endl;
        MPI_Abort(MPI_COMM_WORLD, 6666);
    }
    
    IT bnnz = 0;
    for(int i=0; i<cilen; i++)
    {
        IT startIdx = A.colptr[ci[i]];
        IT endIdx = A.colptr[ci[i]+1];
        if(allrows)
            bnnz += endIdx - startIdx;
        else if(rilen > 0)
        {
            IT* left=ri;
            for(IT j=startIdx; j<endIdx ; j++)
            {
                if(left < (ri+rilen))
                {
                    left = std::lower_bound(left, ri+rilen, A.rowids[j]);
                    if(A.rowids[j] == *left)
                    {
                        bnnz++;
                    }
                }
            }
        }
    }
    
    // can nrows be predicted from bloom_filter ?
    // receiver does not care about nrows
    // also rowids of B remains in rowids from original matrix
    B = new CSC<IT,NT>;
    B->nnz = bnnz;
    B->rows = A.rows;  // not like actual SpRef
    B->cols = cilen;
    B->colptr = new IT[B->cols+1];
    B->rowids = new IT[B->nnz];
    B->values = new NT[B->nnz];
    B->colptr[0] = 0;
    IT idx=0;
    
    
    for(int i=0; i<cilen; i++)
    {
        IT startIdx = A.colptr[ci[i]];
        IT endIdx = A.colptr[ci[i]+1];
        if(allrows)
        {
            B->colptr[i+1] = B->colptr[i] + endIdx - startIdx;
            copy(A.rowids+startIdx, A.rowids+endIdx, B->rowids+B->colptr[i]);
            copy(A.values+startIdx, A.values+endIdx, B->values+B->colptr[i]);
        }
        else if(rilen > 0)
        {
            IT* left=ri;
            for(IT j=startIdx; j<endIdx ; j++)
            {
                if(left < (ri+rilen))
                {
                    left = std::lower_bound(left, ri+rilen, A.rowids[j]);
                    if(A.rowids[j] == *left)
                    {
                        B->values[idx] = A.values[j];
                        B->rowids[idx++] = A.rowids[j];
                    }
                }
            }
            B->colptr[i+1] = idx;
        }
    }
}


void GatherBloomFilters(bloom_filter & bm, bloom_filter * & rbms)
{
	int myrank, nprocs;
    	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

	uint32_t * salcnts = new uint32_t[nprocs];	// salt_count_
	uint64_t * tabs = new uint64_t[nprocs];       // table_size_
	uint64_t * rtss = new uint64_t[nprocs];       // raw_table_size_ 
	uint64_t * projcnts = new uint64_t[nprocs];	// projected_element_count_
	uint32_t * instcnts = new uint32_t[nprocs];	// inserted_element_count_
	uint64_t * randoms = new uint64_t[nprocs];	// random_seed_
	double * dfpr = new double[nprocs]; // desired_false_positive_probability_

	// typedefs inside the bloom filter class
	//  typedef unsigned int bloom_type;
	//  typedef unsigned char cell_type;
	//
	MPI_Allgather(&(bm.salt_count_), 1, MPIType<uint32_t>(), salcnts, 1, MPIType<uint32_t>(), MPI_COMM_WORLD);	// note send&recv diff
	MPI_Allgather(&(bm.table_size_), 1, MPIType<uint64_t>(), tabs, 1, MPIType<uint64_t>(), MPI_COMM_WORLD); 
	MPI_Allgather(&(bm.raw_table_size_), 1, MPIType<uint64_t>(), rtss, 1, MPIType<uint64_t>(), MPI_COMM_WORLD); // note send&recv diff
	MPI_Allgather(&(bm.projected_element_count_), 1, MPIType<uint64_t>(), projcnts, 1, MPIType<uint64_t>(), MPI_COMM_WORLD);
	MPI_Allgather(&(bm.inserted_element_count_), 1, MPIType<uint32_t>(), instcnts, 1, MPIType<uint32_t>(), MPI_COMM_WORLD);
	MPI_Allgather(&(bm.random_seed_), 1, MPIType<uint64_t>(), randoms, 1, MPIType<uint64_t>(), MPI_COMM_WORLD);
	MPI_Allgather(&(bm.desired_false_positive_probability_), 1, MPI_DOUBLE, dfpr, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		
	uint64_t totaltabrecv = accumulate(rtss, rtss+nprocs, static_cast<uint64_t>(0));
	//if(myrank == 0)
    //		cout << "Receiving tables of total " << totaltabrecv << " bytes" << endl;
    int * dpls = new int[nprocs]();	// displacements (zero initialized pid)
    partial_sum(rtss, rtss+nprocs-1, dpls+1);
	vector<int> tablecnts(nprocs);
	copy(rtss, rtss+nprocs, tablecnts.begin());
 
	unsigned char * bit_tables = new unsigned char [totaltabrecv];
	MPI_Allgatherv(bm.bit_table_, bm.raw_table_size_, MPIType<unsigned char>(),
			bit_tables, tablecnts.data(), dpls, MPIType<unsigned char>(), MPI_COMM_WORLD);

   	uint64_t totalsaltrecv = accumulate(salcnts, salcnts+nprocs, static_cast<uint64_t>(0));
    int * saltdpls = new int[nprocs]();	// displacements (zero initialized pid)
    partial_sum(salcnts, salcnts+nprocs-1, saltdpls+1);
	vector<int> saltintcnt(nprocs);
	copy(salcnts, salcnts+nprocs, saltintcnt.begin());

	unsigned int * salts = new unsigned int [totalsaltrecv];
	MPI_Allgatherv((bm.salt_).data(), bm.salt_count_, MPIType<unsigned int>(),
			salts, saltintcnt.data(), saltdpls, MPIType<unsigned int>(), MPI_COMM_WORLD);
    
    	rbms = new bloom_filter[nprocs]();   // remote bloom filters
	for(int i=0; i< nprocs; ++i)
	{
		rbms[i].salt_count_ = salcnts[i];
		rbms[i].table_size_ = tabs[i];
		rbms[i].raw_table_size_ = rtss[i];
		rbms[i].projected_element_count_ = projcnts[i];
		rbms[i].inserted_element_count_ = instcnts[i];
		rbms[i].random_seed_ = randoms[i];
		rbms[i].desired_false_positive_probability_ = dfpr[i];
		rbms[i].bit_table_ = new unsigned char[rtss[i]];
		std::copy(bit_tables+ dpls[i], bit_tables + dpls[i] + tablecnts[i], rbms[i].bit_table_);
		(rbms[i].salt_).resize(salcnts[i]);
		std::copy(salts + saltdpls[i], salts+ saltdpls[i] + saltintcnt[i], (rbms[i].salt_).begin());
	}
    	DeleteAll(dpls, saltdpls);
    	DeleteAll(bit_tables,salts);
	DeleteAll(salcnts, tabs, rtss, projcnts, instcnts, randoms, dfpr);
}



void bmtest()
{
    
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    unsigned int random_seed = 0xA57EC3B2;
    const double desired_probability_of_false_positive = 0.05;
    
    bloom_parameters parameters;
    parameters.projected_element_count    = 10;
    parameters.false_positive_probability = desired_probability_of_false_positive;
    parameters.random_seed                = random_seed++;
    parameters.maximum_number_of_hashes   = 7;
    
    parameters.compute_optimal_parameters();
    bloom_filter bm(parameters);
    
    // first, count nnz in the result matrix
    for(int i=0; i<10; i++)
    {
        bm.insert(i);
    }
    
    bloom_filter * rbms;
    GatherBloomFilters(bm, rbms);
    
    if(myrank==0)
    {
        for(int i=0; i<10; i++)
        {
            if(rbms[0].contains(i))
                cout << i << " ";

            if(bm.contains(i))	
                cout << i << " ";

        }
    }
    cout << endl;
    
}



void debug_print1(const CSC<INDEXTYPE,VALUETYPE>& A_csc, string name)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    cout << "printing matrix" << endl;
    cout << "nrow: " << A_csc.rows << " ncol: " << A_csc.cols << " nnz: "<< A_csc.nnz << endl;
    for(int i=0; i<A_csc.cols ;i++)
    {
        if(A_csc.colptr[i+1] - A_csc.colptr[i])
        {
            cout << i << " : rowids: ";
            copy(A_csc.rowids+A_csc.colptr[i], A_csc.rowids+A_csc.colptr[i+1],ostream_iterator<int>(cout," "));
            cout << " vals: ";
            copy(A_csc.values+A_csc.colptr[i], A_csc.values+A_csc.colptr[i+1],ostream_iterator<int>(cout," "));
            cout << endl;
        }
    }
    
}


void debug_print2(const CSC<INDEXTYPE,VALUETYPE>& A_csc, string name)
{
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    INDEXTYPE until = myrank * A_csc.totalcols/nprocs;
    string filename = "part" + name + "_proc" + patch::to_string(myrank);
    ofstream output(filename.c_str());
    output << "nrow: " << A_csc.rows << " ncol: " << A_csc.cols << endl;
    for(int i=0; i<A_csc.cols ;i++)
    {
        if(A_csc.colptr[i+1] - A_csc.colptr[i])
        {
            output << i+until << " : ";
            for(INDEXTYPE j = A_csc.colptr[i]; j<A_csc.colptr[i+1]; j++)
            {
                output << A_csc.rowids[j] << " ";
            }
            output << endl;
        }
    }
    
}


void debug_print(const CSC<INDEXTYPE,VALUETYPE>& A_csc, string name)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    cout << "printing matrix to file" << endl;
    string filename = "part" + name + "_proc" + patch::to_string(myrank);
    cout << filename  << endl;
    ofstream output(filename.c_str());
    output << "nrow: " << A_csc.rows << " ncol: " << A_csc.cols << endl;
    copy(A_csc.colptr, A_csc.colptr+A_csc.cols+1,ostream_iterator<int>(output," ")); output << " *****"<< endl;
    copy(A_csc.rowids, A_csc.rowids+A_csc.nnz,ostream_iterator<int>(output," ")); output << " *****"<< endl;
    copy(A_csc.values, A_csc.values+A_csc.nnz,ostream_iterator<double>(output," ")); output << " *****"<< endl;
    output.close();
}

template <typename IT, typename NT>
void masking (CSC<IT,NT> & A, const CSC<IT,NT> & mask)
{
    int id = 0;
    int startid=0;
    for(int i=0; i<A.cols; i++)
    {
        for(int j=startid, k=mask.colptr[i]; j<A.colptr[i+1] && k<mask.colptr[i+1];)
        {
            if(A.rowids[j]<mask.rowids[k]) j++;
            else if(A.rowids[j]>mask.rowids[k]) k++;
            else //(A.rowids[j]==mask.rowids[k])
            {
                A.rowids[id] = A.rowids[j];
                A.values[id++] = A.values[j];
                k++;
                j++;
            }
        }
        startid = A.colptr[i+1];
        A.colptr[i+1] = id;
    }
    A.nnz = id;
}

template <typename IT, typename NT>
CSC<IT,NT> TriL_dist(const CSC<IT,NT> & A, bool keepdiagonal = true)
{
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    IT perproc = A.totalcols / nprocs;
    IT coloffs = myrank * perproc;  // column offset (local -> global)
    vector<IT> starts(A.cols);
    CSC<IT,NT> B;
    B.cols = A.cols;
    B.totalcols = A.totalcols;
    B.rows = A.rows;
    B.colptr = new IT[B.cols+1](); // initialize to zero
    
    for(int i=0; i<A.cols ;i++)
    {
        int j = A.colptr[i];
        if(keepdiagonal)    for(; j< A.colptr[i+1] && (A.rowids[j] < (coloffs+i)); ++j);
        else                for(; j< A.colptr[i+1] && (A.rowids[j] <= (coloffs+i)); ++j);
        
        starts[i] = j;
        B.nnz += (A.colptr[i+1]-j); // increment nonzero count
        B.colptr[i+1] = B.colptr[i] + (A.colptr[i+1]-j);
    }
    //copy(B.colptr,B.colptr+B.cols+1, ostream_iterator<IT>(cout, " ")); cout << endl;
    B.rowids = new IT[B.nnz];
    B.values = new NT[B.nnz];
    
    for(int i=0; i<A.cols ;i++)
    {
        copy(A.rowids+starts[i], A.rowids+A.colptr[i+1], B.rowids+B.colptr[i]);
        copy(A.values+starts[i], A.values+A.colptr[i+1], B.values+B.colptr[i]);
    }
    return B;
}

template <typename IT, typename NT>
CSC<IT,NT> TriU_dist(const CSC<IT,NT> & A, bool keepdiagonal = true)
{
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    IT perproc = A.totalcols / nprocs;
    IT coloffs = myrank * perproc;  // column offset (local -> global)
    vector<IT> ends(A.cols);
    CSC<IT,NT> B;
    B.cols = A.cols;
    B.totalcols = A.totalcols;
    B.rows = A.rows;
    B.colptr = new IT[B.cols+1](); // initialize to zero
    
    for(int i=0; i<A.cols ;i++)
    {
        int j = A.colptr[i];
        if(keepdiagonal)    for(; j< A.colptr[i+1] && (A.rowids[j] <= (coloffs+i)); ++j);
        else                for(; j< A.colptr[i+1] && (A.rowids[j] < (coloffs+i)); ++j);
        
        ends[i] = j;
        B.nnz += (j-A.colptr[i]); // increment nonzero count
        B.colptr[i+1] = B.colptr[i] + (j-A.colptr[i]);
    }
    //copy(B.colptr,B.colptr+B.cols+1, ostream_iterator<IT>(cout, " ")); cout << endl;
    B.rowids = new IT[B.nnz];
    B.values = new NT[B.nnz];
    
    for(int i=0; i<A.cols ;i++)
    {
        copy(A.rowids+A.colptr[i], A.rowids+ends[i], B.rowids+B.colptr[i]);
        copy(A.values+A.colptr[i], A.values+ends[i], B.values+B.colptr[i]);
    }
    return B;
}

/*
template <typename IT, typename NT>
CSC<IT,NT> SpGEMM_dist(CSC<IT,NT> & L, CSC<IT,NT> & U)
{
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    for(int i=0; i< nprocs; i++)
    {
        if(myrank == i)
        {
            LRecv = L;
            MPI_Bcast(Lrecv);
            LRecv * U
        }
    }
    
}
 */






template <typename IT, typename NT>
CSC<IT,NT> SpGEMM_dist(const CSC<IT,NT> & L, const CSC<IT,NT> & U, const CSC<IT,NT> & A)
{
    
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    double t1=0, t2=0;
    //-------------------------------------------------------
    // find col indecies ci of L needed by this processor
    // Divide parts of ci to to be requested from different processors
    //-------------------------------------------------------
    double t3=MPI_Wtime();
    vector<IT> ci;
    //ci = RowIds(U);
    ci = SPARowIds(U);
    vector< vector< IT > > cisent(nprocs);
    IT lind, gind;
    int owner;
    IT perproc = L.totalcols / nprocs;
    
    for(int i=0; i<ci.size(); i++)
    {
        gind = ci[i];
        if(perproc != 0)
            owner = std::min(static_cast<int>(gind / perproc), nprocs-1);
        else
            owner = nprocs -1;
        IT lind = gind - (owner * perproc);
        cisent[owner].push_back(lind);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    t3=MPI_Wtime()-t3;
    
    
    //-------------------------------------------------------
    // exchange ci
    //-------------------------------------------------------
    
    double t4=MPI_Wtime();
    MPI_Pcontrol(1,"dist-ci");
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs];
    int * rdispls = new int[nprocs];
    
    
    for(int i=0; i<nprocs; ++i)
        sendcnt[i] = (int) cisent[i].size();
    
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, MPI_COMM_WORLD);  // share the request counts
    
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<(nprocs-1); ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    
    IT totsend = sdispls[nprocs-1] + sendcnt[nprocs-1];
    IT totrecv = rdispls[nprocs-1] + recvcnt[nprocs-1];
    IT * cibuf = new IT[totsend];
    for(int i=0; i<nprocs; ++i)
    {
        copy(cisent[i].begin(), cisent[i].end(), cibuf+sdispls[i]);
        vector<IT>().swap(cisent[i]);
    }
    
    IT * recvci = new IT[totrecv];
    MPI_Alltoallv(cibuf, sendcnt, sdispls, MPIType<IT>(), recvci, recvcnt, rdispls, MPIType<IT>(), MPI_COMM_WORLD);
    
    DeleteAll(cibuf,sendcnt, sdispls);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t4=MPI_Wtime()-t4;
    MPI_Pcontrol(-1,"dist-ci");
    //-------------------------------------------------------
    // retrieve the requested submatrices with
    // conlumn indices requested from proc i in recvci+rdispls[i]
    //-------------------------------------------------------
    
    double t5=MPI_Wtime();
    int * sendcnt_val = new int[nprocs];
    int * sendcnt_colptr = new int[nprocs];
    int * recvcnt_val = new int[nprocs];
    int * recvcnt_colptr = new int[nprocs];
    int * sdispls_val = new int[nprocs];
    int * sdispls_colptr = new int[nprocs];
    int * rdispls_val = new int[nprocs];
    int * rdispls_colptr = new int[nprocs];
    
    
    vector <CSC<IT,NT>* > refmats(nprocs);
    IT * ri=NULL; // dummy, never used in SpRef
    
    for(int i=0; i<nprocs; ++i)
    {
        SpRef_basic(L, recvci+rdispls[i], recvcnt[i], ri, 0, true, refmats[i]); // select all rows of L
        // think carefully for empty D_csc
        sendcnt_colptr[i] = (int) refmats[i]->cols + 1;
        sendcnt_val[i] = (int) refmats[i]->nnz;
    }
    
    DeleteAll(recvcnt, rdispls, recvci);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t5=MPI_Wtime()-t5;
    
    double t6=MPI_Wtime();
    MPI_Pcontrol(1,"dist-submat");
    MPI_Alltoall(sendcnt_val, 1, MPI_INT, recvcnt_val, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(sendcnt_colptr, 1, MPI_INT, recvcnt_colptr, 1, MPI_INT, MPI_COMM_WORLD);
    
    sdispls_val[0] = 0;
    rdispls_val[0] = 0;
    sdispls_colptr[0] = 0;
    rdispls_colptr[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls_val[i+1] = sdispls_val[i] + sendcnt_val[i];
        rdispls_val[i+1] = rdispls_val[i] + recvcnt_val[i];
        sdispls_colptr[i+1] = sdispls_colptr[i] + sendcnt_colptr[i];
        rdispls_colptr[i+1] = rdispls_colptr[i] + recvcnt_colptr[i];
    }
    
    
    IT totsend_colptr = sdispls_colptr[nprocs-1] + sendcnt_colptr[nprocs-1];
    IT totsend_val = sdispls_val[nprocs-1] + sendcnt_val[nprocs-1];
    IT totrecv_colptr = rdispls_colptr[nprocs-1] + recvcnt_colptr[nprocs-1];
    IT totrecv_val = rdispls_val[nprocs-1] + recvcnt_val[nprocs-1];
    
    IT * send_colptr = new IT[totsend_colptr];
    NT * send_values = new NT[totsend_val];
    IT * send_rowids = new IT[totsend_val];
    
    for(int i=0; i<nprocs; ++i)
    {
        copy(refmats[i]->colptr, refmats[i]->colptr + sendcnt_colptr[i], send_colptr + sdispls_colptr[i]);
        copy(refmats[i]->rowids, refmats[i]->rowids + sendcnt_val[i], send_rowids + sdispls_val[i]);
        copy(refmats[i]->values, refmats[i]->values + sendcnt_val[i], send_values + sdispls_val[i]);
    }
    
    vector<CSC<IT,NT>*>().swap(refmats); // destroy refmats
    
    IT * recv_colptr = new IT[totrecv_colptr];
    NT * recv_values = new NT[totrecv_val];
    IT * recv_rowids = new IT[totrecv_val];
    
    MPI_Alltoallv(send_colptr, sendcnt_colptr, sdispls_colptr, MPIType<IT>(), recv_colptr, recvcnt_colptr, rdispls_colptr, MPIType<IT>(), MPI_COMM_WORLD);
    MPI_Alltoallv(send_rowids, sendcnt_val, sdispls_val, MPIType<IT>(), recv_rowids, recvcnt_val, rdispls_val, MPIType<IT>(), MPI_COMM_WORLD);
    MPI_Alltoallv(send_values, sendcnt_val, sdispls_val, MPIType<NT>(), recv_values, recvcnt_val, rdispls_val, MPIType<NT>(), MPI_COMM_WORLD);
    
    DeleteAll(sendcnt_val, sendcnt_colptr, sdispls_val, sdispls_colptr, send_colptr, send_values, send_rowids);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t6=MPI_Wtime()-t6;
    MPI_Pcontrol(-1,"dist-submat");
    
    //-------------------------------------------------------
    // create n*n matrix from the received pieces
    //-------------------------------------------------------
    
    double t7=MPI_Wtime();
    CSC<IT,NT> C;
    C.nnz = totrecv_val;
    C.rows = A.rows;
    C.cols = L.totalcols;
    C.values = recv_values;
    C.rowids = recv_rowids;
    C.colptr = new IT [C.cols + 1] ;
    std::fill_n(C.colptr, C.cols+1, -1);
    C.colptr[0] = 0;
    
    
    // note that, length(recvcnt_colptr[i]) = length(sendcnt[i])+1 = cisent[i].size()+1
    IT k =0;
    for(int i=0; i<nprocs; i++)
    {
        for(int j=0; j < (recvcnt_colptr[i]-1); j++)
        {
            C.colptr[ci[k]] = rdispls_val[i] + recv_colptr[j + rdispls_colptr[i]];
            for(IT l=ci[k]-1; (l>=0) && (C.colptr[l] == -1); l--) C.colptr[l] = C.colptr[ci[k]];
            k++;
        }
    }
    C.colptr[C.cols] = C.nnz;
    for(IT l=C.cols-1; (l>=0) && (C.colptr[l] == -1); l--) C.colptr[l] = C.nnz;
    
    
    DeleteAll(recvcnt_val, recvcnt_colptr, rdispls_val, rdispls_colptr, recv_colptr);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t7=MPI_Wtime()-t7;
    //delete [] recv_values; // directly used in C
    //delete [] recv_rowids; // directly used in C
    
    //-------------------------------------------------------
    // Now multiply C*U
    //-------------------------------------------------------
    
    double t8=MPI_Wtime();
    CSC<IT,NT> res;
    LocalSpGEMM(C, U, multiplies<NT>(), plus<NT>(), myidentity<NT>(), res);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t8=MPI_Wtime()-t8;
    // perform masking
    double t9=MPI_Wtime();
    masking (res, A);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t9=MPI_Wtime()-t9;
    
    /*
    if(myrank==0)
    {
        cout << nprocs << " " << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << " " << t6 << " " << t7 << " " << t8 << " " << t9  << endl;
    }
    */
    return res;
    
}



template <typename IT, typename NT>
CSC<IT,NT> Masked_SpGEMM_dist_nofilter(const CSC<IT,NT> & L, const CSC<IT,NT> & U, const CSC<IT,NT> & A)
{
    
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    //-------------------------------------------------------
    // find row indecies ri of A needed by this processor
    //-------------------------------------------------------
    
    double t1=MPI_Wtime();
    vector<IT> ri;
    //ri = RowIds(A);
    ri = SPARowIds(A);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime()-t1;
    
    
    //-------------------------------------------------------
    // exchange ri
    //-------------------------------------------------------
    
    double t2=MPI_Wtime();
    MPI_Pcontrol(1,"dist-ri");
    int * recvcnt_ri = new int[nprocs];
    int * rdispls_ri = new int[nprocs];
    int ri_size = ri.size();
    MPI_Allgather(&ri_size, 1, MPI_INT, recvcnt_ri, 1, MPI_INT, MPI_COMM_WORLD);
    rdispls_ri[0] = 0;
    for(int i=0; i<(nprocs-1); ++i)
    {
        rdispls_ri[i+1] = rdispls_ri[i] + recvcnt_ri[i];
    }
    IT totrecv_ri = rdispls_ri[nprocs-1] + recvcnt_ri[nprocs-1];
    IT * recvri = new IT[totrecv_ri];
    MPI_Allgatherv(ri.data(), ri.size(), MPIType<IT>(), recvri, recvcnt_ri, rdispls_ri, MPIType<IT>(), MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t2=MPI_Wtime()-t2;
    MPI_Pcontrol(-1,"dist-ri");
    
    //-------------------------------------------------------
    // find col indecies ci of L needed by this processor
    // Divide parts of ci to to be requested from different processors
    //-------------------------------------------------------
    double t3=MPI_Wtime();
    vector<IT> ci;
    ci = SPARowIds(U);
    //ci = RowIds(U);
    vector< vector< IT > > cisent(nprocs);
    IT lind, gind;
    int owner;
    IT perproc = L.totalcols / nprocs;
    
    for(int i=0; i<ci.size(); i++)
    {
        gind = ci[i];
        if(perproc != 0)
            owner = std::min(static_cast<int>(gind / perproc), nprocs-1);
        else
            owner = nprocs -1;
        IT lind = gind - (owner * perproc);
        cisent[owner].push_back(lind);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    t3=MPI_Wtime()-t3;
    
    
    //-------------------------------------------------------
    // exchange ci
    //-------------------------------------------------------
    double t4=MPI_Wtime();
    MPI_Pcontrol(1,"dist-ci");
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs];
    int * rdispls = new int[nprocs];
    
    
    for(int i=0; i<nprocs; ++i)
        sendcnt[i] = (int) cisent[i].size();
    
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, MPI_COMM_WORLD);  // share the request counts
    
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<(nprocs-1); ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    
    IT totsend = sdispls[nprocs-1] + sendcnt[nprocs-1];
    IT totrecv = rdispls[nprocs-1] + recvcnt[nprocs-1];
    IT * cibuf = new IT[totsend];
    for(int i=0; i<nprocs; ++i)
    {
        copy(cisent[i].begin(), cisent[i].end(), cibuf+sdispls[i]);
        vector<IT>().swap(cisent[i]);
    }
    
    IT * recvci = new IT[totrecv];
    MPI_Alltoallv(cibuf, sendcnt, sdispls, MPIType<IT>(), recvci, recvcnt, rdispls, MPIType<IT>(), MPI_COMM_WORLD);
    
    DeleteAll(cibuf,sendcnt, sdispls);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t4=MPI_Wtime()-t4;
    MPI_Pcontrol(-1,"dist-ci");
    
    //-------------------------------------------------------
    // retrieve the requested submatrices with
    // rowindices from requested from proc i in rbms[i]
    // conlumn indices requested from proc i in recvci+rdispls[i]
    //-------------------------------------------------------
    double t5=MPI_Wtime();
    int * sendcnt_val = new int[nprocs];
    int * sendcnt_colptr = new int[nprocs];
    int * recvcnt_val = new int[nprocs];
    int * recvcnt_colptr = new int[nprocs];
    int * sdispls_val = new int[nprocs];
    int * sdispls_colptr = new int[nprocs];
    int * rdispls_val = new int[nprocs];
    int * rdispls_colptr = new int[nprocs];
    
    
    vector <CSC<IT,NT>* > refmats(nprocs);
   
    for(int i=0; i<nprocs; ++i)
    {
        //SpRef_basic(L, recvci+rdispls[i], recvcnt[i],  recvri+rdispls_ri[i], recvcnt_ri[i], false, refmats[i]); // select all rows of L
        SpRef_finger(L, recvci+rdispls[i], recvcnt[i],  recvri+rdispls_ri[i], recvcnt_ri[i], false, refmats[i]);
        // think carefully for empty D_csc
        sendcnt_colptr[i] = (int) refmats[i]->cols + 1;
        sendcnt_val[i] = (int) refmats[i]->nnz;
    }
    
    DeleteAll(recvcnt, rdispls, recvci);
    DeleteAll(recvcnt_ri, rdispls_ri, recvri);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t5=MPI_Wtime()-t5;
    MPI_Pcontrol(1,"dist-submat");
    
    double t6=MPI_Wtime();
    MPI_Alltoall(sendcnt_val, 1, MPI_INT, recvcnt_val, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(sendcnt_colptr, 1, MPI_INT, recvcnt_colptr, 1, MPI_INT, MPI_COMM_WORLD);
    
    sdispls_val[0] = 0;
    rdispls_val[0] = 0;
    sdispls_colptr[0] = 0;
    rdispls_colptr[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls_val[i+1] = sdispls_val[i] + sendcnt_val[i];
        rdispls_val[i+1] = rdispls_val[i] + recvcnt_val[i];
        sdispls_colptr[i+1] = sdispls_colptr[i] + sendcnt_colptr[i];
        rdispls_colptr[i+1] = rdispls_colptr[i] + recvcnt_colptr[i];
    }
    
    
    IT totsend_colptr = sdispls_colptr[nprocs-1] + sendcnt_colptr[nprocs-1];
    IT totsend_val = sdispls_val[nprocs-1] + sendcnt_val[nprocs-1];
    IT totrecv_colptr = rdispls_colptr[nprocs-1] + recvcnt_colptr[nprocs-1];
    IT totrecv_val = rdispls_val[nprocs-1] + recvcnt_val[nprocs-1];
    
    IT * send_colptr = new IT[totsend_colptr];
    NT * send_values = new NT[totsend_val];
    IT * send_rowids = new IT[totsend_val];
    
    for(int i=0; i<nprocs; ++i)
    {
        copy(refmats[i]->colptr, refmats[i]->colptr + sendcnt_colptr[i], send_colptr + sdispls_colptr[i]);
        copy(refmats[i]->rowids, refmats[i]->rowids + sendcnt_val[i], send_rowids + sdispls_val[i]);
        copy(refmats[i]->values, refmats[i]->values + sendcnt_val[i], send_values + sdispls_val[i]);
    }
    
    vector<CSC<IT,NT>*>().swap(refmats); // destroy refmats
    
    IT * recv_colptr = new IT[totrecv_colptr];
    NT * recv_values = new NT[totrecv_val];
    IT * recv_rowids = new IT[totrecv_val];
    
    MPI_Alltoallv(send_colptr, sendcnt_colptr, sdispls_colptr, MPIType<IT>(), recv_colptr, recvcnt_colptr, rdispls_colptr, MPIType<IT>(), MPI_COMM_WORLD);
    MPI_Alltoallv(send_rowids, sendcnt_val, sdispls_val, MPIType<IT>(), recv_rowids, recvcnt_val, rdispls_val, MPIType<IT>(), MPI_COMM_WORLD);
    MPI_Alltoallv(send_values, sendcnt_val, sdispls_val, MPIType<NT>(), recv_values, recvcnt_val, rdispls_val, MPIType<NT>(), MPI_COMM_WORLD);
    
    DeleteAll(sendcnt_val, sendcnt_colptr, sdispls_val, sdispls_colptr, send_colptr, send_values, send_rowids);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t6=MPI_Wtime()-t6;
    MPI_Pcontrol(-1,"dist-submat");
    //-------------------------------------------------------
    // create n*n matrix from the received pieces
    //-------------------------------------------------------
    double t7=MPI_Wtime();
    CSC<IT,NT> C;
    C.nnz = totrecv_val;
    C.rows = A.rows;
    C.cols = L.totalcols;
    C.values = recv_values;
    C.rowids = recv_rowids;
    C.colptr = new IT [C.cols + 1] ;
    std::fill_n(C.colptr, C.cols+1, -1);
    C.colptr[0] = 0;
    
    
    // note that, length(recvcnt_colptr[i]) = length(sendcnt[i])+1 = cisent[i].size()+1
    IT k =0;
    for(int i=0; i<nprocs; i++)
    {
        for(int j=0; j < (recvcnt_colptr[i]-1); j++)
        {
            C.colptr[ci[k]] = rdispls_val[i] + recv_colptr[j + rdispls_colptr[i]];
            for(IT l=ci[k]-1; (l>=0) && (C.colptr[l] == -1); l--) C.colptr[l] = C.colptr[ci[k]];
            k++;
        }
    }
    C.colptr[C.cols] = C.nnz;
    for(IT l=C.cols-1; (l>=0) && (C.colptr[l] == -1); l--) C.colptr[l] = C.nnz;
    
    
    DeleteAll(recvcnt_val, recvcnt_colptr, rdispls_val, rdispls_colptr, recv_colptr);
    //delete [] recv_values; // directly used in C
    //delete [] recv_rowids; // directly used in C
    MPI_Barrier(MPI_COMM_WORLD);
    t7=MPI_Wtime()-t7;
    
    //-------------------------------------------------------
    // Now multiply C*U
    //-------------------------------------------------------
    double t8=MPI_Wtime();
    CSC<IT,NT> res;
    LocalSpGEMM(C, U, multiplies<NT>(), plus<NT>(), myidentity<NT>(), res);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t8=MPI_Wtime()-t8;
    // perform masking
    double t9=MPI_Wtime();
    masking (res, A);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t9=MPI_Wtime()-t9;
    
    /*
    if(myrank==0)
    {
        cout << nprocs << " " << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << " " << t6 << " " << t7 << " " << t8 << " " << t9  << endl;
    }*/
    
    return res;
    
}





template <typename IT, typename NT>
CSC<IT,NT> Masked_SpGEMM_dist(const CSC<IT,NT> & L, const CSC<IT,NT> & U, const CSC<IT,NT> & A)
{
    
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    //-------------------------------------------------------
    // find row indecies of L needed by this processor
    // create bloom filter and send it to every processor
    //-------------------------------------------------------
    double t1=MPI_Wtime();
    //bloom_filter bm = rowFilter(A);
    bloom_filter bm = SPARowIds_bloom(A);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t1=MPI_Wtime()-t1;
    
    double t2=MPI_Wtime();
    MPI_Pcontrol(1,"dist-bloom");
    bloom_filter * rbms;
    GatherBloomFilters(bm, rbms);
    MPI_Pcontrol(-1,"dist-sbloom");
    
    MPI_Barrier(MPI_COMM_WORLD);
    t2=MPI_Wtime()-t2;
    
    //-------------------------------------------------------
    // find col indecies ci of L needed by this processor
    // Divide parts of ci to to be requested from different processors
    //-------------------------------------------------------
    double t3=MPI_Wtime();
    vector<IT> ci;
    //ci = RowIds(U);
    ci = SPARowIds(U);
    vector< vector< IT > > cisent(nprocs);
    IT lind, gind;
    int owner;
    IT perproc = L.totalcols / nprocs;
    
    for(int i=0; i<ci.size(); i++)
    {
        gind = ci[i];
        if(perproc != 0)
            owner = std::min(static_cast<int>(gind / perproc), nprocs-1);
        else
            owner = nprocs -1;
        IT lind = gind - (owner * perproc);
        cisent[owner].push_back(lind);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    t3=MPI_Wtime()-t3;
    /*
    cout << "processor " << myrank << endl;
    for(int i=0; i<nprocs ; i++)
    {
        for(int j=0; j<cisent[i].size() ; j++)
            cout << cisent[i][j] << " ";
        cout << endl;
    }
    */
    

    //-------------------------------------------------------
    // exchange ci
    //-------------------------------------------------------
    double t4=MPI_Wtime();
    MPI_Pcontrol(1,"dist-ci");
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs];
    int * rdispls = new int[nprocs];
    
    
    for(int i=0; i<nprocs; ++i)
        sendcnt[i] = (int) cisent[i].size();

    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, MPI_COMM_WORLD);  // share the request counts
    
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<(nprocs-1); ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
  
    
    IT totsend = sdispls[nprocs-1] + sendcnt[nprocs-1];
    IT totrecv = rdispls[nprocs-1] + recvcnt[nprocs-1];
    IT * cibuf = new IT[totsend];
    for(int i=0; i<nprocs; ++i)
    {
        copy(cisent[i].begin(), cisent[i].end(), cibuf+sdispls[i]);
        vector<IT>().swap(cisent[i]);
    }
    
    IT * recvci = new IT[totrecv];
    MPI_Alltoallv(cibuf, sendcnt, sdispls, MPIType<IT>(), recvci, recvcnt, rdispls, MPIType<IT>(), MPI_COMM_WORLD);
    
    DeleteAll(cibuf,sendcnt, sdispls);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(-1,"dist-ci");
    t4=MPI_Wtime()-t4;
    /*
    cout << "processor " << myrank << endl;
    for(int i=0; i<nprocs ; i++)
    {
        for(int j=0; j< recvcnt[i] ; j++)
            cout << recvci[j+rdispls[i]] << " ";
        cout << endl;
    }
     */
    
    
    
    //-------------------------------------------------------
    // retrieve the requested submatrices with
    // rowindices from requested from proc i in rbms[i]
    // conlumn indices requested from proc i in recvci+rdispls[i]
    //-------------------------------------------------------
    
    double t5=MPI_Wtime();
    int * sendcnt_val = new int[nprocs];
    int * sendcnt_colptr = new int[nprocs];
    int * recvcnt_val = new int[nprocs];
    int * recvcnt_colptr = new int[nprocs];
    int * sdispls_val = new int[nprocs];
    int * sdispls_colptr = new int[nprocs];
    int * rdispls_val = new int[nprocs];
    int * rdispls_colptr = new int[nprocs];

    
    vector <CSC<IT,NT>* > refmats(nprocs);
    for(int i=0; i<nprocs; ++i)
    {
         SpRef(L, recvci+rdispls[i], recvcnt[i], rbms[i], refmats[i]); // rbms[i]: bloom filter received from ith processor
        // think carefully for empty D_csc
        sendcnt_colptr[i] = (int) refmats[i]->cols + 1;
        sendcnt_val[i] = (int) refmats[i]->nnz;
    }
    
    DeleteAll(recvcnt, rdispls, rbms, recvci);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t5=MPI_Wtime()-t5;
    MPI_Pcontrol(1,"dist-submat");
    
    double t6=MPI_Wtime();
    MPI_Alltoall(sendcnt_val, 1, MPI_INT, recvcnt_val, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(sendcnt_colptr, 1, MPI_INT, recvcnt_colptr, 1, MPI_INT, MPI_COMM_WORLD);
    
    sdispls_val[0] = 0;
    rdispls_val[0] = 0;
    sdispls_colptr[0] = 0;
    rdispls_colptr[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls_val[i+1] = sdispls_val[i] + sendcnt_val[i];
        rdispls_val[i+1] = rdispls_val[i] + recvcnt_val[i];
        sdispls_colptr[i+1] = sdispls_colptr[i] + sendcnt_colptr[i];
        rdispls_colptr[i+1] = rdispls_colptr[i] + recvcnt_colptr[i];
    }
    
    
    IT totsend_colptr = sdispls_colptr[nprocs-1] + sendcnt_colptr[nprocs-1];
    IT totsend_val = sdispls_val[nprocs-1] + sendcnt_val[nprocs-1];
    IT totrecv_colptr = rdispls_colptr[nprocs-1] + recvcnt_colptr[nprocs-1];
    IT totrecv_val = rdispls_val[nprocs-1] + recvcnt_val[nprocs-1];

    IT * send_colptr = new IT[totsend_colptr];
    NT * send_values = new NT[totsend_val];
    IT * send_rowids = new IT[totsend_val];

    for(int i=0; i<nprocs; ++i)
    {
        copy(refmats[i]->colptr, refmats[i]->colptr + sendcnt_colptr[i], send_colptr + sdispls_colptr[i]);
        copy(refmats[i]->rowids, refmats[i]->rowids + sendcnt_val[i], send_rowids + sdispls_val[i]);
        copy(refmats[i]->values, refmats[i]->values + sendcnt_val[i], send_values + sdispls_val[i]);
    }
    
    vector<CSC<IT,NT>*>().swap(refmats); // destroy refmats
     
    IT * recv_colptr = new IT[totrecv_colptr];
    NT * recv_values = new NT[totrecv_val];
    IT * recv_rowids = new IT[totrecv_val];
    
    MPI_Alltoallv(send_colptr, sendcnt_colptr, sdispls_colptr, MPIType<IT>(), recv_colptr, recvcnt_colptr, rdispls_colptr, MPIType<IT>(), MPI_COMM_WORLD);
    MPI_Alltoallv(send_rowids, sendcnt_val, sdispls_val, MPIType<IT>(), recv_rowids, recvcnt_val, rdispls_val, MPIType<IT>(), MPI_COMM_WORLD);
    MPI_Alltoallv(send_values, sendcnt_val, sdispls_val, MPIType<NT>(), recv_values, recvcnt_val, rdispls_val, MPIType<NT>(), MPI_COMM_WORLD);
    
    DeleteAll(sendcnt_val, sendcnt_colptr, sdispls_val, sdispls_colptr, send_colptr, send_values, send_rowids);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t6=MPI_Wtime()-t6;
    MPI_Pcontrol(-1,"dist-submat");
    
    //-------------------------------------------------------
    // create n*n matrix from the received pieces
    //-------------------------------------------------------
    
    double t7=MPI_Wtime();
    CSC<IT,NT> C;
    C.nnz = totrecv_val;
    C.rows = A.rows;
    C.cols = L.totalcols;
    C.values = recv_values;
    C.rowids = recv_rowids;
    C.colptr = new IT [C.cols + 1] ;
    std::fill_n(C.colptr, C.cols+1, -1);
    C.colptr[0] = 0;

    
    // note that, length(recvcnt_colptr[i]) = length(sendcnt[i])+1 = cisent[i].size()+1
    IT k =0;
    for(int i=0; i<nprocs; i++)
    {
        for(int j=0; j < (recvcnt_colptr[i]-1); j++)
        {
            C.colptr[ci[k]] = rdispls_val[i] + recv_colptr[j + rdispls_colptr[i]];
            for(IT l=ci[k]-1; (l>=0) && (C.colptr[l] == -1); l--) C.colptr[l] = C.colptr[ci[k]];
            k++;
        }
    }
    C.colptr[C.cols] = C.nnz;
    for(IT l=C.cols-1; (l>=0) && (C.colptr[l] == -1); l--) C.colptr[l] = C.nnz;
    

    DeleteAll(recvcnt_val, recvcnt_colptr, rdispls_val, rdispls_colptr, recv_colptr);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t7=MPI_Wtime()-t7;
    //delete [] recv_values; // directly used in C
    //delete [] recv_rowids; // directly used in C
    
    //-------------------------------------------------------
    // Now multiply C*U
    //-------------------------------------------------------
    double t8=MPI_Wtime();
    CSC<IT,NT> res;
    LocalSpGEMM(C, U, multiplies<NT>(), plus<NT>(), myidentity<NT>(), res);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t8=MPI_Wtime()-t8;
    // perform masking
    double t9=MPI_Wtime();
    
    masking (res, A);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t9=MPI_Wtime()-t9;
    
    /*
    if(myrank==0)
    {
        cout << nprocs << " " << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << " " << t6 << " " << t7 << " " << t8 << " " << t9  << endl;
    }*/
    
    return res;

}





int main(int argc, char* argv[])
{
	bool binary = false;
	bool product = false;
	bool gen = false;
	string inputname1, inputname2, inputname3, outputname;

	int edgefactor, scale;
	generator_type gtype;
    
    int provided, flag, claimed;
    MPI_Init_thread( 0, 0, MPI_THREAD_FUNNELED, &provided );
    
    MPI_Is_thread_main( &flag );
    if (!flag) {
        printf( "This thread called init_thread but Is_thread_main gave false\n" );fflush(stdout);
    }
    MPI_Query_thread( &claimed );
    if (claimed != provided) {
        printf( "Query thread gave thread level %d but Init_thread gave %d\n", claimed, provided );fflush(stdout);
    }
    int nprocs, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    if(argc < 2)
    {
        cout << "Normal usage: ./triangles input.mtx" << endl;
        return -1;
    }
    
    {
        int omp_num_my = omp_get_max_threads();
        cout << "Current OMP_NUM value is: " << omp_get_max_threads() << endl;
        omp_set_num_threads(1); // single threaded at this moment
        cout << "Set OMP_NUM to: " << omp_get_max_threads() << endl;
        inputname1 =  argv[1];
        
        /*
        if(argc > 2 )
        {
            product = true;
            outputname = argv[2];
        }*/
        
        CSC<INDEXTYPE,VALUETYPE> * A_csc, * C_csc_verify;
        //cout << "reading input matrices in text(ascii)... " << endl;
        if(nprocs > 1)
        {
            ReadASCIIDist( inputname1, A_csc, true );
            //if(product)
              //  ReadASCIIDist( outputname, C_csc_verify, true );
        }
        else
        {
            //ReadASCII( inputname1, A_csc );
            ReadBinary( inputname1, A_csc );
            //if(product)
              //  ReadASCII( outputname, C_csc_verify );
        }
        ///
        //
        omp_set_num_threads(omp_num_my);
        cout << "Set OMP_NUM to: " << omp_get_max_threads() << endl;
 
       	CSC<INDEXTYPE,VALUETYPE> L_csc = TriL_dist(*A_csc, false);
        CSC<INDEXTYPE,VALUETYPE> U_csc = TriU_dist(*A_csc, false);
        //A_csc->Sorted();
        //L_csc.Sorted();
        //U_csc.Sorted();
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        double tstart = MPI_Wtime();
        /*
        CSC<INDEXTYPE,VALUETYPE> C_csc1;
        C_csc1 = Masked_SpGEMM_dist(L_csc, U_csc, *A_csc);
        MPI_Barrier(MPI_COMM_WORLD);
        if(myrank==0)
            cout << " Triangle counting using Masked_SpGEMM_dist (communication avoiding and using Bloom filter)\n Runtime: " << MPI_Wtime() - tstart << endl;
        tstart = MPI_Wtime();
        
        CSC<INDEXTYPE,VALUETYPE> C_csc3;
        C_csc3 = Masked_SpGEMM_dist_nofilter(L_csc, U_csc, *A_csc);
        MPI_Barrier(MPI_COMM_WORLD);
        if(myrank==0)
            cout << " Triangle counting using Masked_SpGEMM_dist_nofilter (communication avoiding but no Bloom filter)\n Runtime: " << MPI_Wtime() - tstart << endl;
        tstart = MPI_Wtime();
        */

        CSC<INDEXTYPE,VALUETYPE> C_csc2;
        C_csc2 = SpGEMM_dist(L_csc, U_csc, *A_csc);
        MPI_Barrier(MPI_COMM_WORLD);
        if(myrank==0)
            cout << " Triangle counting using SpGEMM_dist (no communication avoiding and no Bloom filter)\n Runtime: " << MPI_Wtime() - tstart << endl;
        
         
        
        
        /*
        if(product)
        {
            if(*C_csc_verify == C_csc3)
                cout << "LocalSpGEMM is correct" << endl;
            else
                cout << "LocalSpGEMM is INcorrect" << endl;
            
            delete C_csc_verify;
        }
        */
        
        delete A_csc;
    }

    
    MPI_Finalize();
    return 0;
		
}
