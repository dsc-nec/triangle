#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <cassert>

#include <stdio.h>
#include <map>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <tuple>

using namespace std;

template<typename _ForwardIter, typename T>
void my_iota(_ForwardIter __first, _ForwardIter __last, T __val)
{
	while (__first != __last)
     		*__first++ = __val++;
}

int main(int argc, char *argv[] )
{
    if(argc < 2)
    {
        cout << "Usage: " << argv[0] << " <filename> " << endl;
        return 0;
    }
    uint64_t m;
    uint64_t n;
    uint64_t nnz;
    
    string oname(argv[1]);
    oname += ".random";
    FILE * rFile = fopen (argv[1],"r");
    FILE * wFile = fopen (oname.c_str(),"w");
    if(rFile != NULL)
    {
        cout << "Reading text file" << endl;
        
        size_t n=256;
        char * comment = (char*) malloc(n);
        int bytes_read = getline(&comment, &n, rFile);
        while(comment[0] == '%')
        {
            bytes_read = getline(&comment, &n, rFile);
        }
        stringstream ss;
        ss << string(comment);
        ss >> m >> n >> nnz;
	assert(m == n);

	vector<int> permutation(m, 1);
	my_iota(permutation.begin(), permutation.end(), 1);	// first value 1, last value m
	random_shuffle (permutation.begin(), permutation.end()); 
        
        int row;
        int col;
        float val;
        vector< tuple<int, int, double> > tuples;

	fprintf(wFile, "%d %d %d\n", m, n, nnz);
        while(!feof(rFile))
        {
            if(fscanf (rFile, "%d %d %f",&row, &col, &val) > 0)
	    	fprintf(wFile, "%d %d %f\n", permutation[row-1], permutation[col-1], val);
        }
	fclose(wFile);
    }
    return 0;

}

