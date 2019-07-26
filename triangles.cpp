#include <frovedis/matrix/crs_matrix.hpp>
#include <frovedis/matrix/spgemm.hpp>
#include <chrono>
#include <string>

using namespace std;
using namespace frovedis;

int main(int argc, char* argv[]) {
	string path = argv[1];
	// Input graph must be a sparse upper triangular matrix. 
	auto U = make_crs_matrix_local_loadbinary<long,int>(path);
	cout << "Start calculation \n";
	auto start = chrono::high_resolution_clock::now();
	auto L = U.transpose();
	cout << "After transposing...";
	auto B = spgemm(L,U);
	cout << "After spgemm...";

	// U = U .* B; sparse element-wise product
	int indexU = 0;
	int indexB = 0;
	int rowU = 0;
	int rowB = 0;
	int FLAG = 0; //Record whether the current element has been changed
	int sizeU = U.val.size();
	int sizeB = B.val.size();
	int size_offU = U.off.size();
	int size_offB = B.off.size();
	auto offU = U.off.data();
	auto offB = B.off.data();
	auto valU = U.val.data();
	auto valB = B.val.data();
	auto idxU = U.idx.data();
	auto idxB = B.idx.data();


	for ( ; indexU < sizeU; indexU++)
	{
		FLAG = 0;
		// Update rowU
		for ( ; rowU < size_offU; rowU++)
		{
			if ((offU[rowU] <= indexU) && (offU[rowU + 1] > indexU))
			break;
		}
		for ( ; indexB < sizeB; indexB++)
        {
			// Update rowB
			for ( ; rowB < size_offB; rowB++)
			{
				if ((offB[rowB] <= indexB) && (offB[rowB + 1] > indexB))
				break;
			}
			
			if (rowB > rowU)
			{   
				break; 
			}
			if (rowB < rowU)
			{   
				continue; 
			}			
			
			if (idxB[indexB] > idxU[indexU]) 
			{
				break; 
			}
			if ((rowB == rowU) && (idxB[indexB] == idxU[indexU]))
			{
				valU[indexU] = valB[indexB];
				FLAG = 1;
			}
        }
        if (FLAG == 0)
		{
			valU[indexU] = 0;
		}
	}

	long sum = 0;
	for(size_t i = 0; i < sizeU; i++)
	{
		sum += valU[i];
	}
	cout << sum << '\n';
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Time: " << elapsed.count() << " s\n";
}
