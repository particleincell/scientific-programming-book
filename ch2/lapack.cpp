#include <iostream>
#include <lapacke.h>
using namespace std;

int main(int argc, char *argv[])
{
	constexpr lapack_int n = 3;
    lapack_int info, nrhs = 1;
    double A[n*n] = {1,2,0,0,2,1,1,0,1};
    double b[n] = {0,1,2};
    int ipiv[n];

	// (d)ouble precision (ge)neral matrix (s)ol(v)e of A*x=b
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, n, ipiv, b, n);

	cout<<"b = [";
	for (int i=0;i<n;i++) cout<<" "<<b[i]<<" ";
	cout<<"]"<<endl;

    return 0;
}

