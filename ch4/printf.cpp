#include <stdio.h>

int main() {
	const int n = 10;
	double x[n];            // n-sized array of floating point values
	for (int i=0;i<n;i++)   // loop for i=[0,...,n-1]
		x[i] = -1.0 + 2*i/(n-1);  // -1 plus value in [0,2]
	for (int i=0;i<n;i++) 
		printf("%.1f ",x[i]);  // %.1f shows one  numbers after the decimal point
	printf("\n");    // new line
	return 0;
}
