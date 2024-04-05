#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <math.h>
using namespace std;

void outputField(int **data, int ni, int nj);

// support for complex numbers
struct Complex {
	double r;
	double i;
 
	Complex(double a, double b) : r(a), i(b) { }

	double magnitude2() {return r*r + i*i;}
	
	Complex operator*(const Complex& a) {
		return Complex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	Complex operator+(const Complex& a) {
		return Complex(r + a.r, i + a.i);
	}
 };
 
   	
// computes value of julia set at [i,j]*/
int juliaValue(int i, int j, int ni, int nj)  {
	double fi = -1.0 + 2.0*i/ni;	// fi = [-1:1)
	double fj = -1.0 + 2.0*j/nj;	// fj = [-1:1)
	
	Complex c(-0.8, 0.156);	// coefficient for the image
	Complex a(fi, fj);		// pixel pos as a complex number

	int k;					// iteration counter
	for( k = 0; k < 200; k++) {
 		a = a * a + c;
		if (a.magnitude2() > 1000) break;	// check for divergence
	}
	return k;				// return 
}
 

int main() {
	const int ni=4000;
	const int nj=4000;

	/*allocate memory for our domain*/
	int **julia=new int*[ni];
	for (int i=0;i<ni;i++) julia[i]=new int[nj];

	/*start timing*/
  	auto clock_start = chrono::high_resolution_clock::now();

	// compute pixels
	for (int i=0;i<ni;i++)
		for (int j=0;j<nj;j++)
			julia[i][j] = juliaValue(i,j,ni, nj);

	/*capture ending time*/
  	auto clock_end = chrono::high_resolution_clock::now();

	outputField(julia,ni,nj);
 	
	std::chrono::duration<float> delta = clock_end-clock_start;
    cout << "Simulation took "<<delta.count()<< "s\n";

	// free memory
	for (int i=0;i<ni;i++) delete[] julia[i];
	delete[] julia;	

	return 0;
}

/*saves output in VTK format*/
void outputField(int **data, int ni, int nj) {
	stringstream name;
	name<<"julia.vti";

    /*open output file*/
    ofstream out(name.str());
   	if (!out.is_open()) {cerr<<"Could not open "<<name.str()<<endl;return;}

	/*ImageData is vtk format for structured Cartesian meshes*/
	out<<"<VTKFile type=\"ImageData\">\n";
	out<<"<ImageData Origin=\""<<"0 0 0\" ";
	out<<"Spacing=\"1 1 1\" ";
	out<<"WholeExtent=\"0 "<<ni-1<<" 0 "<<nj-1<<" 0 0\">\n";
	
	/*output data stored on nodes (point data)*/
	out<<"<PointData>\n";
	
	/*potential, scalar*/
	out<<"<DataArray Name=\"julia\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Int32\">\n";
	for (int j=0;j<nj;j++)	{
		for (int i=0;i<ni;i++) out<<data[i][j]<<" ";
		out<<"\n";
	}
	out<<"</DataArray>\n";

	/*close out tags*/
	out<<"</PointData>\n";
	out<<"</ImageData>\n";
	out<<"</VTKFile>\n";
 	out.close();
}
	    
