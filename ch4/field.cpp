/*
ASTE-404 example of VTK image data ouput
*/

#include<iostream>  // screen output
#include<fstream>  // file output
#include<math.h>
using namespace std;
int main() {

  int ni = 21;   // number of nodes
  int nj = 21;
  int nk = 15;

  double x0 = -0.1;  // origin
  double y0 = -0.1;
  double z0 = 0.0;

  double dx = 0.01;  // cell spacing
  double dy = 0.01;
  double dz = 0.02;

  // allocate 1D memory block, we cover 3D arrays later
  double *data = new double[ni*nj*nk];

  // set data 
  for (int k=0;k<nk;k++)
    for (int j=0;j<nj;j++)
     for (int i=0;i<ni;i++) {
       double x = x0 + i*dx;
       double y = y0 + j*dy;
       double z = z0 + k*dz;
       int u = k*ni*nj + j*ni + i;
       data[u] = (x*x + y*y)*z;
     }
  
  /* output vti file */
  ofstream out("field.vti");

  out<<"<VTKFile type=\"ImageData\">\n";
  out<<"<ImageData WholeExtent=\"0 "<<ni-1<<" 0 "<<nj-1<<" 0 "<<nk-1<<"\""; 
  out<<" Origin=\""<<x0<<" "<<y0<<" "<<z0<<"\"";
  out<<" Spacing=\""<<dx<<" " <<dy<<" "<<dz<<"\">\n";
  out<<"<Piece Extent=\"0 "<<ni-1<<" 0 "<<nj-1<<" 0 "<<nk-1<<"\">\n"; 
  out<<"<PointData>\n";
  out<<"<DataArray Name=\"value\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float64\">\n";
  for (int u=0;u<ni*nj*nk;u++) out<<data[u]<<" ";	
  out<<"\n</DataArray>\n";
  out<<"</PointData>\n";
  out<<"</Piece>\n";
  out<<"</ImageData>\n";
  out<<"</VTKFile>\n";

  // free memory
  delete[] data;
  return 0;	// normal exit
}

