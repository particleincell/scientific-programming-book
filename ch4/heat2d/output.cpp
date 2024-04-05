#include <fstream>
#include "output.h"
#include "data.h"

using namespace std;

// saves fields using VTK ImageData format
void Output::saveVTI(map<string,Field*> fields, FieldB &fixed) {
  ofstream out("heat.vti");   // output file

  World &world = fixed.world;
  
  out<<"<VTKFile type=\"ImageData\">\n";
  out<<"<ImageData WholeExtent=\"0 "<<world.ni-1<<" 0 "<<world.nj-1<<" 0 "<<0<<"\""; 
  out<<" Origin=\""<<world.x0<<" "<<world.y0<<" "<<0<<"\"";
  out<<" Spacing=\""<<world.dx<<" " <<world.dy<<" "<<0<<"\">\n";
  out<<"<Piece Extent=\"0 "<<world.ni-1<<" 0 "<<world.nj-1<<" 0 "<<0<<"\">\n"; 
  
  out<<"<PointData>\n";
  
  out<<"<DataArray Name=\"fixed\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Int32\">\n";
  for (int j=0;j<world.nj;j++) 
    for (int i=0;i<world.ni;i++) out<<(fixed(i,j)?1:0)<<" ";  // convert bool to 1 or 0
  out<<"\n</DataArray>\n";
  
  for (std::pair<string,Field*> pair:fields) {
    Field &F = *pair.second;
    out<<"<DataArray Name=\""<<pair.first<<"\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float64\">\n";
    for (int j=0;j<world.nj;j++) 
      for (int i=0;i<world.ni;i++) out<<F(i,j)<<" ";
    out<<"\n</DataArray>\n";
  }
  
  out<<"</PointData>\n";
  out<<"</Piece>\n";
  out<<"</ImageData>\n";
  out<<"</VTKFile>\n";
}
