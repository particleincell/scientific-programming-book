#include "Output.h"
#include "Matrix.h"

using namespace std;


void saveVTI(const string &file_name, World &world, map<string,vector<double>> &fields ) {
 	/* output vti file */
  	ofstream out("field.vti");

  	out<<"<VTKFile type=\"ImageData\">\n";
  	out<<"<ImageData WholeExtent=\"0 "<<world.ni-1<<" 0 "<<world.nj-1<<" 0 "<<0<<"\"";
  	out<<" Origin=\""<<world.x0<<" "<<world.y0<<" "<<0.0<<"\"";
  	out<<" Spacing=\""<<world.dx<<" " <<world.dy<<" "<<0.0<<"\">\n";
  	out<<"<Piece Extent=\"0 "<<world.ni-1<<" 0 "<<world.nj-1<<" 0 "<<0<<"\">\n";
  	out<<"<PointData>\n";


  	// iterate over every item in the map
  	for (auto pair:fields) {
  		string name = pair.first;
  		vector<double> &vec = pair.second;

  		out<<"<DataArray Name=\""<<name<<"\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float64\">\n";
 		for (int n=0;n<world.nn;n++) out<<vec[n]<<" ";
 		out<<"\n</DataArray>\n";

  	}

	out<<"</PointData>\n";
	out<<"</Piece>\n";
	out<<"</ImageData>\n";
	out<<"</VTKFile>\n";

}
