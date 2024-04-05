#include <iostream>
using namespace std;

struct Atom {
  double pos[3];
  double vel[3];
  double mass;
  double charge;
  double energy[3];
  
};

void calc(const int &a) {
  a = 4;
}
	 
int main() {

  cout<<sizeof(Atom)<<endl;
  cout<<sizeof(int)<<endl;
  return 0;
}
