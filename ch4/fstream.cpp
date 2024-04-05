#include <fstream>
using namespace std;
int main() {
	float x = 1.2;
	ofstream out("results.txt"); // open file for writing
	out<<"x = "<<x<<", x^2 = "<<x*x<<"\n"; // write formatted string
	return 0;
}
