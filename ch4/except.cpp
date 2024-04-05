#include <iostream>
#include <exception>

using namespace std;

void fun(int x) {
  if (x<0) throw runtime_error("invalid value!");
}

int main() {

  try {
    fun(-1);
  }
  catch (const std::exception &e) { 
    cerr<<"Error occurred: "<<e.what()<<endl;
    return -1;
  }

  return 0;
}

