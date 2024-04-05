#include <iostream>  // support for cout and cin

int main() {
  double val;
  std::cout<<"Enter a number: ";  
  std::cin>>val;  // read a floating point value from the keyboard
  std::cout<<"("<<val<<")^2 = "<<val*val<<std::endl; 
  return 0;
}
