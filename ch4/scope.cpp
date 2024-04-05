#include<iostream>
using namespace std;
int a = 0;         // global variables

void F() {
  	a = 1;          // assign to the global a
  	int a;          // local variable  
  	a = 2;          // assign to the local a
  	{
    	int a = 3;     // another local variable 
	  cout<<a<<endl; // prints 3
  	}
  	cout<<a<<endl;   // prints 2
}

int main() {
  	F();
  	cout<<a<<endl;   // prints 1
}
