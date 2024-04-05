#include <iostream>
using namespace std;

void doA() {cout<<"A"<<endl;}
void doB() {cout<<"B"<<endl;}
void doC() {cout<<"C"<<endl;}
void doDefault() {cout<<"other"<<endl;}

enum Operation {OP_A, OP_B, OP_C, OTHER};
enum class OpPP {OP_A, OP_B, OP_C, OTHER};

int main() {
    Operation op = OP_A;
    
    switch (op) {
      case OP_A: doA(); break;
      case OP_B: doB(); break;
      case OP_C: doC(); break;
      default: doDefault(); break;      
    }

  OpPP op_pp = OpPP::OP_A;
  return 0;
}

