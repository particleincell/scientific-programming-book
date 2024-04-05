#include <thread>
#include <iostream>
using namespace std;


void msg(const string &str) {
  cout<<str;
}

int main() 
{

  thread t1(msg,"Hello ");  // start 'msg("Hello")' in parallel
  thread t2(msg,"Sci ");
  thread t3(msg,"Comp ");
  thread t4(msg,"Readers");

  // need to wait for threads to finish running
  t1.join();  // block until t1 finishes
  t2.join();  // block until t2 finishes
  t3.join();  // block until t3 finishes
  t4.join();  // block until t4 finishes


  cout<<endl;

  
 return 0;
}
