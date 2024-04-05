#include<iostream>
#include<thread>
#include<mutex>

using namespace std;

mutex mtx;

void doWork(int *counter) {
  //unique_lock<mutex> lock(mtx);   //or mtx.lock();
  *counter += 1; 
}

int main() {
 int s;
 int bad = 0;
 for (s=0;s<1000000;s++) {
   int counter = 0;
   thread t1(doWork, &counter);
   thread t2(doWork, &counter);

   t1.join();
   t2.join();
   if (counter!=2) {cout<<"*"<<flush;bad++;}
 }	 
 
 cout<<"\nWe got "<<bad<<" bad values out of "<<s<<" (or "<<100*bad/(double)s<<" %)"<<endl;
 return 0;
}
