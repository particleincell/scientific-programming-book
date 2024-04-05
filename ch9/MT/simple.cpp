/*very simple threading example*/
#include <thread>
#include <iostream>
#include <sstream>

using namespace std;

//function to be executed from thread
void f(int i)
{
//	this_thread::sleep_for(chrono::milliseconds(5000));
	stringstream ss;
	ss<<"Hello from f("<<i<<")"<<endl;
	cout<<ss.str();
}


/*main function*/
int main()
{
	thread t1(f,1);
	thread t2(f,2);
		
	cout<<"Hello from main()"<<endl;

	/*wait for threads to finish*/
	t1.join();
	t2.join();

	return 0;
}
