/*
Neural network example for Scientific Computing textbook
uses a 1-2-2-1 neural network to classify numbers in [0,1) as smaller or larger than 0.5
Written by L. Brieda
*/
#include <iostream>
#include <math.h>
#include <vector>
#include <random>
using namespace std;

/*object for sampling random numbers*/
class Rnd {
public:
	//constructor: set initial random seed and distribution limits
	Rnd(): mt_gen{std::random_device()()}, rnd_dist{0,1.0} {}
	double operator() () {return rnd_dist(mt_gen);}
protected:
	std::mt19937 mt_gen;	    //random number generator
	std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
};

Rnd rnd;		

template<int NL, int MN>
struct NN {
  double w[NL][MN][MN];
  double b[NL][MN];
  double x[NL][MN];    // neuron value before squishing
  double a[NL][MN];    // neuron value after squishing
  
  //error sums
  double dE_dw[NL][MN][MN];
  double dE_db[NL][MN];
  
  int num_neurons[NL];
  int num_layers = NL;  
  
  // evaluates the output layer using currently set weights and input activations
  void evaluate(double (*squish)(double)) {
    for (int l=1;l<NL;l++) {
      for (int n=0;n<num_neurons[l];n++) {
        x[l][n] = 0;
        for (int m=0;m<num_neurons[l-1];m++) {
          x[l][n] += w[l][n][m]*a[l-1][m];
        }
        x[l][n] += b[l][n];		   
        a[l][n] = squish(x[l][n]);     
      }
    }     // for layers
  }  // evaluate
};

int main() {
  auto eval = [](double x) {return x>=0.5;};
  
  // create training data
  std::vector<std::pair<double,double>> training_data;
  
  for (size_t i=0;i<10000;i++) {
  	double x = rnd();
	training_data.emplace_back(x,eval(x));
  }
  
  // print training data
  cout<<"--- TRAINING DATA ---"<<endl;
  for (size_t n=0;n<10;n++) cout<<training_data[n].first<<" -> "<<training_data[n].second<<endl;
  
  // build neural net: 1->2->2->1
  const int NL = 4;
  NN<4,2> nn;
  nn.num_neurons[0] = 1;
  nn.num_neurons[1] = 2;
  nn.num_neurons[2] = 2;
  nn.num_neurons[3] = 1;
  
  // generate random weights  
  for (int l=1;l<NL;l++) 
   for (int n=0;n<nn.num_neurons[l];n++) {
    nn.b[l][n] = 0;
    for (int m=0;m<nn.num_neurons[l-1];m++) {
      nn.w[l][n][m] = -1 + 2*rnd();
    }   
  }

  // set squishing function (logistic)  
  auto squish = [](double x)->double{return 1/(1+exp(-x));};
  auto dsf = [](double a)->double{return a*(1-a);};  // derivative, a=f(x)
  
  double eta = 10; 		// learning rate
  constexpr int max_epochs = 500;
  double E_initial = 1e66;
  
  // start training
  for (int epoch = 0; epoch<max_epochs; epoch++) {
    double E_sum = 0;
	  
    // clear error sums
    for (int l=1;l<NL;l++) 
      for (int n=0;n<nn.num_neurons[l];n++) {
        for (int m=0;m<nn.num_neurons[l-1];m++) 
          nn.dE_dw[l][n][m] = 0;
        nn.dE_db[l][n] = 0;
      }
	  
    // loop over training pairs
    for (std::pair<double,double> &pair:training_data) {
		nn.a[0][0] = pair.first;	// set input layer	
		nn.evaluate(squish);    // evaluate output layer
			
		double T[1] = {pair.second};   // true solution, keeping as an array for generality
		
		// local error
		double E = 0.5*(nn.a[NL-1][0]-T[0])*(nn.a[NL-1][0]-T[0]);
		E_sum += E;
		
		// hardcoded error derivatives per chain rule
		nn.dE_dw[3][0][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.a[2][0]);
		nn.dE_dw[3][0][1] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.a[2][1]);
		nn.dE_db[3][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (1);

		nn.dE_dw[2][0][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.w[3][0][0]) * dsf(nn.a[2][0]) * (nn.a[1][0]);
		nn.dE_dw[2][0][1] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.w[3][0][0]) * dsf(nn.a[2][0]) * (nn.a[1][1]);
		nn.dE_db[2][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.w[3][0][0]) * dsf(nn.a[2][0]) * (1);
			
		nn.dE_dw[2][1][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.w[3][0][1]) * dsf(nn.a[2][1]) * (nn.a[1][0]);
		nn.dE_dw[2][1][1] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.w[3][0][1]) * dsf(nn.a[2][1]) * (nn.a[1][1]);
		nn.dE_db[2][1] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (nn.w[3][0][1]) * dsf(nn.a[2][1]) * (1);
			
		double a = (nn.w[3][0][0]) * dsf(nn.a[2][0])*(nn.w[2][0][0]) * dsf(nn.a[1][0]);
		double b = (nn.w[3][0][1]) * dsf(nn.a[2][1])*(nn.w[2][1][0]) * dsf(nn.a[1][0]);
		nn.dE_dw[1][0][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (a+b) * dsf(nn.a[0][0])*(nn.a[0][0]);
		nn.dE_db[1][0] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (a+b) * dsf(nn.a[0][0])*(1);
			
		a = (nn.w[3][0][0]) * dsf(nn.a[2][0]) * (nn.w[2][0][1]) * dsf(nn.a[1][1]);
		b = (nn.w[3][0][1]) * dsf(nn.a[2][1]) * (nn.w[2][1][1]) * dsf(nn.a[1][1]);
		nn.dE_dw[1][0][1] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (a+b) * dsf(nn.a[0][1]) * (nn.a[0][1]);
		nn.dE_db[1][1] += (nn.a[3][0]-T[0]) * dsf(nn.a[3][0]) * (a+b) * dsf(nn.a[0][1]) * (1);	
	}  // for training data
	  
    // compute derivative averages over the test population
    for (int l=1;l<NL;l++) 
      for (int n=0;n<nn.num_neurons[l];n++) {
        for (int m=0;m<nn.num_neurons[l-1];m++)
          nn.dE_dw[l][n][m] /= training_data.size();
        nn.dE_db[l][n] /= training_data.size();
     }

    // apply correction to w and b terms
    for (int l=1;l<NL;l++)
      for (int n=0;n<nn.num_neurons[l];n++) {
        for (int m=0;m<nn.num_neurons[l-1];m++) 
          nn.w[l][n][m] -= eta*nn.dE_dw[l][n][m];        
        nn.b[l][n] -= eta*nn.dE_db[l][n];
      }  
      
    // display total error and check for convergence
    double E = E_sum/training_data.size();
    if (epoch>0) {
      double E_frac = E/E_initial;  // in percent
      cout<<epoch<<"\t"<<E_frac<<endl;
      if (E_frac<0.1) break;
    } else E_initial = E;
 
  }   // epoch
  
  // visualize computed weights and offsets
  cout<<"--- w: ---"<<endl;
   for (int l=1;l<NL;l++) 
      for (int n=0;n<nn.num_neurons[l];n++) {
        for (int m=0;m<nn.num_neurons[l-1];m++)
          cout<<"w"<<l<<n<<m<<"="<<nn.w[l][n][m]<<endl;
      }
    
  cout<<"--- b: ---"<<endl;
  for (int l=1;l<NL;l++) 
    for (int n=0;n<nn.num_neurons[l];n++) {
      cout<<"b"<<l<<n<<"="<<nn.b[l][n]<<endl;
    }
     
 // generate test data
  std::vector<std::pair<double,double>> test_data;
  for (size_t i=0;i<1000;i++) {
    double x = rnd();
    test_data.emplace_back(x,eval(x));
  }
  
  cout<<endl<<"--- TEST DATA ---"<<endl;
  int num_correct = 0;
  int item = 0;
  for (std::pair<double,double> &pair:test_data) {
      nn.a[0][0] = pair.first;	// set input layer
      nn.evaluate(squish);

      int o = (int)(nn.a[NL-1][0]+0.5);  // round to nearest (i.e. 0.6 -> 1)
      int T = (int)pair.second;
      if (o==T) num_correct++;	
      
      // display first 10 entries
      if (++item<10) cout<<nn.a[0][0]<<" -> "<<o<<" ("<<(o==T?'Y':'x')<<")"<<endl;
  }
      
  cout<<"Sample size: "<<test_data.size()<<endl;    
  cout<<"Correct: "<<num_correct/(double)test_data.size()*100<<"%"<<endl;
 
  return 0;
}
