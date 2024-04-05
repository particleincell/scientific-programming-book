#ifndef __DATA_H
#define __DATA_H

/* storage containers used by the 2D heat equation solver */

#include <memory.h>
#include <vector>
#include <math.h>
#include <stdexcept>

struct World {
  World(int ni, int nj, double dx, double dy):
      ni{ni},nj{nj},nu{ni*nj},dx{dx},dy{dy} {}
      
  int U(int i, int j) {return j*ni+i;}
  
  const int ni;
  const int nj;
  const int nu;
  const double dx;
  const double dy;
  const double x0 = 0;  // hardcoded origin
  const double y0 = 0;
};



// matrix base class
class Matrix {
  public:
  Matrix (int nr): nr{nr} {}
  virtual double dotRow(int r,  const std::vector<double>&x) const = 0;
  virtual double& operator()(int i,int j)=0;
  virtual double operator()(int i, int j) const =0; 
  
  virtual ~Matrix() {}
  
  const int nr;   // number of rows
};

// a sparse 5-banded matrix
class FiveBandMatrix : public Matrix {
  public:
  // this constructor requires a second argument to set offset
  FiveBandMatrix(int nr, int di) : Matrix (nr), di{di} {
    a = newAndClear(nr);
    b = newAndClear(nr);
    c = newAndClear(nr);
    d = newAndClear(nr);
    e = newAndClear(nr);  
  }

  ~FiveBandMatrix() {
    delete[] a;   a = nullptr;
    delete[] b;   b = nullptr;
    delete[] c;   c = nullptr;
    delete[] d;   d = nullptr;
    delete[] e;   e = nullptr;
  }

  double& operator()(int r, int c)  {
    if (c-r==-di) return a[r];
    else if (c-r==-1) return b[r];
    else if (c-r==0) return this->c[r];  // to distinguish argument from member
    else if (c-r==1) return d[r];
    else if (c-r==di) return e[r];
    else throw std::runtime_error("Unsupported operation: " + std::to_string(r) + " " + std::to_string(c));
  }   

  double operator()(int r, int c) const {
    if (c-r==-di) return a[r];
    else if (c-r==-1) return b[r];
    else if (c-r==0) return this->c[r];
    else if (c-r==1) return d[r];
    else if (c-r==di) return e[r];
    else throw std::runtime_error("Unsupported operation!");
  }

  double dotRow(int r, const std::vector<double>&x) const {
    double sum = 0;
    
    if (a[r] && r-di>=0) sum += a[r]*x[r-di];
    if (b[r] && r-1>=0) sum += b[r]*x[r-1];
    sum += c[r]*x[r];
    if (d[r] && r+1<nr) sum += d[r]*x[r+1];
    if (e[r] && r+di<nr) sum += e[r]*x[r+di];
    return sum;   
  }

  protected:
  // helper function to allocate and clear an array
  static double *newAndClear(int nr) {
	double *p = new double[nr];
        memset(p,0,sizeof(double)*nr);
        return p;
  }

  double *a = nullptr;   // coefficients
  double *b = nullptr; 
  double *c = nullptr;   // main diagonal
  double *d = nullptr;
  double *e = nullptr;

  const int di;         // offset from a to b
};

// generic 2D data container
template<typename T>
class _Field {
  public:
  _Field(World &world, T def=0) : world{world} {
    data = new T*[world.ni];
    for (int i=0;i<world.ni;i++) 
      data[i] = new T[world.nj];
      
    //initialize
    for (int i=0;i<world.ni;i++)
      for (int j=0;j<world.nj;j++)
        data[i][j] = def;
  }
  
  ~_Field() {
    for (int i=0;i<world.ni;i++) delete[] data[i];
    delete[] data;
    data = nullptr;
  }

  // no bounds checking!
  T& operator()(int i,int j) {return data[i][j];}
  T operator()(int i,int j) const {return data[i][j];}

  World &world;     // world reference
  
  protected:
  T **data;         // 2D array

};

using Field = _Field<double>;
using FieldB = _Field<bool>;
using dvector = std::vector<double>;

#endif
