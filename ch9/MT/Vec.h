#ifndef _VEC_H
#define _VEC_H

#include <ostream>

// template version of a 3 component vector
template<typename T>
struct _vec3{
  //constructor with an initializer list
  _vec3(T x, T y, T z) : d{x,y,z} {}
  _vec3() : _vec3(0,0,0) {}     // default constructor
  void operator+= (const _vec3<T> &o) {d[0]+=o.d[0]; d[1]+=o.d[1]; d[2]+=o.d[2];}
  void operator-= (const _vec3<T> &o) {d[0]-=o.d[0]; d[1]-=o.d[1]; d[2]-=o.d[2];}
  T &operator[] (int i) {return d[i];}
  T operator[] (int i) const {return d[i];}   // read only getter
  friend _vec3<T> operator*(const _vec3<T> &a, double s) {return _vec3<T>(a.d[0]*s,a.d[1]*s,a.d[2]*s);}
  friend _vec3<T> operator*(double s, const _vec3<T> &a) {return _vec3<T>(a.d[0]*s,a.d[1]*s,a.d[2]*s);}
  friend _vec3<T> operator-(const _vec3<T> &a, double s) {return _vec3<T>(a.d[0]-s,a.d[1]-s,a.d[2]-s);}

  friend std::ostream& operator<<(std::ostream &out, const _vec3<T> &o) { out<<o.d[0]<<" "<<o.d[1]<<" "<<o.d[2];return out; }

 protected:
  T d[3];
};

//define "nicknames"
using double3 = _vec3<double>;
using int3 = _vec3<int>;

#endif
