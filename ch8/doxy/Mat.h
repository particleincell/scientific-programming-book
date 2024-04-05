#ifndef _MAT_H
#define _MAT_H

#include <vector>
#include <stdexcept>
#include "Vec.h"

//! matrix base class, specifies the interface to be implemented
class Mat {
public: 
  Mat(size_t nu) : nu{nu} {}
  virtual double& operator()(size_t i, size_t j) = 0;
  virtual double operator()(size_t i, size_t j) const = 0;
  virtual double dot(const std::vector<double> &x, size_t r) const = 0;
  virtual size_t memsize() = 0;  //!< matrix size in bytes
  const size_t nu;		//!< number of rows (unknowns)
};


/* dense matrix, only good for small matrixes!*/
class DenseMat : public Mat {
public:

	DenseMat(size_t nu) : Mat{nu} {
		// memory for coefficients
		a = new double*[nu];
		for (size_t i=0;i<nu;i++)
			a[i] = new double[nu];

		// initialize all to zero
		for (size_t i=0;i<nu;i++)
          for (size_t j=0;j<nu;j++)
            a[i][j] = 0;
	}

	~DenseMat() {
		for (size_t i=0;i<nu;i++)
			delete[] a[i];
		delete[] a;
	}

	// returns reference to the A[i,j] coefficient in the full matrix
	double& operator()(size_t i, size_t j) {return a[i][j];}
	double operator()(size_t i, size_t j) const {return a[i][j];}

	// returns dot product of matrix row r with vector x
	double dot(const std::vector<double> &x, size_t r) const {
		double sum = 0;
		for (size_t c=0;c<nu;c++)
			sum+=a[r][c]*x[c];
		return sum;
	}

   // return memory size in bytes
   size_t memsize() {return nu*nu*sizeof(double);}

protected:
	double **a;		//coefficients, [nu][nu]
};


/* sparse matrix with up to 7 non-zero values per row*/
class SparseMat : public Mat {
public:

	SparseMat(size_t nu) : Mat{nu} {

		// memory for coefficients
		a = new double*[nu];
		for (size_t i=0;i<nu;i++)
			a[i] = new double[max_vals];

        // memory for column indexes
		c = new int*[nu];
			for (size_t i=0;i<nu;i++)
			c[i] = new int[max_vals];

		// clear data, column set to -1 are not set
		for (size_t i=0;i<nu;i++)
			for (size_t j=0;j<max_vals;j++) {
				c[i][j] = -1;  
			    a[i][j] = 0;   
			}
	}

	~SparseMat() {
		for (size_t i=0;i<nu;i++)
			delete[] a[i];
		delete[] a;

		for (size_t i=0;i<nu;i++)
			delete[] c[i];
		delete[] c;
		
	}

	// returns reference to the A[i,j] coefficient in the full matrix
	double& operator()(size_t i, size_t j) {
		//search for the sparse column corresponding to full matrix column j
		for (int v=0;v<max_vals;v++) {
			//did we reach an empty slot? If so, make it correspond to column j
			if (c[i][v]<0) c[i][v] = j;

			// does this sparse column map to j? If so, return the coeff
			if (c[i][v]==j) return a[i][v];
		}

		//getting here implies that all max_val slots are already occupied by columns other than j
		std::runtime_error("Sparse matrix too small!");
	}

	// identical to the function above but for read-only access
	double operator()(size_t i, size_t j) const {
		//search for the sparse column corresponding to full matrix column j
		for (int v=0;v<max_vals;v++) {
			//did we reach an empty slot? If so, make it correspond to column j
			if (c[i][v]<0) c[i][v] = j;

			// does this sparse column map to j? If so, return the coeff
			if (c[i][v]==j) return a[i][v];
		}

		//getting here implies that all max_val slots are already occupied by columns other than j
		std::runtime_error("Sparse matrix too small!");
	}

	// returns dot product of matrix row r with vector x
	double dot(const std::vector<double> &x, size_t r) const {
		double sum = 0;
		// loop up to max_vals time and until c[r][v] becomes negative
		for (size_t v=0;v<max_vals && c[r][v]>=0;v++)
			sum+=a[r][v]*x[c[r][v]];		// c[v] is effective the "j" in full matrix
		return sum;
	}

   // return memory size in bytes
   size_t memsize() {return nu*max_vals*sizeof(double);}

protected:
    static constexpr size_t max_vals = 7;   //!<max non-zeros per row
	double **a;		//!<coefficients, [nu][7]
    int **c;        //!<columns in full matrix, [nu][7], -1 if not set
};



#endif
