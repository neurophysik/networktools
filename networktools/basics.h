// Datatypes, structs and functions for the Python interface.

# ifndef BASICS
# define BASICS

# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-pedantic"
# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# include <Python.h>
# include <numpy/arrayobject.h>
# pragma GCC diagnostic pop

# include <stdbool.h>
# include <gsl/gsl_rng.h>


// Data type used for network weights
typedef double DataType;
# define TYPE_INDEX NPY_DOUBLE

static inline PyObject * PyFloat_FromDataType(DataType x){
  return PyFloat_FromDouble(x);
}

// Abstraction layer for networks:
typedef PyArrayObject * Network;
static inline DataType get_weight(Network const N, unsigned int const i, unsigned int const j)
{
	return * (DataType *) PyArray_GETPTR2(N, i, j);
}

static inline void set_weight(Network const N, unsigned int const i, unsigned int const j, DataType const value)
{
	 * (DataType *) PyArray_GETPTR2(N, i, j) = value;
}


// Global random-number generator
gsl_rng * rng;

# endif
