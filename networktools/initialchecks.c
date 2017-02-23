# include "initialchecks.h"

void check_for_type(PyArrayObject const * const input_array)
{
	if (PyArray_TYPE(input_array) != TYPE_INDEX)
	{
		PyErr_SetString(PyExc_TypeError, "Array must be of type double."); //TODO: dynamic error message
		Py_Exit(1);
	}
}

void check_dimension_match(PyArrayObject const * const input_array, unsigned int const dim_1, unsigned int const dim_2)
{
	if (((unsigned int) PyArray_NDIM(input_array) <= dim_1) || ((unsigned int) PyArray_NDIM(input_array) <= dim_2))
	{
		PyErr_SetString(PyExc_TypeError, "Array has too few dimensions.");
		Py_Exit(1);
	}
	
	if (PyArray_DIM(input_array, dim_1) != PyArray_DIM(input_array, dim_2))
	{
		char message[100];
		sprintf(message, "Array must be quadratic in dimensions %i and %i.", dim_1, dim_2);
		PyErr_SetString(PyExc_TypeError, message);
		Py_Exit(1);
	}
}

void check_cross_dimension_match(PyArrayObject const * const input_array_1, unsigned int const dim_1, PyArrayObject const * const input_array_2, unsigned int const dim_2)
{
	if (((unsigned int) PyArray_NDIM(input_array_1) <= dim_1) || ((unsigned int)  PyArray_NDIM(input_array_2) <= dim_2))
	{
		PyErr_SetString(PyExc_TypeError, "Array has too few dimensions.");
		Py_Exit(1);
	}
	
	if (PyArray_DIM(input_array_1, dim_1) != PyArray_DIM(input_array_2, dim_2))
	{
		PyErr_SetString(PyExc_TypeError, "Dimensions of input arrays do not match.");
		Py_Exit(1);
	}
}

void check_minimum_value(int const param, int const value, char const * const parname)
{
	if (param<value)
	{
		char message[1000];
		sprintf(message, "%s must be at least %i.", parname, value);
		PyErr_SetString(PyExc_ValueError, message);
		Py_Exit(1);
	}
}
