# include "euclid.h"
# include <math.h>

DataType distance(
	PyArrayObject const * const input_networks_1,
	unsigned int const i,
	PyArrayObject const * const input_networks_2,
	unsigned int const j
)
{
	DataType sum = 0;
	for (unsigned int k=0; k<PyArray_DIM(input_networks_1,1); k++)
		for (unsigned int l=0; l<k; l++)
		{
			DataType diff = * (DataType *) PyArray_GETPTR3(input_networks_1, i, k, l) - * (DataType *) PyArray_GETPTR3(input_networks_2, j, k, l);
			sum += diff*diff;
		}
	return sqrt(sum);
}

void calculate_distances(
	PyArrayObject const * const input_networks_1,
	PyArrayObject const * const input_networks_2,
	PyArrayObject const * const distances
)
{
	for (unsigned int i=0; i<PyArray_DIM(input_networks_1,0); i++)
		for (unsigned int j=0; j<PyArray_DIM(input_networks_2,0); j++)
			* (DataType *) PyArray_GETPTR2(distances, i, j) = distance(input_networks_1, i, input_networks_2, j);
}
