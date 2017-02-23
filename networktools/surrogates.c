# include "surrogates.h"

void copy_network(PyArrayObject const * const output_networks, unsigned int const k, unsigned int const l)
{
	for (unsigned int j=0; j<PyArray_DIM(output_networks, 2); j++)
		for (unsigned int i=0; i<j; i++)
			* (DataType *) PyArray_GETPTR3(output_networks, l, i, j) = * (DataType *) PyArray_GETPTR3(output_networks, k, i, j);
}

static inline DataType * get_pointer(PyArrayObject const * const output_networks, unsigned int const m, unsigned int const i, unsigned int const j)
{
	assert(i!=j);
	if (i<j)
		return (DataType *) PyArray_GETPTR3(output_networks, m, i, j);
	else
		return (DataType *) PyArray_GETPTR3(output_networks, m, j, i);
}

void transform(PyArrayObject const * const output_networks, unsigned int const m, unsigned int const steps)
{
	unsigned int n = PyArray_DIM(output_networks, 1);

	for (unsigned int a=0; a<steps; a++)
	{
		unsigned int i, j, k, l;

		// Generate random tetragon
		i = gsl_rng_uniform_int(rng,n);

		do
			j = gsl_rng_uniform_int(rng,n);
		while (i==j);

		do
			k = gsl_rng_uniform_int(rng,n);
		while (i==k || j==k);

		do
			l = gsl_rng_uniform_int(rng,n);
		while (i==l || j==l || k==l);

		// Get pointers
		DataType * v = get_pointer(output_networks, m, i, j);
		DataType * w = get_pointer(output_networks, m, j, k);
		DataType * x = get_pointer(output_networks, m, k, l);
		DataType * y = get_pointer(output_networks, m, l, i);

		// Actual transformation
		DataType min = -fmin(*w, *y);
		DataType max =  fmin(*v, *x);
		DataType shift = min + gsl_rng_uniform(rng) * (max - min);
		*v -= shift;
		*w += shift;
		*x -= shift;
		*y += shift;
	}
}

void copy_upper_triangle(PyArrayObject const * const output_networks)
{
	for (unsigned int k=0; k<PyArray_DIM(output_networks, 0); k++)
		for (unsigned int j=0; j<PyArray_DIM(output_networks, 2); j++)
			for (unsigned int i=0; i<j; i++)
				* (DataType *) PyArray_GETPTR3(output_networks, k, j, i) = * (DataType *) PyArray_GETPTR3(output_networks, k, i, j);
}



void generate_strength_surrogates(
	PyArrayObject const * const output_networks,
	unsigned int const tree_steps,
	unsigned int const tree_step_size,
	unsigned int const chain_length,
	unsigned int const step_size)
{
	assert(PyArray_DIM(output_networks, 0) == (1<<tree_steps)*chain_length);
	unsigned int network_stride = PyArray_DIM(output_networks, 0); // =number_surrogates;

	// Tree
	unsigned int number_branches = 1;
	for (unsigned int i=0; i<tree_steps; i++)
	{
		assert(network_stride%2==0);
		network_stride /= 2;
		
		for (unsigned int j=0; j<number_branches; j++)
		{
			transform(output_networks, (2*j)*network_stride, tree_step_size);
			copy_network(output_networks, (2*j)*network_stride, (2*j+1)*network_stride);
		}
		
		number_branches *= 2;
	}
	
	assert(number_branches==(unsigned int)(1<<tree_steps));
	
	// Chains
	for (unsigned int i=0; i<number_branches; i++)
		for (unsigned int j=0; j<chain_length-1; j++)
		{
			copy_network(output_networks, i*network_stride+j, i*network_stride+j+1);
			transform(output_networks, i*network_stride+j+1, step_size);
		}

	copy_upper_triangle(output_networks);

}

void generate_weight_surrogates(
	PyArrayObject const * const input_network,
	PyArrayObject const * const output_networks
){
	# ifdef DEBUG
	unsigned int steps = 0;
	# endif
	unsigned int n = PyArray_DIM(input_network,1);
	for (unsigned int j = 0; j < n; j++){
		for (unsigned int i = 0; i < j; i++){
			assert((j*(j-1))/2 + i == steps++);
			for (unsigned int k = 0; k < PyArray_DIM(output_networks,0) ; k++){
				unsigned int r = gsl_rng_uniform_int(rng,(j*(j-1))/2 + i + 1);
				unsigned int j_new = (unsigned int) (1 + sqrt(2*r+0.25) + sqrt(2*r+1.25)) * 0.5;
				unsigned int i_new = r - (j_new * (j_new-1)) / 2;
				* (DataType *) PyArray_GETPTR3(output_networks, k, i, j) = * (DataType *) PyArray_GETPTR3(output_networks, k, i_new, j_new);
				* (DataType *) PyArray_GETPTR3(output_networks, k, i_new, j_new) = * (DataType *) PyArray_GETPTR2(input_network, i, j);
			}
		}
	}
			
	copy_upper_triangle(output_networks);
}
