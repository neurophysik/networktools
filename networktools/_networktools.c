// The Python interface / module

# include "basics.h"
# include "initialchecks.h"
# include "surrogates.h"
# include "euclid.h"
# include "measures.h"

# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"

const char strengthsurrogates_docstring[] = "\n"
	"Generates strength-preserving surrogates from a weighted network using the method from Ansmann and Lehnertz, Phys. Rev. E 84, 026103 (2011).\n"
	"\n"
	"Parameters\n"
	"----------\n"
	"input_network: The network from which surrogates are to be generated.\n"
	"tree_steps: The number of steps in which new networks are obtained from each other in a binary-tree manner. If you cannot decide, choose a number between three and five.\n"
	"tree_step_size: The number of tetragon transformations in each tree step. A reasonable choice is to make this ten times the step_size.\n"
	"chain_length: The number of surrogates obtained from each of the chains hanging at the end of each branch of the tree.\n"
	"step_size: The number of tetragon transformations between adjacent surrogates in one chain. Use the strengthsurrogate_tester to optimally adjust this parameter.\n"
	"seed: The seed of the random-number generator employed.\n"
	"\n"
	"The number of surrogates generated is chain_length*2^tree_steps.\n"
	;

static PyObject * py_strengthsurrogates(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	unsigned int tree_steps;
	unsigned int tree_step_size;
	unsigned int chain_length;
	unsigned int step_size;
	unsigned long seed = 0;

	if (!PyArg_ParseTuple(args, "O!IIII|L", &PyArray_Type, &input_network, &tree_steps, &tree_step_size, &chain_length, &step_size, &seed))
		return NULL;

	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	check_minimum_value(tree_steps, 1, "The number of tree steps");
	check_minimum_value(chain_length, 1, "The chain length");

	// int number_surrogates = (2^tree_steps)*chain_length;
	unsigned int const number_surrogates = (1<<tree_steps)*chain_length;

	// initiate output array
	unsigned int n = PyArray_DIM(input_network, 0);
	npy_intp dims[3] = {number_surrogates, n, n};
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	PyArrayObject * output_networks = (PyArrayObject *)PyArray_SimpleNew(3, dims, TYPE_INDEX);
	# pragma GCC diagnostic pop
	if (output_networks == NULL)
	{
		PyErr_SetString(PyExc_MemoryError,"Could not allocate output array");
		return NULL;
	}

	// prepare output array
	for (unsigned int i=0; i<n; i++)
		for (unsigned int j=0; j<n; j++)
			* (DataType *) PyArray_GETPTR3(output_networks, 0, i, j) = * (DataType *) PyArray_GETPTR2(input_network, i, j);

	// set seed for random-number generator
	if (seed == 0)
		PyErr_Warn(PyExc_UserWarning, "Warning: Seed is zero. Results may not be repeatable by using the same seed.");
	gsl_rng_set(rng, seed);

	generate_strength_surrogates(output_networks, tree_steps, tree_step_size, chain_length, step_size);

	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	return (PyObject*) PyArray_Return(output_networks);
	# pragma GCC diagnostic pop
}

const char weightsurrogates_docstring[] = "\n"
	"Generates edge-weight-preserving surrogates from a network (see Ansmann and Lehnertz, Phys. Rev. E 84, 026103 (2011)).\n"
	"\n"
	"Parameters\n"
	"----------\n"
	"input_network: The network from which surrogates are to be generated.\n"
	"number_surrogates: The number of surrogates to be generated.\n"
	"seed: The seed of the random-number generator employed.\n"
	;

static PyObject * py_weightsurrogates(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	unsigned int number_surrogates;
	unsigned long seed = 0;
	
	if (!PyArg_ParseTuple(args, "O!I|L", &PyArray_Type, &input_network, &number_surrogates,  &seed))
		return NULL;
	
	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	
	// initiate output array
	unsigned int n = PyArray_DIM(input_network, 0);
	npy_intp dims[3] = {number_surrogates, n, n};
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	PyArrayObject * output_networks = (PyArrayObject *)PyArray_SimpleNew(3, dims, TYPE_INDEX);
	# pragma GCC diagnostic pop
	if (output_networks == NULL)
	{
		PyErr_SetString(PyExc_MemoryError,"Could not allocate output array");
		return NULL;
	}

	// prepare output array
	for (unsigned int k=0; k<number_surrogates; k++)
		for (unsigned int i=0; i<n; i++)
			for (unsigned int j=0; j<n; j++)
				* (DataType *) PyArray_GETPTR3(output_networks, k, i, j) = * (DataType *) PyArray_GETPTR2(input_network, i, j);

	// set seed for random-number generator
	if (seed == 0)
		PyErr_Warn(PyExc_UserWarning, "Warning: Seed is zero. Results may not be repeatable by using the same seed.");
	gsl_rng_set(rng, seed);	
		
	generate_weight_surrogates(input_network, output_networks);

	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	return (PyObject*) PyArray_Return(output_networks);
	# pragma GCC diagnostic pop
}



static PyObject * py_distances(PyObject *self, PyObject *args)
{
	PyArrayObject * input_networks_1;
	PyArrayObject * input_networks_2;
	
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &input_networks_1, &PyArray_Type, &input_networks_2))
		return NULL;

	check_for_type(input_networks_1);
	check_for_type(input_networks_2);
	check_dimension_match(input_networks_1, 1, 2);
	check_dimension_match(input_networks_2, 1, 2);
	check_cross_dimension_match(input_networks_1, 1, input_networks_2, 1);
	
	// initiate output array
	npy_intp dims[2] = {PyArray_DIM(input_networks_1, 0), PyArray_DIM(input_networks_2, 0)};
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	PyArrayObject * distances = (PyArrayObject *)PyArray_SimpleNew(2, dims, TYPE_INDEX);
	# pragma GCC diagnostic pop
	if (distances == NULL)
	{
		PyErr_SetString(PyExc_MemoryError,"Could not allocate output array");
		return NULL;
	}

	calculate_distances(input_networks_1, input_networks_2, distances);
	
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	return (PyObject*) distances;
	# pragma GCC diagnostic pop
}

const char assortativity_docstring[] = "\n"
	"Returns the average weighted assortativty of a network "
	"(a from Lehnertz et al, Physica D 267, 7–15 (2011))."
	;

static PyObject * py_assortativity(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_network))
		return NULL;

	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	
	return PyFloat_FromDataType(calculate_assortativity(input_network));
}

const char clustering_docstring[] = "\n"
	"Returns the average weighted clustering coefficient of a network "
	"(K from Ansmann and Lehnertz, Phys. Rev. E 84, 026103 (2011))."
	;

static PyObject * py_clustering(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_network))
		return NULL;

	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	
	return PyFloat_FromDataType(calculate_clustering(input_network));
}

static PyObject * py_clustering_binary(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_network))
		return NULL;

	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	
	return PyFloat_FromDataType(calculate_clustering_binary(input_network));
}

const char path_docstring[] = "\n"
	"Returns the average shortest weighted pathlength of a network, where the length of each edge is considered the inverse of its weight."
	;

static PyObject * py_path(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_network))
		return NULL;

	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	
	return PyFloat_FromDataType(calculate_path(input_network));
}

const char distanceMatrix_docstring[] = "\n"
	"Returns a matrix containing the shortest weighted pathlengths for all pairs of nodes in a network, where the length of each edge is considered the inverse of its weight."
	;

static PyObject * py_distanceMatrix(PyObject *self, PyObject *args)
{
	PyArrayObject * input_network;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_network))
		return NULL;

	check_for_type(input_network);
	check_dimension_match(input_network, 0, 1);
	
	// initiate output array
	npy_intp dims[2] = {PyArray_DIM(input_network, 0),PyArray_DIM(input_network, 1)};
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	PyArrayObject * distanceMatrix = (PyArrayObject *)PyArray_SimpleNew(2, dims, TYPE_INDEX);
	# pragma GCC diagnostic pop
	if (distanceMatrix == NULL)
	{
		PyErr_SetString(PyExc_MemoryError,"Could not allocate output array");
		return NULL;
	}

	calculate_distanceMatrix(input_network, distanceMatrix);
	
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	return (PyObject*) distanceMatrix;
	# pragma GCC diagnostic pop
}

# pragma GCC diagnostic pop

static PyMethodDef networktools_methods[] = {
	{"strengthsurrogates", py_strengthsurrogates, METH_VARARGS, strengthsurrogates_docstring},
	{"weightsurrogates", py_weightsurrogates, METH_VARARGS, weightsurrogates_docstring},
	{"distances", py_distances, METH_VARARGS, NULL},
	{"assortativity", py_assortativity, METH_VARARGS, assortativity_docstring},
	{"clustering", py_clustering, METH_VARARGS, clustering_docstring},
	{"clustering_binary", py_clustering_binary, METH_VARARGS, NULL},
	{"path", py_path, METH_VARARGS, path_docstring},
	{"distanceMatrix", py_distanceMatrix, METH_VARARGS, distanceMatrix_docstring},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef =
{
        PyModuleDef_HEAD_INIT,
        "_networktools",
        NULL,
        -1,
        networktools_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__networktools(void)
{
	PyObject * module = PyModule_Create(&moduledef);
	import_array();
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_default);
	return module;
}

#else

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init_networktools()
{
	Py_InitModule("_networktools", networktools_methods);
	import_array();
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_default);
}

#endif

