#include "measures.h"
#include <math.h>

DataType calculate_assortativity(PyArrayObject const * const weightMatrix){
	int n = PyArray_DIM(weightMatrix,1);  
	DataType strength[n];
	
	for (int i = 0 ; i < n ; i++){
		strength[i] = 0;
		for (int j = 0 ; j < n ; j++){
			strength[i] += * (DataType *) PyArray_GETPTR2(weightMatrix, i, j);
		}
	}
	
	DataType S1 = 0;
	DataType S2 = 0;
	DataType S3 = 0;
	
	for (int l = 0 ; l < n ; l++){
		S1 += strength[l];
		S2 += strength[l] * strength[l];
		S3 += strength[l] * strength[l] * strength[l];
	}
	
	
	DataType dummy = 0;
	
	for (int i = 0 ; i < n ; i++){
		for (int j = 0 ; j < i ; j++){
			dummy += * (DataType *) PyArray_GETPTR2(weightMatrix, i, j) * strength[i] * strength[j];
		}
	}
	
	return (2.0 * S1 * dummy - S2 * S2) /(DataType) (S1 * S3 - S2 * S2); 
}


//Calculate clustering coefficient for weighted undirected network
DataType calculate_clustering(PyArrayObject const * const weightMatrix){
	int n = PyArray_DIM(weightMatrix,1);   
	DataType cbrtWeights[n][n];
	
	for (int i = 0; i < n; i++){
		for (int j = 0; j < i ; j++){
			cbrtWeights[i][j] = cbrt( * (DataType *) PyArray_GETPTR2(weightMatrix, i, j) );
		}
	}
	
	DataType sum = 0.0;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < i ; j++){
			for (int k = 0; k < j ; k++){
				sum += cbrtWeights[i][j] * cbrtWeights[j][k] * cbrtWeights[i][k];
			}
		}
	}
	
	// /n because of global clustering coefficient *3.0 since every triangle is only seen once
 	return (sum * 3.0* 2.0 /(DataType) (n * (n - 1) * (n - 2))); 
}

//Calculate clustering coefficient for binary network 
DataType calculate_clustering_binary(PyArrayObject const * const A){
	int n = PyArray_DIM(A,1);  
	DataType degree[n];
	
	for (int i = 0 ; i < n ; i++){
		degree[i] = 0;
		for (int j = 0 ; j < n ; j++){
			degree[i] += * (DataType *) PyArray_GETPTR2(A, i, j);
		}
	}

	DataType localClusteringCoefficient[n];
	for (int i = 0; i < n; i++){
		localClusteringCoefficient[i] = 0.0;
		for (int j = 0; j < n ; j++){
			for (int k = 0; k < j ; k++){
				if (degree[i] == 0 || degree[i] == 1){
					localClusteringCoefficient[i] = 0.0;
				}
				else{
					localClusteringCoefficient[i] += 2.0 * (* (DataType *) PyArray_GETPTR2(A, i, j) * * (DataType *) PyArray_GETPTR2(A, j, k) * * (DataType *) PyArray_GETPTR2(A, i, k))/(DataType) (degree[i] * (degree[i] - 1.0));
				}
			}
		}
	}
	
	DataType GlobalClusteringCoefficient = 0;
	for (int i = 0; i < n; i++){
		GlobalClusteringCoefficient += localClusteringCoefficient[i];
	}
	GlobalClusteringCoefficient = GlobalClusteringCoefficient /(DataType) n;

	return GlobalClusteringCoefficient;
}


DataType calculate_path(PyArrayObject const * const weightMatrix){
	int n = PyArray_DIM(weightMatrix,1);    
	DataType distance[n][n];
	
	// Floyd-Warshall algorithm for calculating distance matrix of an undirected, weighted network
	DataType weight;
	for (int i = 0 ; i < n ; i++){
		for (int j = 0 ; j < i ; j++){
			weight =  * (DataType *) PyArray_GETPTR2(weightMatrix, i, j);
			if (weight == 0){
				distance[i][j] = distance[j][i] = INFINITY; 
			} else {
				distance[i][j] = distance[j][i] = 1.0 /(DataType) weight;
			}
		}
		distance[i][i] = 0.0;
	}
	
	for (int k = 0 ; k < n ; k++){
		for (int i = 0 ; i < n ; i++){
			for (int j = 0 ; j < i ; j++){
				if ( distance[i][j] > (distance[i][k] + distance[k][j]) ){
					distance[i][j] = distance[j][i] = distance[i][k] + distance[k][j];
				} 
			}
		}
	}
	
	// calculate average shortest path length 
	DataType sum = 0;
	for (int i = 0 ; i < n ; i++){
		for (int j = 0 ; j < i ; j++){
			sum += distance[i][j];
		}
	}
	
	return (2.0 * sum) / (n * (n - 1));
}

void calculate_distanceMatrix(PyArrayObject const * const weightMatrix, PyArrayObject const * const distance){
	int n = PyArray_DIM(weightMatrix,1);
	// Floyd-Warshall algorithm for calculating distance matrix of an undirected, weighted network
	DataType weight;
	for (int i = 0 ; i < n ; i++){
		for (int j = 0 ; j < i ; j++){
			weight =  * (DataType *) PyArray_GETPTR2(weightMatrix, i, j);
			if (weight == 0){
				* (DataType *) PyArray_GETPTR2(distance, i, j) = * (DataType *) PyArray_GETPTR2(distance, j, i) = INFINITY; 
			} else {
				* (DataType *) PyArray_GETPTR2(distance, i, j) = * (DataType *) PyArray_GETPTR2(distance, j, i) = 1.0 /(DataType) weight;
			}
		}
		* (DataType *) PyArray_GETPTR2(distance, i, i) = 0.0;
	}
	
	for (int k = 0 ; k < n ; k++){
		for (int i = 0 ; i < n ; i++){
			for (int j = 0 ; j < i ; j++){
				if ( * (DataType *) PyArray_GETPTR2(distance, i, j) > (* (DataType *) PyArray_GETPTR2(distance, i, k) + * (DataType *) PyArray_GETPTR2(distance, k, j)) ){
					* (DataType *) PyArray_GETPTR2(distance, i, j) = * (DataType *) PyArray_GETPTR2(distance, j, i) = * (DataType *) PyArray_GETPTR2(distance, i, k) + * (DataType *) PyArray_GETPTR2(distance, k, j);
				} 
			}
		}
	}		
}

