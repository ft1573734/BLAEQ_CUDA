#include <algorithm>
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <iostream>
#include <stdio.h>

class BLAEQ_Dimension {
public:
	cusparseSpMatDescr_t* P_Matrices;		// A list of prolongation matrices
	cusparseSpVecDescr_t* Coarsest_Mesh;	// Coarsest mesh
	cusparseHandle_t* cusparseHandle;
	double* Bandwidths;						// The bandwidths of each layer
	int Dimension;
	int L;
	int NUM_THREADS = 256;
	int N;
	int K;
	int MAX_COUNT_PER_COL = N / K;

	BLAEQ_Dimension(int dim, int K, int N, double* M, cusparseHandle_t* cusparseHandle);

	void BLAEQ_Generate_P_Matrices_Dimension(cusparseSpMatDescr_t** P_Matrices, cusparseSpVecDescr_t* coarsestMesh, double* original_mesh, cusparseHandle_t* cusparseHandle);

	void BLAEQ_Query_Dimension(double min, double max, cusparseSpVecDescr_t* input_result, cusparseSpVecDescr_t* output_result);
};