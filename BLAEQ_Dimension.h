#include <algorithm>
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <iostream>
#include <stdio.h>
#include "BLAEQ_CUDA_Kernel.h"
#include "debug.h"

#ifndef DEBUG
#define DEBUG false
#endif

#define BOOST_DISABLE_CURRENT_LOCATION

class BLAEQ_Dimension {
public:
	cusparseSpMatDescr_t* P_Matrices;		// A list of prolongation matrices
	cusparseSpVecDescr_t Coarsest_Mesh;	// Coarsest mesh
	cusparseHandle_t* cusparseHandle;

	double* Bandwidths;						// The bandwidths of each layer
	int Dimension;
	int L;
	int N;
	int K;
	int MAX_COUNT_PER_COL;
	int* sorted_idx;
	int* sorted_idx_device;
	BLAEQ_CUDA_Kernel kernel;

	BLAEQ_Dimension(int dim, int L, int K, int N, double* M, cusparseHandle_t* cusparseHandle);

	void BLAEQ_Generate_P_Matrices_Dimension(cusparseSpVecDescr_t* coarsestMesh, double* original_mesh, cusparseHandle_t* cusparseHandle);

	//void BLAEQ_Query_Dimension(double min, double max, cusparseSpVecDescr_t* output_result);

	void BLAEQ_Sort(double* original_V, int* original_idx, double** sorted_V, int** sorted_idx);

	void BLAEQ_Query_Dimension(double min, double max, int* res_size, int** restored_indices, double** data);

private:

	double _bandwidth_generator(double* vector, int size, int K);

	double _compute_range(double* vector, int size);

	void _generate_P_matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc_balanced, double* M_ip1_d, int M_ip1_length, cusparseHandle_t* cusparseHandle);

	void _balance_P_matrix(int MAX_BIN_SIZE, int M_index_count, int M_indptr_count, int* M_indptr, double* V_data, int* M_indptr_balanced, double* V_data_balanced);

	void _logical_in_range_judgement(double min, double max, cusparseSpVecDescr_t* input, cusparseSpVecDescr_t* output);

	void _BLAEQ_SpMSpV(cusparseSpMatDescr_t* P_matrix, cusparseSpVecDescr_t* input_vec, cusparseSpVecDescr_t* result_vec);
};
