#include "BLAEQ_Dimension.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>


BLAEQ_Dimension::BLAEQ_Dimension(int input_dim, int input_K, int input_N, double* M, cusparseHandle_t* cusparseHandle) {
	double EPSILON = 0.00001;

	Dimension = input_dim;
	L = _compute_layer(input_N, input_K);
	//cudaMalloc(&P_Matrices, L * sizeof(cusparseSpMatDescr_t*));	//Allocating space for P_matrix pointers
	//cudaMalloc(&Bandwidths, L * sizeof(double));	//Allocating space for bandwidths

	P_Matrices = (cusparseSpMatDescr_t**)malloc(L * sizeof(cusparseSpMatDescr_t*));
	Bandwidths = (double*)malloc(L * sizeof(double));

	N = input_N;
	K = input_K;
	MAX_COUNT_PER_COL = N / K;
	kernel = BLAEQ_CUDA_Kernel(MAX_COUNT_PER_COL);

	int* idx = (int*)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++) {
		idx[i] = i;
		M[i] = M[i] + EPSILON;
	}
	double* sorted_M = (double*)malloc(N * sizeof(double));
	int* sorted_idx = (int*)malloc(N * sizeof(int));
	BLAEQ_Sort(M, idx, &sorted_M, &sorted_idx);
	

	BLAEQ_Generate_P_Matrices_Dimension(P_Matrices, &Coarsest_Mesh, sorted_M, cusparseHandle);
}

void BLAEQ_Dimension::BLAEQ_Sort(double* original_V, int* original_idx, double** sorted_V, int** sorted_idx) {

	// Wrap raw arrays with device_vector
	thrust::device_vector<double> d_original_V(original_V, original_V + N);
	thrust::device_vector<int> d_original_idx(original_idx, original_idx + N);

	// Sort keys and rearrange values
	thrust::sort_by_key(d_original_V.begin(), d_original_V.end(), d_original_idx.begin());

	// Copy sorted data back to host
	std::copy(d_original_V.begin(), d_original_V.end(), *sorted_V);
	std::copy(d_original_idx.begin(), d_original_idx.end(), *sorted_idx);
}

void BLAEQ_Dimension::BLAEQ_Generate_P_Matrices_Dimension(cusparseSpMatDescr_t** P_Matrices, cusparseSpVecDescr_t* coarsestMesh, double* original_mesh, cusparseHandle_t* cusparseHandle) {

	std::cout << "Generating Prolongation matrix for dimension " << Dimension << " ...";
	double* M_i_d = original_mesh;
	int N_i_d = N;

	////Copying mesh to DRAM, assuming DRAM space is sufficient
	//cudaError_t cudaStatus;
	//double* M_i_d_DRAM;
	//cudaStatus = cudaMalloc(&M_i_d_DRAM, N_i_d * sizeof(double));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	exit(EXIT_FAILURE);
	//}
	//cudaMemcpy(M_i_d_DRAM, M_i_d, N_i_d * sizeof(double), cudaMemcpyHostToDevice);
	//free(M_i_d);


	//Filling RAM variables

	for (int i = 0; i < L - 1; i++) {
		double bandwidth = _bandwidth_generator(M_i_d, N_i_d, K);
		Bandwidths[(L - 1) - i] = bandwidth; //Store bandwidths in reverse order so that the coarsest layer corresponds to Bandwidths[0], second layer corresponds to Bandwidths[1] and so forth.
		double* M_ip1_d;
		int N_ip1_d;

		double* balanced_M_ip1_d;
		int balanced_N_ip1_d;
		cusparseSpMatDescr_t tmp_P_matrix;
		cusparseSpMatDescr_t balanced_P_matrix;
		kernel.Generate_P_Matrix(M_i_d, N_i_d, bandwidth, &tmp_P_matrix, &M_ip1_d, &N_ip1_d, cusparseHandle);

		kernel.Balance_P_Matrix(tmp_P_matrix, &balanced_P_matrix, M_ip1_d, N_ip1_d, &balanced_M_ip1_d, &balanced_N_ip1_d);

		P_Matrices[(L - 1) - i] = &balanced_P_matrix;

		M_i_d = balanced_M_ip1_d;
		N_i_d = N_ip1_d;
	}



	int* coarsest_mesh_indices_device;
	cudaMalloc(&coarsest_mesh_indices_device, N_i_d * sizeof(int));
	int* coarsets_mesh_indices_host = (int*)malloc(N_i_d * sizeof(int));
	for (int i = 0; i < N_i_d; i++) {
		coarsets_mesh_indices_host[i] = i;
	}
	cudaMemcpy(coarsest_mesh_indices_device, coarsets_mesh_indices_host, N_i_d * sizeof(int), cudaMemcpyHostToDevice);

	double* coarsest_mesh_data_device;
	cudaMalloc(&coarsest_mesh_data_device, N_i_d * sizeof(double));
	cudaMemcpy(coarsest_mesh_data_device, M_i_d, N_i_d * sizeof(double), cudaMemcpyHostToDevice);

	cusparseSpVecDescr_t coarsestMesh_local;
	cusparseCreateSpVec(&(coarsestMesh_local), N_i_d, N_i_d, coarsest_mesh_indices_device, coarsest_mesh_data_device, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	*coarsestMesh = coarsestMesh_local;

}

void BLAEQ_Dimension::BLAEQ_Query_Dimension(double min, double max, cusparseSpVecDescr_t* result) {

	cusparseSpVecDescr_t* logical_result;
	cusparseSpVecDescr_t* this_layer;
	cusparseSpVecDescr_t* next_layer;

	this_layer = &Coarsest_Mesh;
	for (int i = 0; i < L; i++) {
		kernel.In_Range(min, max, Bandwidths[Dimension] / 2, this_layer, logical_result);
		cusparseSpMatDescr_t* P_matrix = P_Matrices[i];
		kernel.SpMSpV(P_matrix, logical_result, next_layer);
		this_layer = next_layer;

	}
	result = this_layer;
}


/*
*
	Below are tools necessary for BLAEQ. These functions should not be called outside of BLAEQ.
*
*/

int BLAEQ_Dimension::_compute_layer(int N, int k) {
	return log2(N) / log2(k) + 1;
}

double BLAEQ_Dimension::_bandwidth_generator(double* vector, int size, int K) {
	int bin_count = size / K;
	double bandwidth = _compute_range(vector, size)/bin_count;
	double epsilon = bandwidth / 1000;
	return bandwidth + epsilon;
}


double BLAEQ_Dimension::_compute_range(double* vector, int size) {
	double max_val = DBL_MIN;
	double min_val = DBL_MAX;
	for (int i = 0; i < size; i++) {
		if (*vector < min_val) {
			min_val = *vector;
		}
		if (*vector > max_val) {
			max_val = *vector;
		}
		vector++;
	}
	//return max_val - min_val;
	return max_val; //Setting the lowerbound to 0 manually, seems more logical.
}
/*
void BLAEQ_Dimension::_generate_P_matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc_balanced, double* M_ip1_d, int M_ip1_length, cusparseHandle_t* cusparseHandle) {
	//Initializing P_matrix inn COO format
	int P_row_count = M_i_d_length;
	int P_col_count = M_i_d_length;
	int P_nnz_count = M_i_d_length;
	cusparseSpMatDescr_t* P_matrix_coo;

	double* P_data;
	int* P_rows;
	int* P_cols;

	cudaMalloc(&P_data, P_nnz_count * sizeof(double));
	cudaMalloc(&P_rows, P_row_count * sizeof(int));
	cudaMalloc(&P_cols, P_col_count * sizeof(int));

	//int NUM_BLOCKS = (int)ceil(M_i_d_length / NUM_THREADS);
	CUDA_generate_P_matrix_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (M_i_d, M_i_d_length, bandwidth, P_data, P_rows, P_cols);
	//CUDA_generate_P_matrix(M_i_d, M_i_d_length, bandwidth, P_data, P_rows, P_cols, NUM_BLOCKS, NUM_THREADS);

	cusparseCreateCoo(P_matrix_coo, P_row_count, P_col_count, P_nnz_count, P_rows, P_cols, P_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	int P_index_count_csc = M_i_d_length;
	int P_indptr_count_csc = floor(*thrust::max_element(M_i_d, M_i_d + M_i_d_length) / bandwidth) + 1;
	int P_data_count_csc = M_i_d_length;

	//Constructing M_ip1_d
	double* M_ip1_d_local;
	cudaMalloc(&M_ip1_d_local, P_indptr_count_csc * sizeof(double));

	for (int i = 0; i < P_indptr_count_csc; i++) {
		M_ip1_d_local[i] = i * bandwidth + bandwidth / 2;
	}
	M_ip1_d = M_ip1_d_local;
	M_ip1_length = P_indptr_count_csc;

	//Initializing P_matrix in CSC format
	double* P_data_csc;
	int* P_index_csc;
	int* P_indptr_csc;

	cudaMalloc(&P_data_csc, P_data_count_csc * sizeof(double));
	cudaMalloc(&P_index_csc, P_index_count_csc * sizeof(int));
	cudaMalloc(&P_indptr_csc, P_indptr_count_csc * sizeof(int));

	//Note: Function 'cusparseXcoo2csr' can also be used for converting COO to csc, since it basically just merges indexes into indptrs, whose principle are identical for CSR & CSC.
	cusparseXcoo2csr(*cusparseHandle, P_cols, P_nnz_count, P_col_count, P_indptr_csc, CUSPARSE_INDEX_BASE_ZERO); //Converting COO to CSC


	int* P_indptr_balanced;
	double* M_ip1_d_local_balanced;
	_balance_P_matrix(MAX_COUNT_PER_COL, P_index_count_csc, P_indptr_count_csc, P_indptr_csc, M_ip1_d_local, P_indptr_balanced, M_ip1_d_local_balanced);

	cudaFree(M_ip1_d_local);
	cudaFree(P_indptr_csc);

	cusparseCreateCsc(P_matrix_csc_balanced, P_row_count, P_col_count, P_nnz_count, P_indptr_balanced, P_index_csc, P_data_csc, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

}


void BLAEQ_Dimension::_logical_in_range_judgement(double min, double max, cusparseSpVecDescr_t *input, cusparseSpVecDescr_t* output) {
	void* index;
	void* data;
	int64_t* size;
	int64_t* nnz;
	cusparseIndexType_t* index_type;
	cusparseIndexBase_t* index_base;
	cudaDataType* data_type;

	cusparseSpVecGet(*input, size, nnz, & index, &data, index_type, index_base, data_type);



	int* tmp_result_indexes;
	double* tmp_result_data;
	cudaMalloc(&tmp_result_indexes, sizeof(int) * *size);
	cudaMalloc(&tmp_result_data, sizeof(double) * *size);
	// CUDA_in_range_kernel(double q_min, double q_max, double relaxation, double* data, double* indices, int size, double* result_data, int* result_indices)

	CUDA_in_range_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (min, max, Bandwidths[Dimension] / 2, (double*) data, (int*) index, *size, tmp_result_data, tmp_result_indexes);
	//CUDA_in_range(min, max, Bandwidths[Dimension] / 2, (double*)data, (int*)index, *size, tmp_result_data, tmp_result_indexes, NUM_BLOCKS, NUM_THREADS);

	int* index_end_ptr = thrust::remove(tmp_result_indexes, tmp_result_indexes + *size, 0);
	double* data_end_ptr = thrust::remove(tmp_result_data, tmp_result_data + *size, 0.0);

	int64_t nnz_size = index_end_ptr - tmp_result_indexes;

	cusparseCreateSpVec(output, *size, nnz_size, tmp_result_indexes, tmp_result_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cudaFree(input);
}


*/