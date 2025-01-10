#include "BLAEQ_CUDA_Kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <stdio.h>
#include <math.h>



/*
	CUDA Kernel Functions
*/
__global__ void CUDA_in_range_kernel(double q_min, double q_max, double relaxation, double* data, int* indices, int size, double* result_data, int* result_indices) {
	// int i = threadIdx.x;
	// c[i] = a[i] + b[i];
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < size) {
		if (q_min <= data[t_idx] <= q_max) {
			result_data[t_idx] = data[t_idx];
			result_indices[t_idx] = indices[t_idx];
		}
		else {
			result_data[t_idx] = 0.0;
			result_indices[t_idx] = 0;
		}
	}
}

__global__ void CUDA_generate_P_matrix_kernel(double* M_i_d, int M_i_d_length, double bandwidth, double* data, int* row, int* col) {
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < M_i_d_length) {
		int bin_index = floor(M_i_d[t_idx] / bandwidth);
		col[t_idx] = t_idx;
		row[t_idx] = bin_index;
		data[t_idx] = bandwidth * bin_index + bandwidth / 2;
	}
}


__global__ void CUDA_BLAEQ_SpMSpV_kernel(int64_t* P_row_count, int64_t* P_col_count, int64_t* P_nnz, int MAX_COL_SIZE, double* P_data, int* P_indexes, int* P_indptr, int64_t* V_nnz, double* V_data, int* V_indexes, double* Res_data, int* Res_indexes) {

	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < *P_col_count) {
		int tmp_col = V_indexes[t_idx];
		int P_start_index = P_indptr[tmp_col];
		int P_end_index = P_indptr[tmp_col + 1];
		int tmp_col_size = P_end_index - P_start_index;
		int result_arr_start_index = t_idx * MAX_COL_SIZE;
		for (int i = 0; i < tmp_col_size; i++) {
			Res_data[result_arr_start_index + i] = P_data[P_start_index + i] * V_data[t_idx];
			Res_indexes[result_arr_start_index + i] = P_indexes[P_start_index + i];
		}
		for (int i = tmp_col_size; i < MAX_COL_SIZE; i++) {
			Res_data[result_arr_start_index + i] = 0.0;
			Res_indexes[result_arr_start_index + i] = 0;
		}
	}
}

BLAEQ_CUDA_Kernels::BLAEQ_CUDA_Kernels() {

}
BLAEQ_CUDA_Kernels::~BLAEQ_CUDA_Kernels() {

}

void BLAEQ_CUDA_Kernels::In_Range(double min, double max, cusparseSpVecDescr_t* input, cusparseSpVecDescr_t* output) {
	void* index;
	void* data;
	int64_t* size;
	int64_t* nnz;
	cusparseIndexType_t* index_type;
	cusparseIndexBase_t* index_base;
	cudaDataType* data_type;

	cusparseSpVecGet(*input, size, nnz, &index, &data, index_type, index_base, data_type);


	int* tmp_result_indexes;
	double* tmp_result_data;
	cudaMalloc(&tmp_result_indexes, sizeof(int) * *size);
	cudaMalloc(&tmp_result_data, sizeof(double) * *size);
	// CUDA_in_range_kernel(double q_min, double q_max, double relaxation, double* data, double* indices, int size, double* result_data, int* result_indices)

	CUDA_in_range_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (min, max, Bandwidths[Dimension] / 2, (double*)data, (int*)index, *size, tmp_result_data, tmp_result_indexes);
	//CUDA_in_range(min, max, Bandwidths[Dimension] / 2, (double*)data, (int*)index, *size, tmp_result_data, tmp_result_indexes, NUM_BLOCKS, NUM_THREADS);

	int* index_end_ptr = thrust::remove(tmp_result_indexes, tmp_result_indexes + *size, 0);
	double* data_end_ptr = thrust::remove(tmp_result_data, tmp_result_data + *size, 0.0);

	int64_t nnz_size = index_end_ptr - tmp_result_indexes;

	cusparseCreateSpVec(output, *size, nnz_size, tmp_result_indexes, tmp_result_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cudaFree(input);
}


void BLAEQ_CUDA_Kernels::Generate_P_Matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc_balanced, double* M_ip1_d, int M_ip1_length, cusparseHandle_t* cusparseHandle) {
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
	CUDA_generate_P_matrix_kernel << <NUM_BLOCKS, NUM_THREADS >> > (M_i_d, M_i_d_length, bandwidth, P_data, P_rows, P_cols);
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

}

//void BLAEQ_CUDA_Kernels::Balance_P_Matrix(int MAX_BIN_SIZE, int M_index_count, int M_indptr_count, int* M_indptr, double* V_data, int* M_indptr_balanced, double* V_data_balanced) {
void BLAEQ_CUDA_Kernels::Balance_P_Matrix(cusparseSpMatDescr_t original_P_matrix, cusparseSpMatDescr_t balanced_P_matrix, cusparseSpVecDescr_t original_V, cusparseSpVecDescr_t balanced_V,int MAX_BIN_SIZE) {
	//The code below is used for balancing the P matrix, such that the largest column size does not exceed MAX_BIN_SIZE.
	/*
		This part of the logic is implemented as follows:
		1.	Copy the first element into balanced_P directly;
		2.	For later elements:
		3.		Compare the temporary element with its previous element;
		4.		If the gap is smaller than MAX_BIN_SIZE, write temporary element into balanced_P directly;
		5.		If the gap is larger, write prev + MAX_BIN_SIZE into balanced_P;
		6.		Update prev = prev + MAX_BIN_SIZE, offset += 1;
		7.		Goto step 3;
		8.	Repeat 1-7 until balanced P indptr is built.
		NOTE: the indexes and data of P is identical to balanced_P, no change is required.
	*/
	int64_t* row_count;
	int64_t* col_count;
	int64_t* nnz;
	void** indptr;
	void** indexes;
	void** data;
	cusparseIndexType_t* indptr_type;
	cusparseIndexType_t* indexes_type;
	cusparseIndexBase_t* idxBase;
	cudaDataType* data_type;

	cusparseCscGet(original_P_matrix, row_count, col_count, nnz, indptr, indexes, data, indptr_type, indexes_type, idxBase, data_type);

	int* balanced_indptr_buffer;
	double* balancd_V_data_buffer;
	int balanced_bin_count_upperbound = *col_count + *nnz / MAX_BIN_SIZE;

	cudaMalloc(&balanced_indptr_buffer, balanced_bin_count_upperbound * sizeof(int));
	cudaMalloc(&balancd_V_data_buffer, balanced_bin_count_upperbound * sizeof(double));

	//int balanced_P_col_count_csc = *col_count;
	int balanced_array_offset = 0;
	for (int i = 0; i < *col_count + 1; i++) {
		int tmp_csc_index = (int)indptr_type[i];
		int prev_csc_index = 0;
		if (i == 0) {
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer[i + balanced_array_offset] = V_data[i];
		}
		else if (tmp_csc_index - prev_csc_index <= MAX_BIN_SIZE) {
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer[i + balanced_array_offset] = V_data[i];
		}
		else if (tmp_csc_index - prev_csc_index > MAX_BIN_SIZE) {
			while (tmp_csc_index - prev_csc_index > MAX_BIN_SIZE) {
				balanced_indptr_buffer[i + balanced_array_offset] = prev_csc_index + MAX_BIN_SIZE;

				balancd_V_data_buffer[i + balanced_array_offset] = V_data[i];

				balanced_array_offset += 1;
				prev_csc_index += MAX_COUNT_PER_COL;
			}
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;
		}
		else {
			std::cerr << "WTF??? The program should never reach here. Error when calling function _generate_P_matrix()." << std::endl;
		}
	}
	int balanced_M_indptr_count = M_indptr_count + balanced_array_offset;

	int* result_balanced_indptr;
	double* result_V_data;


	cudaMalloc(&result_balanced_indptr, balanced_M_indptr_count * sizeof(int));
	cudaMalloc(&result_V_data, (balanced_M_indptr_count - 1) * sizeof(double)); //The '-1' operator is required since |indptr| = |col| + 1, and |V| = |col|.

	for (int i = 0; i < balanced_M_indptr_count; i++) {
		result_balanced_indptr[i] = balanced_indptr_buffer[i];
	}
	for (int i = 0; i < balanced_M_indptr_count - 1; i++) {
		result_V_data[i] = balancd_V_data_buffer[i];
	}

	cudaFree(&balanced_indptr_buffer);
	cudaFree(&balancd_V_data_buffer);
}


void BLAEQ_CUDA_Kernels::SpMSpV(cusparseSpMatDescr_t* P_matrix, cusparseSpVecDescr_t* input_vec, cusparseSpVecDescr_t* result_vec) {

	int64_t* row_count;
	int64_t* col_count;
	int64_t* nnz_count;
	void** indptr;
	void** indexes;
	void** data;
	cusparseIndexType_t* indptr_type;
	cusparseIndexType_t* indexes_type;
	cusparseIndexBase_t* idx_base;
	cudaDataType* dataType;
	cusparseCscGet(*P_matrix, row_count, col_count, nnz_count, indptr, indexes, data, indptr_type, indexes_type, idx_base, dataType);


	void** vec_indexes;
	void** vec_data;
	int64_t* vec_size;
	int64_t* vec_nnz;
	cusparseIndexType_t* vec_index_type;
	cusparseIndexBase_t* vec_index_base;
	cudaDataType* vec_data_type;

	cusparseSpVecGet(*input_vec, vec_size, vec_nnz, vec_indexes, vec_data, vec_index_type, vec_index_base, vec_data_type);

	int* res_indexes;
	double* res_data;

	int raw_result_vec_size = MAX_COUNT_PER_COL * *col_count;

	cudaMalloc(&res_data, raw_result_vec_size * sizeof(double));
	cudaMalloc(&res_indexes, raw_result_vec_size * sizeof(int));

	CUDA_BLAEQ_SpMSpV_kernel << <NUM_BLOCKS, NUM_THREADS >> > (row_count, col_count, nnz_count, MAX_COUNT_PER_COL, (double*)data, (int*)indexes, (int*)indptr, vec_nnz, (double*)vec_data, (int*)vec_indexes, res_data, res_indexes);
	//CUDA_BLAEQ_SpMSpV(row_count, col_count, nnz_count, MAX_COUNT_PER_COL, (double*)data, (int*)indexes, (int*)indptr, vec_nnz, (double*)vec_data, (int*)vec_indexes, res_data, res_indexes, NUM_BLOCKS, NUM_THREADS);

	int* cleaned_indexes = thrust::remove(res_indexes, res_indexes + raw_result_vec_size, 0);
	double* cleaned_data = thrust::remove(res_data, res_data + raw_result_vec_size, 0.0);

	if (cleaned_indexes - res_indexes != cleaned_data - res_data) {
		std::cerr << "WTF" << std::endl;
	}

	int cleaned_res_size = cleaned_indexes - res_indexes;
	cusparseCreateSpVec(result_vec, *row_count, cleaned_res_size, cleaned_indexes, cleaned_data, *vec_index_type, *vec_index_base, *vec_data_type);

	cudaFree(vec_indexes);
	cudaFree(vec_data);
}