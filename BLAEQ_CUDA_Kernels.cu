
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

void CUDA_in_range(double q_min, double q_max, double relaxation, double* data, int* indices, int size, double* result_data, int* result_indices, int NUM_BLOCKS, int NUM_THREADS) {
	CUDA_in_range_kernel <<<NUM_BLOCKS, NUM_THREADS>>> (double q_min, double q_max, double relaxation, double* data, int* indices, int size, double* result_data, int* result_indices);
}

void CUDA_generate_P_matrix(double* M_i_d, int M_i_d_length, double bandwidth, double* data, int* row, int* col, int NUM_BLOCKS, int NUM_THREADS) {
	CUDA_generate_P_matrix_kernel <<<NUM_BLOCKS, NUM_THREADS>>> (double* M_i_d, int M_i_d_length, double bandwidth, double* data, int* row, int* col);
}

void CUDA_BLAEQ_SpMSpV(int64_t* P_row_count, int64_t* P_col_count, int64_t* P_nnz, int MAX_COL_SIZE, double* P_data, int* P_indexes, int* P_indptr, int64_t* V_nnz, double* V_data, int* V_indexes, double* Res_data, int* Res_indexes, int NUM_BLOCKS, int NUM_THREADS) {
	CUDA_BLAEQ_SpMSpV_kernel <<<NUM_BLOCKS, NUM_THREADS>>> (int64_t * P_row_count, int64_t * P_col_count, int64_t * P_nnz, int MAX_COL_SIZE, double* P_data, int* P_indexes, int* P_indptr, int64_t * V_nnz, double* V_data, int* V_indexes, double* Res_data, int* Res_indexes);
}
