#ifndef BLAEQ_CUDA_KERNELS_H
#define BLAEQ_CUDA_KERNELS_H

#include "cuda_runtime.h"

void CUDA_in_range(double q_min, double q_max, double relaxation, double* data, int* indices, int size, double* result_data, int* result_indices, int NUM_BLOCKS, int NUM_THREADS);
void CUDA_generate_P_matrix(double* M_i_d, int M_i_d_length, double bandwidth, double* data, int* row, int* col, int NUM_BLOCKS, int NUM_THREADS);
void CUDA_BLAEQ_SpMSpV(int64_t* P_row_count, int64_t* P_col_count, int64_t* P_nnz, int MAX_COL_SIZE, double* P_data, int* P_indexes, int* P_indptr, int64_t* V_nnz, double* V_data, int* V_indexes, double* Res_data, int* Res_indexes, int NUM_BLOCKS, int NUM_THREADS);\


#endif 