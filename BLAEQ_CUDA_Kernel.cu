#include "BLAEQ_CUDA_Kernel.h"
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
__global__ void CUDA_in_range_kernel(double q_min, double q_max, double relaxation, double* data, int* indices, int size, double* result_data, int* result_indices) 
{
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

__global__ void CUDA_generate_P_matrix_kernel(double* M_i_d, unsigned int M_i_d_length, double bandwidth, double* data, int* row, int* col) 
{

	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < M_i_d_length) {
		int bin_index = floor(M_i_d[t_idx] / bandwidth);
		col[t_idx] = t_idx;
		row[t_idx] = bin_index;
		data[t_idx] = bandwidth * bin_index + bandwidth / 2;
	}
	//return cudaStatus;
}


__global__ void CUDA_BLAEQ_SpMSpV_kernel(int64_t* P_row_count, int64_t* P_col_count, int64_t* P_nnz, int MAX_COL_SIZE, double* P_data, int* P_indexes, int* P_indptr, int64_t* V_nnz, double* V_data, int* V_indexes, double* Res_data, int* Res_indexes) 
{

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

BLAEQ_CUDA_Kernel::BLAEQ_CUDA_Kernel(int input_COL_SIZE_THRESHOLD) {
	COL_SIZE_THRESHOLD = input_COL_SIZE_THRESHOLD;
	NUM_BLOCKS = 16;
	NUM_THREADS = 256;


}

BLAEQ_CUDA_Kernel::~BLAEQ_CUDA_Kernel() {

}

void BLAEQ_CUDA_Kernel::In_Range(double min, double max, double relaxation, cusparseSpVecDescr_t* input, cusparseSpVecDescr_t* output) {
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

	CUDA_in_range_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (min, max, relaxation, (double*)data, (int*)index, *size, tmp_result_data, tmp_result_indexes);


	int* index_end_ptr = thrust::remove(tmp_result_indexes, tmp_result_indexes + *size, 0);
	double* data_end_ptr = thrust::remove(tmp_result_data, tmp_result_data + *size, 0.0);

	int64_t nnz_size = index_end_ptr - tmp_result_indexes;

	cusparseCreateSpVec(output, *size, nnz_size, tmp_result_indexes, tmp_result_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cudaFree(input);
}


void BLAEQ_CUDA_Kernel::Generate_P_Matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc, double* M_ip1_d, int *M_ip1_length, cusparseHandle_t* cusparseHandle) {

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Initializing P_matrix inn COO format
	int P_row_count = M_i_d_length;
	int P_col_count = M_i_d_length;
	int P_nnz_count = M_i_d_length;

	// Allocate GPU buffers for three vectors (two input, one output)

	double* P_data;
	int* P_rows;
	int* P_cols;

	cudaStatus = cudaMalloc(&P_data, P_nnz_count * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&P_rows, P_row_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&P_cols, P_col_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	double* M_i_d_DRAM;
	cudaStatus = cudaMalloc(&M_i_d_DRAM, M_i_d_length * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemcpy(M_i_d_DRAM, M_i_d, M_i_d_length * sizeof(double), cudaMemcpyHostToDevice);
	CUDA_generate_P_matrix_kernel <<<NUM_BLOCKS, NUM_THREADS>>> (M_i_d_DRAM, M_i_d_length, bandwidth, P_data, P_rows, P_cols);
	//;
	

	cusparseSpMatDescr_t P_matrix_coo = nullptr;
	cusparseCreateCoo(&P_matrix_coo, P_row_count, P_col_count, P_nnz_count, P_rows, P_cols, P_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	cudaDeviceSynchronize();
	// Converting COO to CSC
	int P_index_count_csc = M_i_d_length;
	int P_indptr_count_csc = floor(*thrust::max_element(M_i_d, M_i_d + M_i_d_length) / bandwidth) + 1;
	int P_data_count_csc = M_i_d_length;
	int M_ip1_d_size = P_indptr_count_csc - 1;

	double* P_data_csc;
	int* P_index_csc;
	int* P_indptr_csc;

	cudaStatus = cudaMalloc(&P_data_csc, P_data_count_csc * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&P_index_csc, P_index_count_csc * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&P_indptr_csc, P_indptr_count_csc * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Note: Function 'cusparseXcoo2csr' can also be used for converting COO to csc, since it basically just merges indexes into indptrs, whose principle are identical for CSR & CSC.
	cusparseXcoo2csr(*cusparseHandle, P_cols, P_nnz_count, P_col_count, P_indptr_csc, CUSPARSE_INDEX_BASE_ZERO); //Converting COO to CSC
	cusparseCreateCsc(P_matrix_csc, P_row_count, P_col_count, P_nnz_count, P_indptr_csc, P_index_csc, P_data_csc, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);


	cudaDeviceSynchronize();

	//Constructing M_ip1_d, we not only need to construct the P_matrix, we also need to construct the next-layer vector.
	double* M_ip1_d_local = (double*)malloc(M_ip1_d_size * sizeof(double));

	for (int i = 0; i < M_ip1_d_size; i++) {
		M_ip1_d_local[i] = i * bandwidth + bandwidth / 2;
	}

	cudaStatus = cudaMalloc(&M_ip1_d, M_ip1_d_size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(M_ip1_d, M_ip1_d_local, M_ip1_d_size * sizeof(double), cudaMemcpyHostToDevice);
	M_ip1_length = &M_ip1_d_size;

	Error:
	free(P_data);
	free(P_rows);
	free(P_cols);

	free(M_i_d_DRAM);

	free(P_data_csc);
	free(P_index_csc);
	free(P_indptr_csc);
}

//void BLAEQ_CUDA_Kernels::Balance_P_Matrix(int MAX_BIN_SIZE, int M_index_count, int M_indptr_count, int* M_indptr, double* V_data, int* M_indptr_balanced, double* V_data_balanced) {
void BLAEQ_CUDA_Kernel::Balance_P_Matrix(cusparseSpMatDescr_t* original_P_matrix, cusparseSpMatDescr_t* balanced_P_matrix, double* original_V, double* balanced_V, int* balanced_V_size_ptr) {
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

	//Getting components of original_P_matrix
	int64_t* row_count;
	int64_t* col_count;
	int64_t* nnz;
	void** indptr;
	void** indices;
	void** data;
	cusparseIndexType_t* indptr_type;
	cusparseIndexType_t* index_type;
	cusparseIndexBase_t* idxBase;
	cudaDataType* data_type;

	cusparseCscGet(*original_P_matrix, row_count, col_count, nnz, indptr, indices, data, indptr_type, index_type, idxBase, data_type);

	//We only need to update the indptr of P matrix and generate new V, so we only need two buffers.
	int* balanced_indptr_buffer; 
	double* balancd_V_data_buffer;
	int balanced_bin_count_upperbound = std::ceil(*col_count + *nnz / COL_SIZE_THRESHOLD);

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

			balancd_V_data_buffer[i + balanced_array_offset] = original_V[i];
		}
		else if (tmp_csc_index - prev_csc_index <= COL_SIZE_THRESHOLD) {
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer[i + balanced_array_offset] = original_V[i];
		}
		else if (tmp_csc_index - prev_csc_index > COL_SIZE_THRESHOLD) {
			while (tmp_csc_index - prev_csc_index > COL_SIZE_THRESHOLD) {
				balanced_indptr_buffer[i + balanced_array_offset] = prev_csc_index + COL_SIZE_THRESHOLD;

				balancd_V_data_buffer[i + balanced_array_offset] = original_V[i];

				balanced_array_offset += 1;
				prev_csc_index += COL_SIZE_THRESHOLD;
			}
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;
		}
		else {
			std::cerr << "WTF??? The program should never reach here. Error when calling function _generate_P_matrix()." << std::endl;
		}
	}


	
	//Generating & returning balanced vector
	int balanced_V_size = *col_count + balanced_array_offset;
	balanced_V_size_ptr = &balanced_V_size;
	cudaMalloc(&balanced_V, balanced_V_size * sizeof(double)); //The '-1' operator is required since |indptr| = |col| + 1, and |V| = |col|.
	for (int i = 0; i < balanced_V_size; i++) {
		balanced_V[i] = balancd_V_data_buffer[i];
	}

	cudaFree(&balanced_indptr_buffer);
	cudaFree(&balancd_V_data_buffer);


	//Generating & returning balanced_P_matrix
	int* balanced_indptr;

	cudaMalloc(&balanced_indptr, (balanced_V_size + 1) * sizeof(int));

	for (int i = 0; i < (balanced_V_size + 1); i++) {
		balanced_indptr[i] = balanced_indptr_buffer[i];
	}

	//Here, balanced_V_size equals column count.
	cusparseCreateCsc(balanced_P_matrix, *row_count, balanced_V_size, *nnz, balanced_indptr, indices, data, *indptr_type, *index_type, *idxBase, *data_type);
}


void BLAEQ_CUDA_Kernel::SpMSpV(cusparseSpMatDescr_t* P_matrix, cusparseSpVecDescr_t* input_vec, cusparseSpVecDescr_t* result_vec) {

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

	int raw_result_vec_size = COL_SIZE_THRESHOLD * *col_count;

	cudaMalloc(&res_data, raw_result_vec_size * sizeof(double));
	cudaMalloc(&res_indexes, raw_result_vec_size * sizeof(int));

	CUDA_BLAEQ_SpMSpV_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (row_count, col_count, nnz_count, COL_SIZE_THRESHOLD, (double*)data, (int*)indexes, (int*)indptr, vec_nnz, (double*)vec_data, (int*)vec_indexes, res_data, res_indexes);
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