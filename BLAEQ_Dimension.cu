#include "BLAEQ_Dimension.h"
#include "BLAEQ_CUDA_Kernels.cu"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>



BLAEQ_Dimension::BLAEQ_Dimension(int dim, int K, int N, double* M, cusparseHandle_t* cusparseHandle) {
	Dimension = dim;
	L = compute_layer(N, K);
	cudaMalloc(&P_Matrices, L * sizeof(void*));	//Allocating space for P_matrix pointers
	cudaMalloc(&Bandwidths, L * sizeof(double));	//Allocating space for bandwidths
	N = N;
	K = K;
	cudaMalloc(&Coarsest_Mesh, L * sizeof(void*));
	BLAEQ_Generate_P_Matrices_Dimension(P_Matrices, Coarsest_Mesh, M, cusparseHandle);
}

void BLAEQ_Dimension::BLAEQ_Generate_P_Matrices_Dimension(cusparseSpMatDescr_t** P_Matrices, cusparseSpVecDescr_t* coarsestMesh, double* original_mesh, cusparseHandle_t* cusparseHandle) {

	std::cout << "Generating Prolongation matrix for dimension " << Dimension << " ...";
	double* M_i_d = original_mesh;
	int N_i_d = N;
	for (int i = 0; i < L; i++) {
		double bandwidth = _bandwidth_generator(M_i_d, N_i_d, K);
		Bandwidths[(L - 1) - i] = bandwidth; //Store bandwidths in reverse order so that the coarsest layer corresponds to Bandwidths[0], second layer corresponds to Bandwidths[1] and so forth.
		double* M_ip1_d;
		int N_ip1_d;
		_generate_P_matrix(M_i_d, N_i_d, bandwidth, P_Matrices[(L - 1) - i], M_ip1_d, N_ip1_d, cusparseHandle); //Store P_Matrices in reverse order just like bandwidths.
		M_i_d = M_ip1_d;
		N_i_d = N_ip1_d;
	}
	int* coarsest_mesh_indices;
	cudaMalloc(&coarsest_mesh_indices, N_i_d * sizeof(int));
	for (int i = 0; i < N_i_d; i++) {
		coarsest_mesh_indices[i] = i;
	}
	cusparseCreateSpVec(coarsestMesh, N_i_d, N_i_d, coarsest_mesh_indices, M_i_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

}

void BLAEQ_Dimension::BLAEQ_Query_Dimension(double min, double max, cusparseSpVecDescr_t* input_result, cusparseSpVecDescr_t* output_result) {


	for (int i = 0; i < L; i++) {
		_logical_in_range_judgement(min, max, input_result, output_result);
		cusparseSpMatDescr_t* P_matrix = P_Matrices[i];
		_BLAEQ_SpMSpV(P_matrix, input_result, output_result);
		output_result = input_result;
		//CUDA_in_range(double q_min, double q_max, double relaxation, double* data, double* indices, double* result_data, double* result_indices, int size)

	}
}


/*
*
	Below are tools necessary for BLAEQ. These functions should not be called outside of BLAEQ.
*
*/

int BLAEQ_Dimension::compute_layer(int N, int k) {
	return log2(N) / log2(k) + 1;
}

double BLAEQ_Dimension::_bandwidth_generator(double* vector, int size, int K) {
	int bin_count = size / K;
	double bandwidth = _compute_range(vector, size);
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

	int NUM_BLOCKS = (int)ceil(M_i_d_length / NUM_THREADS);
	CUDA_generate_P_matrix_kernel << <NUM_BLOCKS, NUM_THREADS >> > (M_i_d, M_i_d_length, bandwidth, P_data, P_rows, P_cols);

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
void BLAEQ_Dimension::_balance_P_matrix(int MAX_BIN_SIZE, int M_index_count, int M_indptr_count, int* M_indptr, double* V_data, int* M_indptr_balanced, double* V_data_balanced) {
	//The code below is used for balancing the P matrix, such that the largest column size does not exceed MAX_COUNT_PER_COL.
	/*
		This part of the logic is implemented as follows:
		1.	Copy the first element into balanced_P directly;
		2.	For later elements:
		3.		Compare the temporary element with its previous element;
		4.		If the gap is smaller than MAX_COUNT_PER_COL, write temporary element into balanced_P directly;
		5.		If the gap is larger, write prev + MAX_COUNT_PER_COL into balanced_P;
		6.		Update prev = prev + MAX_COUNT_PER_COL, offset += 1;
		7.		Goto step 3;
		8.	Repeat 1-7 until balanced P indptr is built.
		NOTE: the indexes and data of P is identical to balanced_P, no change is required.
	*/
	int* balanced_indptr_buffer;
	double* balancd_V_data_buffer;
	int balanced_bin_count_upperbound = M_indptr_count + (N / MAX_COUNT_PER_COL);

	cudaMalloc(&balanced_indptr_buffer, balanced_bin_count_upperbound * sizeof(int));
	cudaMalloc(&balancd_V_data_buffer, balanced_bin_count_upperbound * sizeof(double));

	int balanced_P_col_count_csc = M_index_count;
	int balanced_array_offset = 0;
	for (int i = 0; i < M_indptr_count; i++) {
		int tmp_csc_index = M_indptr[i];
		int prev_csc_index = 0;
		if (i == 0) {
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer[i + balanced_array_offset] = V_data[i];
		}
		else if (tmp_csc_index - prev_csc_index <= MAX_COUNT_PER_COL) {
			balanced_indptr_buffer[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer[i + balanced_array_offset] = V_data[i];
		}
		else if (tmp_csc_index - prev_csc_index > MAX_COUNT_PER_COL) {
			while (tmp_csc_index - prev_csc_index > MAX_COUNT_PER_COL) {
				balanced_indptr_buffer[i + balanced_array_offset] = prev_csc_index + MAX_COUNT_PER_COL;

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

void BLAEQ_Dimension::_logical_in_range_judgement(double min, double max, cusparseSpVecDescr_t *input, cusparseSpVecDescr_t* output) {
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

	CUDA_in_range_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (min, max, Bandwidths[i] / 2, data, index, *size, tmp_result_data, tmp_result_indexes);

	int* index_end_ptr = thrust::remove(tmp_result_indexes, tmp_result_indexes + *size, 0);
	double* data_end_ptr = thrust::remove(tmp_result_data, tmp_result_data + *size, 0.0);

	int64_t nnz_size = index_end_ptr - tmp_result_indexes;

	cusparseCreateSpVec(output, *size, nnz_size, tmp_result_indexes, tmp_result_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cudaFree(input);
}

void BLAEQ_Dimension::_BLAEQ_SpMSpV(cusparseSpMatDescr_t *P_matrix, cusparseSpVecDescr_t *input_vec, cusparseSpVecDescr_t *result_vec) {

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

	CUDA_BLAEQ_SpMSpV_kernel(row_count, col_count, nnz_count, MAX_COUNT_PER_COL, (double*)data, (int*) indexes, (int*) indptr, vec_nnz, (double*) vec_data, (int*) vec_indexes, res_data, res_indexes);


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