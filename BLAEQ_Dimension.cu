#include "BLAEQ_Dimension.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>


BLAEQ_Dimension::BLAEQ_Dimension(int input_dim, int input_L, int input_K, int input_N, double* M, cusparseHandle_t* cusparseHandle) {
	double EPSILON = 0.00001;

	Dimension = input_dim;
	L = input_L;
	//cudaMalloc(&P_Matrices, L * sizeof(cusparseSpMatDescr_t*));	//Allocating space for P_matrix pointers
	//cudaMalloc(&Bandwidths, L * sizeof(double));	//Allocating space for bandwidths

	P_Matrices = (cusparseSpMatDescr_t*)malloc((L - 1) * sizeof(cusparseSpMatDescr_t));
	Bandwidths = (double*)malloc((L - 1) * sizeof(double));

	N = input_N;
	K = input_K;
	//MAX_COUNT_PER_COL = N / K;
	MAX_COUNT_PER_COL = 150;
	kernel = BLAEQ_CUDA_Kernel(MAX_COUNT_PER_COL);

	int* idx = (int*)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++) {
		idx[i] = i;
		M[i] = M[i] + EPSILON;
	}
	double* sorted_M = (double*)malloc(N * sizeof(double));
	sorted_idx = (int*)malloc(N * sizeof(int));
	BLAEQ_Sort(M, idx, &sorted_M, &sorted_idx);

	sorted_idx_device;
	cudaMalloc(&sorted_idx_device, N * sizeof(int));
	cudaMemcpy(sorted_idx_device, sorted_idx, N * sizeof(int), cudaMemcpyHostToDevice);

	BLAEQ_Generate_P_Matrices_Dimension(&Coarsest_Mesh, sorted_M, cusparseHandle);
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

void BLAEQ_Dimension::BLAEQ_Generate_P_Matrices_Dimension(cusparseSpVecDescr_t* coarsestMesh, double* original_mesh, cusparseHandle_t* cusparseHandle) {

	std::cout << "Generating Prolongation matrix for dimension " << Dimension << " ..." <<std::endl;
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
		Bandwidths[(L - 2) - i] = bandwidth; //Store bandwidths in reverse order so that the coarsest layer corresponds to Bandwidths[0], second layer corresponds to Bandwidths[1] and so forth.
		double* M_ip1_d;
		int N_ip1_d;

		double* balanced_M_ip1_d;
		int balanced_N_ip1_d;
		cusparseSpMatDescr_t tmp_P_matrix;
		cusparseSpMatDescr_t balanced_P_matrix;
		kernel.Generate_P_Matrix(M_i_d, N_i_d, bandwidth, &tmp_P_matrix, &M_ip1_d, &N_ip1_d, cusparseHandle);

		if (DEBUG) {

			int64_t row_count;
			int64_t col_count;
			int64_t nnz_count;
			void* debug_indptr_device;
			void* debug_idx_device;
			void* debug_data_device;
			cusparseIndexType_t indptr_type;
			cusparseIndexType_t idx_type;
			cusparseIndexBase_t idx_base;
			cudaDataType_t data_type;

			cusparseCscGet(tmp_P_matrix, &row_count, &col_count, &nnz_count, &debug_indptr_device, &debug_idx_device, &debug_data_device, &indptr_type, &idx_type, &idx_base, &data_type);

			int* debug_m_idx = (int*)malloc(nnz_count * sizeof(int));
			int* debug_m_indptr = (int*)malloc((col_count + 1) * sizeof(int));
			double* debug_m_data = (double*)malloc(nnz_count * sizeof(double));

			cudaMemcpy(debug_m_idx, (int*)debug_idx_device, nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_m_indptr, (int*)debug_indptr_device, (col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_m_data, (double*)debug_data_device, nnz_count * sizeof(double), cudaMemcpyDeviceToHost);

			free(debug_m_idx);
			free(debug_m_indptr);
			free(debug_m_data);
		}



		kernel.Balance_P_Matrix(tmp_P_matrix, &balanced_P_matrix, M_ip1_d, N_ip1_d, &balanced_M_ip1_d, &balanced_N_ip1_d);

		cusparseSpMatDescr_t restored_balanced_P_matrix;
		if (i == 0) {
			// Restore the indexes of the lowest-level P-matrix
			kernel.Restore_P_Index(balanced_P_matrix, this->sorted_idx_device, &restored_balanced_P_matrix);
		}
		else {
			restored_balanced_P_matrix = balanced_P_matrix;
		}

		//P_Matrices[i] = &balanced_P_matrix;
		//cusparseSpMatDescr_t* const P = &balanced_P_matrix;
		P_Matrices[(L - 2) - i] = restored_balanced_P_matrix;

		if (DEBUG) {

			int64_t row_count;
			int64_t col_count;
			int64_t nnz_count;
			void* debug_indptr_device;
			void* debug_idx_device;
			void* debug_data_device;
			cusparseIndexType_t indptr_type;
			cusparseIndexType_t idx_type;
			cusparseIndexBase_t idx_base;
			cudaDataType_t data_type;

			cusparseCscGet(restored_balanced_P_matrix, &row_count, &col_count, &nnz_count, &debug_indptr_device, &debug_idx_device, &debug_data_device, &indptr_type, &idx_type, &idx_base, &data_type);

			int* debug_m_idx = (int*)malloc(nnz_count * sizeof(int));
			int* debug_m_indptr = (int*)malloc((col_count + 1) * sizeof(int));
			double* debug_m_data = (double*)malloc(nnz_count * sizeof(double));

			cudaMemcpy(debug_m_idx, debug_idx_device, nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_m_indptr, debug_indptr_device, (col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_m_data, debug_data_device, nnz_count * sizeof(double), cudaMemcpyDeviceToHost);

			free(debug_m_idx);
			free(debug_m_indptr);
			free(debug_m_data);
		}

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

void BLAEQ_Dimension::BLAEQ_Query_Dimension(double min, double max, int* res_size, int** restored_indices, double** data) {

	cusparseSpVecDescr_t logical_result;
	cusparseSpVecDescr_t this_layer;
	cusparseSpVecDescr_t next_layer;

	this_layer = Coarsest_Mesh;
	for (int i = 0; i < L - 1; i++) {
		kernel.In_Range(min, max, Bandwidths[Dimension] / 2, this_layer, &logical_result);
		if (logical_result == NULL) {
			return;
		}
		cusparseSpMatDescr_t P_matrix = this->P_Matrices[i];

		if (DEBUG) {

			int64_t row_count;
			int64_t col_count;
			int64_t nnz_count;
			void* debug_indptr_device;
			void* debug_idx_device;
			void* debug_data_device;
			cusparseIndexType_t indptr_type;
			cusparseIndexType_t index_type;
			cusparseIndexBase_t idxBase;
			cudaDataType data_type;

			cusparseCscGet(P_matrix, &row_count, &col_count, &nnz_count, &debug_indptr_device, &debug_idx_device, &debug_data_device, &indptr_type, &index_type, &idxBase, &data_type);

			int* debug_m_idx = (int*)malloc(nnz_count * sizeof(int));
			int* debug_m_indptr = (int*)malloc((col_count + 1) * sizeof(int));
			double* debug_m_data = (double*)malloc(nnz_count * sizeof(double));

			cudaMemcpy(debug_m_idx, debug_idx_device, nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_m_indptr, debug_indptr_device, (col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_m_data, debug_data_device, nnz_count * sizeof(double), cudaMemcpyDeviceToHost);

			free(debug_m_idx);
			free(debug_m_indptr);
			free(debug_m_data);
		}

		kernel.SpMSpV(P_matrix, logical_result, &next_layer);
		this_layer = next_layer;

	}
	//result = this_layer;

	kernel.In_Range(min, max, 0.0, this_layer, &logical_result);



	int64_t size;
	int64_t nnz;
	void* indices_device;
	void* values_device;
	cusparseIndexType_t idx_type;
	cusparseIndexBase_t idx_base;
	cudaDataType_t data_type;
	cusparseSpVecGet(logical_result, &size, &nnz, &indices_device, &values_device, &idx_type, &idx_base, &data_type);

	//re-arrange the result back to default sequence
	//thrust::device_ptr<int> sorted_result_device_thrust((int*)indices_device);
	//int* original_idx_device;
	//cudaMalloc(&original_idx_device, nnz * sizeof(int));
	//thrust::device_ptr<int> restored_idx_device_thrust(original_idx_device);
	//thrust::device_ptr<int> sorted_idx_device_thrust(sorted_idx_device);

	//thrust::gather(sorted_result_device_thrust, sorted_result_device_thrust + nnz, sorted_idx_device_thrust, restored_idx_device_thrust);

	////*restored_indices = restored_idx_device_thrust.get();
	////*data = (double*)values_device;


	//int* restored_indices_host = (int*)malloc(nnz * sizeof(int));
	//cudaMemcpy(restored_indices_host, indices_device, nnz * sizeof(int), cudaMemcpyDeviceToHost);
	//*restored_indices = restored_indices_host;

	//double* res_data_host = (double*)malloc(nnz * sizeof(double));
	//cudaMemcpy(res_data_host, values_device, nnz * sizeof(int), cudaMemcpyDeviceToHost);
	//*data = res_data_host;

	*restored_indices = (int*)indices_device;
	*data = (double*)values_device;
	*res_size = nnz;


}


/*
*
	Below are tools necessary for BLAEQ. These functions should not be called outside of BLAEQ.
*
*/

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