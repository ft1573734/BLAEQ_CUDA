#include "BLAEQ_CUDA_Kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <stdio.h>
#include <math.h>
#include <cmath>


// Define the maximum number of threads per block for your GPU
const int MAX_THREADS_PER_BLOCK = 1024;

// Define the warp size for your GPU
const int WARP_SIZE = 32;


const unsigned int CONST_NUM_BLOCKS = 24;

/*
	CUDA Kernel Functions
*/
__global__ void CUDA_in_range_kernel(double q_min, double q_max, double relaxation, double* data, int* indices, int size, int* result_indices, double* result_data)
{
	// int i = threadIdx.x;
	// c[i] = a[i] + b[i];
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < size) {
		if (data[t_idx] >= q_min - relaxation) {
			if (data[t_idx] <= q_max + relaxation) {
				result_data[t_idx] = data[t_idx];
				result_indices[t_idx] = indices[t_idx];
			}
			else {
				result_data[t_idx] = 0.0;
				result_indices[t_idx] = -1;
			}
		}
		else {
			result_data[t_idx] = 0.0;
			result_indices[t_idx] = -1;
		}
	}
	//if (t_idx < size) {
	//	if (data[t_idx] >= q_min - relaxation && data[t_idx] <= q_max + relaxation) {
	//		result_data[t_idx] = 0.0;
	//		result_indices[t_idx] = 0;
	//	}
	//	else {
	//		result_data[t_idx] = 0.0;
	//		result_indices[t_idx] = 0;
	//	}
	//}
}

__global__ void CUDA_generate_P_matrix_kernel(double* M_i_d, int M_i_d_length, double bandwidth, double* data, int* row, int* col) 
{
	//int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//for (int i = t_idx; i < M_i_d_length; i += blockDim.x * gridDim.x) {
	//	int bin_index = floor(M_i_d[t_idx] / bandwidth);
	//	row[t_idx] = t_idx;
	//	col[t_idx] = bin_index;
	//	data[t_idx] = bandwidth * bin_index + bandwidth / 2;
	//}
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < M_i_d_length) {
		int bin_index = floor(M_i_d[t_idx] / bandwidth);
		row[t_idx] = t_idx;
		col[t_idx] = bin_index;
		data[t_idx] = M_i_d[t_idx] / (bandwidth * bin_index + bandwidth / 2);
	}
	//return cudaStatus;
}


__global__ void CUDA_BLAEQ_SpMSpV_kernel(int64_t P_row_count, int64_t P_col_count, int64_t P_nnz, int MAX_COL_SIZE, double* P_data, int* P_indexes, int* P_indptr, int64_t V_size, int64_t V_nnz, double* V_data, int* V_indexes, double* Res_data, int* Res_indexes)
{
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < V_nnz) {
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
			Res_indexes[result_arr_start_index + i] = -1;
		}
	}
}

BLAEQ_CUDA_Kernel::BLAEQ_CUDA_Kernel(int input_COL_SIZE_THRESHOLD) {
	COL_SIZE_THRESHOLD = input_COL_SIZE_THRESHOLD;
	NUM_THREADS = 512;
	//DEBUG = true;

}


void BLAEQ_CUDA_Kernel::In_Range(double min, double max, double relaxation, cusparseSpVecDescr_t input, cusparseSpVecDescr_t* output) {
	void* index;
	void* data;
	int64_t size;
	int64_t nnz;
	cusparseIndexType_t index_type;
	cusparseIndexBase_t index_base;
	cudaDataType data_type;

	cusparseSpVecGet(input, &size, &nnz, &index, &data, &index_type, &index_base, &data_type);

	if (DEBUG) {
		int* debug_v_idx = (int*)malloc(nnz * sizeof(int));
		double* debug_v_data = (double*)malloc(nnz * sizeof(double));

		cudaMemcpy(debug_v_idx, index, nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_v_data, data, nnz * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_v_idx);
		free(debug_v_data);
	}


	int* tmp_result_indexes;
	double* tmp_result_data;
	cudaMalloc(&tmp_result_indexes, sizeof(int) * size);
	cudaMalloc(&tmp_result_data, sizeof(double) * size);
	// CUDA_in_range_kernel(double q_min, double q_max, double relaxation, double* data, double* indices, int size, double* result_data, int* result_indices)
	NUM_THREADS = calculate_optimal_NUM_THREADS(nnz, CONST_NUM_BLOCKS);
	int NUM_BLOCKS = nnz / NUM_THREADS + 1;
	CUDA_in_range_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (min, max, relaxation, (double*)data, (int*)index, size, tmp_result_indexes, tmp_result_data);

	if (DEBUG) {
		int* debug_v_idx = (int*)malloc(nnz * sizeof(int));
		double* debug_v_data = (double*)malloc(nnz * sizeof(double));

		cudaMemcpy(debug_v_idx, tmp_result_indexes, nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_v_data, tmp_result_data, nnz * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_v_idx);
		free(debug_v_data);
	}

	thrust::device_ptr<int> idx_device_thrust(tmp_result_indexes);
	thrust::device_ptr<double> data_device_thrust(tmp_result_data);

	//thrust::device_ptr<int> new_end = thrust::remove(dev_ptr, dev_ptr + array_size, 0);
	thrust::device_ptr<int> idx_end_thrust = thrust::remove(idx_device_thrust, idx_device_thrust + nnz, -1);
	thrust::device_ptr<double> data_end_thrust = thrust::remove(data_device_thrust, data_device_thrust + nnz, 0.0);

	//if (idx_end_thrust - idx_device_thrust != data_end_thrust - data_device_thrust) {
	//	std::cerr << "WTF?" << std::endl;
	//}
	int64_t compressed_idx_size = idx_end_thrust - idx_device_thrust;
	int64_t compressed_data_size = data_end_thrust - data_device_thrust;

	if (compressed_data_size != compressed_idx_size) {
		std::cerr << "Something is wrong in In_Range()..." << std::endl;
	}

	int compressed_size = compressed_data_size;

	int* compressed_idx_device;
	double* compressed_data_device;

	cudaMalloc(&compressed_idx_device, compressed_size * sizeof(int));
	cudaMalloc(&compressed_data_device, compressed_size * sizeof(double));

	cudaMemcpy(compressed_idx_device, idx_device_thrust.get(), compressed_size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(compressed_data_device, data_device_thrust.get(), compressed_size * sizeof(double), cudaMemcpyDeviceToDevice);

	cudaFree(tmp_result_indexes);
	cudaFree(tmp_result_data);

	if (DEBUG) {
		int* debug_v_idx = (int*)malloc(nnz * sizeof(int));
		double* debug_v_data = (double*)malloc(nnz * sizeof(double));

		cudaMemcpy(debug_v_idx, compressed_idx_device, nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_v_data, compressed_data_device, nnz * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_v_idx);
		free(debug_v_data);
	}

	cusparseCreateSpVec(output, size, compressed_size, compressed_idx_device, compressed_data_device, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

}


void BLAEQ_CUDA_Kernel::Generate_P_Matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc, double** M_ip1_d, int *M_ip1_size, cusparseHandle_t* cusparseHandle) {

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Initializing P_matrix inn COO format
	double* M_i_d_DRAM;
	cudaStatus = cudaMalloc(&M_i_d_DRAM, M_i_d_length * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemcpy(M_i_d_DRAM, M_i_d, M_i_d_length * sizeof(double), cudaMemcpyHostToDevice);

	thrust::device_ptr<double> thrust_ptr_DRAM = thrust::device_pointer_cast(M_i_d_DRAM);
	int P_col_count = floor(*thrust::max_element(thrust_ptr_DRAM, thrust_ptr_DRAM + M_i_d_length) / bandwidth) + 1;
	int P_nnz_count = M_i_d_length;
	int P_row_count = M_i_d_length;

	*M_ip1_size = P_col_count;

	// Allocate GPU buffers for three vectors (two input, one output)

	double* P_data;
	int* P_rows;
	int* P_cols;

	cudaStatus = cudaMalloc(&P_data, P_nnz_count * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&P_rows, P_nnz_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&P_cols, P_nnz_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	NUM_THREADS = calculate_optimal_NUM_THREADS(M_i_d_length, CONST_NUM_BLOCKS);
	int NUM_BLOCKS = M_i_d_length / NUM_THREADS + 1; // Basically equal to ceil(M_i_d_length / NUM_THREADS)
	CUDA_generate_P_matrix_kernel <<<NUM_BLOCKS, NUM_THREADS>>> (M_i_d_DRAM, M_i_d_length, bandwidth, P_data, P_rows, P_cols);
	free(M_i_d);


	cusparseSpMatDescr_t P_matrix_coo = nullptr;
	cusparseCreateCoo(&P_matrix_coo, P_row_count, P_col_count, P_nnz_count, P_rows, P_cols, P_data, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	if (DEBUG) {
		int* debug_coo_rows = (int*)malloc(P_nnz_count * sizeof(int));
		int* debug_coo_cols = (int*)malloc(P_nnz_count * sizeof(int));
		double* debug_coo_data = (double*)malloc(P_nnz_count * sizeof(double));

		cudaMemcpy(debug_coo_rows, P_rows, P_nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_coo_cols, P_cols, P_nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_coo_data, P_data, P_nnz_count * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_coo_rows);
		free(debug_coo_cols);
		free(debug_coo_data);
	}


	cudaDeviceSynchronize();
	// Converting COO to CSC

	double* P_data_csc = P_data;
	int* P_index_csc = P_rows;
	int* P_indptr_csc;

	cudaStatus = cudaMalloc(&P_indptr_csc, (P_col_count + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Note: Function 'cusparseXcoo2csr' can also be used for converting COO to csc, since it basically just merges indexes into indptrs, whose principle are identical for CSR & CSC.
	cusparseXcoo2csr(*cusparseHandle, P_cols, P_nnz_count, P_col_count, P_indptr_csc, CUSPARSE_INDEX_BASE_ZERO); //Converting COO to CSC
	cusparseSpMatDescr_t P_matrix_csc_local;
	cusparseCreateCsc(&P_matrix_csc_local, P_row_count, P_col_count, P_nnz_count, P_indptr_csc, P_index_csc, P_data_csc, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	if (DEBUG) {
		int* indexes_host_debug = (int*)malloc(P_nnz_count * sizeof(int));
		int* indptr_host_debug = (int*)malloc((P_col_count + 1) * sizeof(int));
		double* data_host_debug = (double*)malloc(P_nnz_count * sizeof(double));

		cudaMemcpy(indexes_host_debug, P_index_csc, P_nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(indptr_host_debug, P_indptr_csc, (P_col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(data_host_debug, P_data_csc, P_nnz_count * sizeof(double), cudaMemcpyDeviceToHost);

		// Pause here and debug

		free(indexes_host_debug);
		free(indptr_host_debug);
		free(data_host_debug);
	}
	*P_matrix_csc = P_matrix_csc_local;

	cudaDeviceSynchronize();

	double* M_ip1_d_host = (double*)malloc(*M_ip1_size * sizeof(double));

	for (int i = 0; i < *M_ip1_size; i++) {
		M_ip1_d_host[i] = i * bandwidth + bandwidth / 2;
	}
	*M_ip1_d = M_ip1_d_host;


Error:

	cusparseDestroySpMat(P_matrix_coo);

	cudaFree(P_cols);
	// You MUST NOT free P_data & P_rows here, since they are used in P_matrix_csc as well.
	// DON'T cudaFree(P_data);
	// DON'T cudaFree(P_rows);

	cudaFree(M_i_d_DRAM);
}

//void BLAEQ_CUDA_Kernels::Balance_P_Matrix(int MAX_BIN_SIZE, int M_index_count, int M_indptr_count, int* M_indptr, double* V_data, int* M_indptr_balanced, double* V_data_balanced) {
void BLAEQ_CUDA_Kernel::Balance_P_Matrix(cusparseSpMatDescr_t original_P_matrix, cusparseSpMatDescr_t* balanced_P_matrix, double* original_V, int original_V_size, double** balanced_V, int* balanced_V_size) {
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

	int64_t row_count;
	int64_t col_count;
	int64_t nnz;
	void* indptr_receiver;
	void* indexes_receiver;
	void* data_receiver;
	cusparseIndexType_t indptr_type;
	cusparseIndexType_t index_type;
	cusparseIndexBase_t idxBase;
	cudaDataType data_type;

	//cusparseSpMatGetSize(original_P_matrix, &row_count, &col_count, &nnz);

	cusparseCscGet(original_P_matrix, &row_count, &col_count, &nnz, &indptr_receiver, &indexes_receiver, &data_receiver, &indptr_type, &index_type, &idxBase, &data_type);

	if (DEBUG) {
		int* indexes_host_debug = (int*)malloc(nnz * sizeof(int));
		int* indptr_host_debug = (int*)malloc((col_count + 1) * sizeof(int));
		double* data_host_debug = (double*)malloc(nnz * sizeof(double));

		cudaMemcpy(indexes_host_debug, indexes_receiver, nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(indptr_host_debug, indptr_receiver, (col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(data_host_debug, data_receiver, nnz * sizeof(double), cudaMemcpyDeviceToHost);

		// Pause here and debug

		free(indexes_host_debug);
		free(indptr_host_debug);
		free(data_host_debug);
	}

	int* indptr = (int*)indptr_receiver;
	int* indices = (int*)indexes_receiver;
	double* data = (double*)data_receiver;

	//We only need to update the indptr of P matrix and new V, so we only need two buffers.
	int balanced_bin_count_upperbound = std::ceil(col_count + nnz / COL_SIZE_THRESHOLD);

	int* balanced_indptr_buffer_host = (int*)malloc((balanced_bin_count_upperbound + 1) * sizeof(int));
	double* balancd_V_data_buffer_host = (double*)malloc(balanced_bin_count_upperbound * sizeof(double));

	//Since indptr is on CUDA, we need to fetch it to host.
	int* indptr_host = (int*)malloc((col_count + 1) * sizeof(int));
	cudaMemcpy(indptr_host, indptr, (col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	//int balanced_P_col_count_csc = *col_count;
	int balanced_array_offset = 0;
	int prev_csc_index = 0;
	for (int i = 0; i < col_count + 1; i++) {
		int tmp_csc_index = indptr_host[i];
		if (i == 0) {
			balanced_indptr_buffer_host[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer_host[i + balanced_array_offset] = original_V[i];
		}
		else if (tmp_csc_index - prev_csc_index <= COL_SIZE_THRESHOLD) {
			balanced_indptr_buffer_host[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;

			balancd_V_data_buffer_host[i + balanced_array_offset] = original_V[i];
		}
		else if (tmp_csc_index - prev_csc_index > COL_SIZE_THRESHOLD) {
			while (tmp_csc_index - prev_csc_index > COL_SIZE_THRESHOLD) {
				balanced_indptr_buffer_host[i + balanced_array_offset] = prev_csc_index + COL_SIZE_THRESHOLD;

				balancd_V_data_buffer_host[i + balanced_array_offset] = original_V[i];

				balanced_array_offset += 1;
				prev_csc_index += COL_SIZE_THRESHOLD;
			}
			balanced_indptr_buffer_host[i + balanced_array_offset] = tmp_csc_index;
			prev_csc_index = tmp_csc_index;
		}
		else {
			std::cerr << "WTF??? The program should never reach here. Error when calling function _generate_P_matrix()." << std::endl;
		}
	}

	*balanced_V_size = col_count + balanced_array_offset;
	
	//Generating & returning balanced vector, notice that we return vectors on host instead of on device.
	double* balanced_V_local = (double*)malloc(*balanced_V_size * sizeof(double));
	memcpy(balanced_V_local, balancd_V_data_buffer_host, *balanced_V_size * sizeof(double));
	*balanced_V = balanced_V_local;

	//Generating & returning balanced_P_matrix
	int* balanced_indptr;
	cudaMalloc(&balanced_indptr, (*balanced_V_size + 1) * sizeof(int));
	cudaMemcpy(balanced_indptr, balanced_indptr_buffer_host, (*balanced_V_size + 1) * sizeof(int), cudaMemcpyHostToDevice);


	cusparseSpMatDescr_t balanced_P_matrix_local;
	//Here, balanced_V_size equals column count.
	cusparseCreateCsc(&balanced_P_matrix_local, row_count, *balanced_V_size, nnz, balanced_indptr, indices, data, indptr_type, index_type, idxBase, data_type);

	*balanced_P_matrix = balanced_P_matrix_local;


	free(balancd_V_data_buffer_host);
	free(balanced_indptr_buffer_host);
	free(indptr_host);
}


void BLAEQ_CUDA_Kernel::SpMSpV(cusparseSpMatDescr_t P_matrix, cusparseSpVecDescr_t input_vec, cusparseSpVecDescr_t* result_vec) {

	int64_t row_count;
	int64_t col_count;
	int64_t nnz_count;
	void* indptr;
	void* indexes;
	void* data;
	cusparseIndexType_t indptr_type;
	cusparseIndexType_t indexes_type;
	cusparseIndexBase_t idx_base;
	cudaDataType dataType;
	cusparseCscGet(P_matrix, &row_count, &col_count, &nnz_count, &indptr, &indexes, &data, &indptr_type, &indexes_type, &idx_base, &dataType);


	void* vec_indexes;
	void* vec_data;
	int64_t vec_size;
	int64_t vec_nnz;
	cusparseIndexType_t vec_index_type;
	cusparseIndexBase_t vec_index_base;
	cudaDataType vec_data_type;

	cusparseSpVecGet(input_vec, &vec_size, &vec_nnz, &vec_indexes, &vec_data, &vec_index_type, &vec_index_base, &vec_data_type);

	int* res_indexes;
	double* res_data;
	int64_t res_size;
	int64_t res_nnz;

	int raw_result_vec_size = COL_SIZE_THRESHOLD * vec_nnz;

	cudaMalloc(&res_data, raw_result_vec_size * sizeof(double));
	cudaMalloc(&res_indexes, raw_result_vec_size * sizeof(int));

	if (DEBUG) {
		int* debug_m_idx = (int*)malloc(nnz_count * sizeof(int));
		int* debug_m_indptr = (int*)malloc((col_count + 1) * sizeof(int));
		double* debug_m_data = (double*)malloc(nnz_count * sizeof(double));

		cudaMemcpy(debug_m_idx, indexes, nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_m_indptr, indptr, (col_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_m_data, data, nnz_count * sizeof(double), cudaMemcpyDeviceToHost);

		int* debug_v_idx = (int*)malloc(vec_nnz * sizeof(int));
		double* debug_v_data = (double*)malloc(vec_nnz * sizeof(double));

		cudaMemcpy(debug_v_idx, vec_indexes, vec_nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_v_data, vec_data, vec_nnz * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_m_idx);
		free(debug_m_indptr);
		free(debug_m_data);
		free(debug_v_idx);
		free(debug_v_data);
	}
	//TODOTODOTODO
	NUM_THREADS = calculate_optimal_NUM_THREADS(vec_nnz, CONST_NUM_BLOCKS);
	int NUM_BLOCKS = vec_nnz / NUM_THREADS + 1;
	CUDA_BLAEQ_SpMSpV_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (row_count, col_count, nnz_count, COL_SIZE_THRESHOLD, (double*)data, (int*)indexes, (int*)indptr, vec_size, vec_nnz, (double*)vec_data, (int*)vec_indexes, res_data, res_indexes);

	//thrust::device_ptr<int> idx_device_thrust(tmp_result_indexes);
	//thrust::device_ptr<double> data_device_thrust(tmp_result_data);

	////thrust::device_ptr<int> new_end = thrust::remove(dev_ptr, dev_ptr + array_size, 0);
	//thrust::device_ptr<int> idx_end_thrust = thrust::remove(idx_device_thrust, idx_device_thrust + nnz, 0);
	//thrust::device_ptr<double> data_end_thrust = thrust::remove(data_device_thrust, data_device_thrust + nnz, 0.0);

	////if (idx_end_thrust - idx_device_thrust != data_end_thrust - data_device_thrust) {
	////	std::cerr << "WTF?" << std::endl;
	////}
	//int64_t compressed_size = idx_end_thrust - idx_device_thrust;

	//int* compressed_idx_device;
	//double* compressed_data_device;

	//cudaMalloc(&compressed_idx_device, compressed_size * sizeof(int));
	//cudaMalloc(&compressed_data_device, compressed_size * sizeof(double));

	//cudaMemcpy(compressed_idx_device, idx_device_thrust.get(), compressed_size, cudaMemcpyDeviceToDevice);
	//cudaMemcpy(compressed_data_device, data_device_thrust.get(), compressed_size, cudaMemcpyDeviceToDevice);

	//cudaFree(tmp_result_indexes);
	//cudaFree(tmp_result_data);
	cudaFree(vec_indexes);
	cudaFree(vec_data);

	if (DEBUG) {
		int* debug_res_idx = (int*)malloc(raw_result_vec_size * sizeof(int));
		double* debug_res_data = (double*)malloc(raw_result_vec_size * sizeof(double));

		cudaMemcpy(debug_res_idx, res_indexes, raw_result_vec_size * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_res_data, res_data, raw_result_vec_size * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_res_idx);
		free(debug_res_data);
	}

	thrust::device_ptr<int> res_idx_begin_thrust(res_indexes);
	thrust::device_ptr<double> res_data_begin_thrust(res_data);

	thrust::device_ptr<int> res_idx_end_thrust = thrust::remove(res_idx_begin_thrust, res_idx_begin_thrust + raw_result_vec_size, -1);
	thrust::device_ptr<double> res_data_end_thrust = thrust::remove(res_data_begin_thrust, res_data_begin_thrust + raw_result_vec_size, 0.0);

	int64_t cleaned_idx_size = res_idx_end_thrust - res_idx_begin_thrust;
	int64_t cleaned_data_size = res_data_end_thrust - res_data_begin_thrust;

	if (cleaned_idx_size != cleaned_data_size) {
		std::cerr << "WTF?" << std::endl;
	}
	int64_t cleaned_v_size = cleaned_idx_size; // or cleaned_data_size.
	
	int* cleaned_idx_device;
	double* cleaned_data_device;

	cudaMalloc(&cleaned_idx_device, cleaned_v_size * sizeof(int));
	cudaMalloc(&cleaned_data_device, cleaned_v_size * sizeof(double));

	cudaMemcpy(cleaned_idx_device, res_idx_begin_thrust.get(), cleaned_v_size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cleaned_data_device, res_data_begin_thrust.get(), cleaned_v_size * sizeof(double), cudaMemcpyDeviceToDevice);

		if (DEBUG) {
		int* debug_res_idx = (int*)malloc(cleaned_v_size * sizeof(int));
		double* debug_res_data = (double*)malloc(cleaned_v_size * sizeof(double));

		cudaMemcpy(debug_res_idx, cleaned_idx_device, cleaned_v_size * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_res_data, cleaned_data_device, cleaned_v_size * sizeof(double), cudaMemcpyDeviceToHost);

		free(debug_res_idx);
		free(debug_res_data);
	}

	cusparseSpVecDescr_t result_vec_local;
	cusparseCreateSpVec(&result_vec_local, row_count, cleaned_v_size, cleaned_idx_device, cleaned_data_device, vec_index_type, vec_index_base, vec_data_type);
	*result_vec = result_vec_local;

	cudaFree(res_indexes);
	cudaFree(res_data);
}

void  BLAEQ_CUDA_Kernel::Restore_P_Index(cusparseSpMatDescr_t P_matrix, int* original_indexes, cusparseSpMatDescr_t* restored_P_matrix) {
	int64_t row_count;
	int64_t col_count;
	int64_t nnz_count;
	void* indptr;
	void* indexes;
	void* data;
	cusparseIndexType_t indptr_type;
	cusparseIndexType_t indexes_type;
	cusparseIndexBase_t idx_base;
	cudaDataType dataType;
	cusparseCscGet(P_matrix, &row_count, &col_count, &nnz_count, &indptr, &indexes, &data, &indptr_type, &indexes_type, &idx_base, &dataType);
	
	cusparseSpMatDescr_t restored_P_matrix_local;
	cusparseCreateCsc(&restored_P_matrix_local, row_count, col_count, nnz_count, indptr, original_indexes, data, indptr_type, indexes_type, idx_base, dataType);
	*restored_P_matrix = restored_P_matrix_local;

}


// Function to calculate the optimal number of threads per block
int BLAEQ_CUDA_Kernel::calculate_optimal_NUM_THREADS(int N, int NUM_BLOCKS) {
	// Calculate the initial number of threads per block
	int num_threads = static_cast<int>(std::ceil(static_cast<double>(N) / NUM_BLOCKS));

	// Round to the nearest multiple of the warp size
	num_threads = ((num_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

	// Ensure that the number of threads per block does not exceed the GPU's limit
	num_threads = std::min(num_threads, MAX_THREADS_PER_BLOCK);

	return num_threads;
}