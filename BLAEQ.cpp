#include "BLAEQ.h";
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <iostream>


BLAEQ::BLAEQ(int input_N, int input_D, double* multi_dimensional_mesh, int input_K) {
	N = input_N;
	D = input_D;
	cusparseHandle_t cusparseHandle;
	cusparseCreate(&cusparseHandle);
	L = _compute_layer(N, input_K);

	BLAEQ_Dimensions = (BLAEQ_Dimension**)malloc(D * sizeof(BLAEQ_Dimension*));
	all_bandwidths = (double*)malloc(D * (L - 1) * sizeof(double));

	for (int i = 0; i < D; i++) {
		double* temp_mesh = &multi_dimensional_mesh[i * input_N];
		BLAEQ_Dimension BLAEQ_Dim = BLAEQ_Dimension(i, L, input_K, input_N, temp_mesh, &cusparseHandle);
		BLAEQ_Dimensions[i] = &BLAEQ_Dim;
		for (int j = 0; j < (L - 1); j++) {
			all_bandwidths[i * (L - 1) + j] = BLAEQ_Dim.Bandwidths[j];
		}
	}
}
void BLAEQ::BLAEQ_Query(double* ranges, int Q_count) {
	int Q_size = D * 2;
	for (int i = 0; i < Q_count; i++) {
		//Executing each query
		int64_t result_count = -1;
		int* result = nullptr;
		BLAEQ_Single_Query(ranges, &result_count, &result);
		std::cout <<"The size of query result is£º"<< result_count <<"." << std::endl;
	}
}

void BLAEQ::BLAEQ_Single_Query(double* ranges, int64_t* result_count_ptr, int** result) {

	int** result_idx_of_each_dimension = (int**)malloc(D * sizeof(int*));
	double** result_data_of_each_dimension = (double**)malloc(D * sizeof(double*));

	for (int i = 0; i < D; i++) {
		//2 for length {min, max}
		double min = ranges[i * 2];
		double max = ranges[i * 2 + 1];

		int* result_idx_host;
		double* result_data_host;
		BLAEQ_Dimension tmp_dimension = *BLAEQ_Dimensions[i];
		tmp_dimension.BLAEQ_Query_Dimension(min, max, &result_idx_host, &result_data_host);
		if (result_idx_of_each_dimension == NULL) {
			std::cerr << "Allocating memory for 'result_idx_of_each_dimension' failed." << std::endl;
			goto Cleanup;
		}
		result_idx_of_each_dimension[i] = result_idx_host;
		if (result_data_of_each_dimension == NULL) {
			std::cerr << "Allocating memory for 'result_data_of_each_dimension' failed." << std::endl;
			goto Cleanup;
		}
		result_data_of_each_dimension[i] = result_data_host;
		
		//int64_t size;
		//int64_t nnz;
		//void* indices;
		//void* values;
		//cusparseIndexType_t idxType;
		//cusparseIndexBase_t idxBase;
		//cudaDataType valueType;

		//cusparseSpVecGet(dim_result, &size, &nnz, &indices, &values, &idxType, &idxBase, &valueType);
		//if (all_results != nullptr) {
		//	all_results[i] = dim_result;
		//}



		//int* result_arr;

		//if (result == NULL) {
		//	result_arr = (int*)indices;
		//	*result = result_arr;
		//	*result_count_ptr = nnz;
		//}
		//else {
		//	int* result_start;
		//	int* result_end = thrust::set_intersection((int*)indices, (int*)indices + nnz, result, result + *result_count_ptr, result_start);
		//	*result = result_start;
		//	int64_t result_count = static_cast<int64_t>((result_end - result_start)/sizeof(int));
		//	result_count_ptr = &result_count;
		//}
	Cleanup:
		free(result_idx_of_each_dimension);
		free(result_data_of_each_dimension);
	}




}

double BLAEQ::_compute_layer(int N, int k) {
	return log2(N) / log2(k) + 1;
}

Multidimensional_Arr::Multidimensional_Arr(int input_N, int input_D) {
	D = input_D;
	N = input_N;
	data = (double*)malloc(D * N * sizeof(double));
}

void Multidimensional_Arr::getDim(int dim, double* result, int* size) {
	result = &data[N * dim];
	size = &N;
}

void Multidimensional_Arr::getRecord(int n, double* result, int* size) {
	size = &D;
	double* result_arr_ptr = (double*)malloc(D * sizeof(double));
	for (int i = 0; i < D; i++) {
		result_arr_ptr[i] = data[i * N + n];
	}
	result = result_arr_ptr;
}





