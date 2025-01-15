#include "BLAEQ.h";
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <iostream>


BLAEQ::BLAEQ(int input_N, int input_D, double* multi_dimensional_mesh, int input_K) {
	N = input_N;
	D = input_D;
	cusparseHandle_t cusparseHandle;
	cusparseCreate(&cusparseHandle);

	BLAEQ_Dimensions = (BLAEQ_Dimension*)malloc(D * sizeof(BLAEQ_Dimension*));

	for (int i = 0; i < D; i++) {
		double* temp_mesh = &multi_dimensional_mesh[i * input_N];
		BLAEQ_Dimension BLAEQ_Dim = BLAEQ_Dimension(i, input_K, input_N, temp_mesh, &cusparseHandle);
		BLAEQ_Dimensions[i] = &BLAEQ_Dim;
		all_bandwidths[i] = BLAEQ_Dim.Bandwidths;
	}
}
void BLAEQ::BLAEQ_Query(double* ranges, int Q_count) {
	int Q_size = D * 2;
	for (int i = 0; i < Q_count; i++) {
		//Executing each query
		int64_t* result_count_ptr;
		int* result;
		BLAEQ_Single_Query(&ranges[Q_size * i], result_count_ptr, result);
		std::cout <<"The size of query result is£º"<< * result_count_ptr <<"." << std::endl;
	}
}

void BLAEQ::BLAEQ_Single_Query(double* ranges, int64_t* result_count_ptr, int* result) {
	cusparseSpVecDescr_t* dim_result;
	result = NULL;
	for (int i = 0; i < D; i++) {
		//2 for length {min, max}
		double min = ranges[i * 2];
		double max = ranges[i * 2 + 1];

		BLAEQ_Dimension tmp_dimension = *BLAEQ_Dimensions[i];
		tmp_dimension.BLAEQ_Query_Dimension(min, max, dim_result);
		
		int64_t* size;
		int64_t* nnz;
		void** indices;
		void** values;
		cusparseIndexType_t* idxType;
		cusparseIndexBase_t* idxBase;
		cudaDataType* valueType;

		cusparseSpVecGet(*dim_result, size, nnz, indices, values, idxType, idxBase, valueType);

		if (result == NULL) {
			result = (int*)indices;
			result_count_ptr = nnz;
		}
		else {
			int* result_start;
			int* result_end = thrust::set_intersection((int*)indices, (int*)indices + *nnz, result, result + *result_count_ptr, result_start);
			result = result_start;
			int64_t result_count = static_cast<int64_t>((result_end - result_start)/sizeof(int));
			result_count_ptr = &result_count;
		}
	}
}



Multidimensional_Arr::Multidimensional_Arr(int input_N, int input_D) {
	D = input_D;
	N = input_N;
	data = static_cast<double*>(std::malloc(D * N * sizeof(double)));
}

void Multidimensional_Arr::getDim(int dim, double* result, int* size) {
	result = &data[N * dim];
	size = &N;
}

void Multidimensional_Arr::getRecord(int n, double* result, int* size) {
	size = &D;

	double* result_arr_ptr = static_cast<double*>(std::malloc(D * sizeof(double)));

	for (int i = 0; i < D; i++) {
		result_arr_ptr[i] = data[i * N + n];
	}

	result = result_arr_ptr;
}





