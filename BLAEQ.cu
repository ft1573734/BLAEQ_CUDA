#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <iostream>
#include <stdio.h>
#include "BLAEQ_Dimension.h"

class BLAEQ {
public:
	int N;
	int D;
	double** all_bandwidths;
	BLAEQ_Dimension** BLAEQ_Dimensions;
	//BLAEQ Initializer
	BLAEQ(int N, int D, double* multi_dimensional_mesh, int K) {
		cusparseHandle_t cusparseHandle;
		cusparseCreate(&cusparseHandle);

		cudaMalloc(&BLAEQ_Dimensions, sizeof(void*));

		for (int i = 0; i < D; i++) {
			double* temp_mesh = &multi_dimensional_mesh[i * N];
			BLAEQ_Dimension BLAEQ_Dim = BLAEQ_Dimension(D, K, N, temp_mesh, &cusparseHandle);
			BLAEQ_Dimensions[i] = &BLAEQ_Dim;
			all_bandwidths[i] = BLAEQ_Dim.Bandwidths;
		}
	}

	/*
	 *	Executing Queries
	 */
	void BLAEQ_Query(double* ranges, BLAEQ_Dimension** BLAEQ_Dimensions, cusparseSpVecDescr_t* result) {
		for (int i = 0; i < D; i++) {
			double min = ranges[i * D];
			double max = ranges[i * D + 1];

			BLAEQ_Dimension tmp_dimension = *BLAEQ_Dimensions[i];
			tmp_dimension.BLAEQ_Query_Dimension(min, max, result);
		}
	}
private:
};



