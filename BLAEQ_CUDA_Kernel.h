#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include "debug.h"

#define BOOST_DISABLE_CURRENT_LOCATION

#ifndef DEBUG
#define DEBUG false
#endif

//Configuration based on NVIDIA Card

class BLAEQ_CUDA_Kernel {
	int COL_SIZE_THRESHOLD;
	unsigned int NUM_THREADS;
	//bool DEBUG;

public:
	BLAEQ_CUDA_Kernel(int input_MAX_COL_SIZE);

	BLAEQ_CUDA_Kernel() {  };


	void In_Range(double min, double max, double relaxation, cusparseSpVecDescr_t input, cusparseSpVecDescr_t* output);

	void Generate_P_Matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc, double** M_ip1_d, int* M_ip1_length, cusparseHandle_t* cusparseHandle);

	void SpMSpV(cusparseSpMatDescr_t P_matrix, cusparseSpVecDescr_t input_vec, cusparseSpVecDescr_t* result_vec);

	void Balance_P_Matrix(cusparseSpMatDescr_t original_P_matrix, cusparseSpMatDescr_t* balanced_P_matrix, double* original_V, int original_V_size, double** balanced_V, int* balanced_V_size);

	void Restore_P_Index(cusparseSpMatDescr_t _P_matrix, int* original_indexes, cusparseSpMatDescr_t* restored_indexes);


private:

	int calculate_optimal_NUM_THREADS(int nnz, int NUM_THREADS);

};
