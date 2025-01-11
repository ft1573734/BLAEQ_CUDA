#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"

#define BOOST_DISABLE_CURRENT_LOCATION

//Configuration based on NVIDIA Card

class BLAEQ_CUDA_Kernel {
	int COL_SIZE_THRESHOLD;
	unsigned int NUM_BLOCKS;
	unsigned int NUM_THREADS;

public:
	BLAEQ_CUDA_Kernel(int input_MAX_COL_SIZE);
	~BLAEQ_CUDA_Kernel();

	void In_Range(double min, double max, double relaxation, cusparseSpVecDescr_t* input, cusparseSpVecDescr_t* output);

	void Generate_P_Matrix(double* M_i_d, int M_i_d_length, double bandwidth, cusparseSpMatDescr_t* P_matrix_csc_balanced, double* M_ip1_d, int* M_ip1_length, cusparseHandle_t* cusparseHandle);

	void SpMSpV(cusparseSpMatDescr_t* P_matrix, cusparseSpVecDescr_t* input_vec, cusparseSpVecDescr_t* result_vec);

	void Balance_P_Matrix(cusparseSpMatDescr_t* original_P_matrix, cusparseSpMatDescr_t* balanced_P_matrix, double* original_V, double* balanced_V, int* balanced_V_size);

private:

};
