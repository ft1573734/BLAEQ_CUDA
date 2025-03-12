#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <iostream>
#include <stdio.h>
#include "BLAEQ_Dimension.h"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <cstdlib>
#include <pcl/point_cloud.h>
#include <chrono>


#ifndef DEBUG
#define DEBUG true
#endif

#define BOOST_DISABLE_CURRENT_LOCATION

class BLAEQ {
public:
	int N;
	int D;
	int L;
	double* all_bandwidths;
	BLAEQ_Dimension* BLAEQ_Dimensions;
	//BLAEQ Initializer
	BLAEQ(int N, int D, double* multi_dimensional_mesh, int K);

	void BLAEQ_Single_Query(double* ranges, int64_t* result_count_ptr, int** result);

	void BLAEQ_Query(double* ranges, int Q_count);

private:

	double _compute_layer(int N, int k);
};


class Multidimensional_Arr {
public:
	int D;
	int N;
	double* data;
	Multidimensional_Arr(int N, int D);

	void getDim(int dim, double* result, int* size);

	void getRecord(int n, double* result, int* size);
};