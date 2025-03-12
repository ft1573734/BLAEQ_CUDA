#include "BLAEQ.h";
#include <iostream>


BLAEQ::BLAEQ(int input_N, int input_D, double* multi_dimensional_mesh, int input_K) {
	N = input_N;
	D = input_D;
	cusparseHandle_t cusparseHandle;
	cusparseCreate(&cusparseHandle);
	L = _compute_layer(N, input_K);

	BLAEQ_Dimensions = (BLAEQ_Dimension*)malloc(D * sizeof(BLAEQ_Dimension));
	all_bandwidths = (double*)malloc(D * (L - 1) * sizeof(double));

	for (int i = 0; i < D; i++) {
		double* temp_mesh = &multi_dimensional_mesh[i * input_N];
		BLAEQ_Dimension BLAEQ_Dim = BLAEQ_Dimension(i, L, input_K, input_N, temp_mesh, &cusparseHandle);
		BLAEQ_Dimensions[i] = BLAEQ_Dim;
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

		std::cout << "Processing Query " << i << " ..." << std::endl;
		auto hostStart = std::chrono::high_resolution_clock::now();
		BLAEQ_Single_Query(&ranges[Q_size * i], &result_count, &result);
		auto hostEnd = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> hostDuration = hostEnd - hostStart;
		std::cout <<"The size of query result is£º"<< result_count <<"." << std::endl;
		std::cout << "Time consumption of query " << i << " is " << hostDuration.count() << " ms." << std::endl;
	}
}

void BLAEQ::BLAEQ_Single_Query(double* ranges, int64_t* result_count_ptr, int** result) {

	thrust::device_ptr<int> final_idx_thrust;
	//double** final_result = (double**)malloc(D * sizeof(double*));
	int final_size = -1;

	for (int i = 0; i < D; i++) {
		//2 for length {min, max}
		double min = ranges[i * 2];
		double max = ranges[i * 2 + 1];

		int* r_idx_device;
		double* r_data_device;
		int r_size;
		BLAEQ_Dimension tmp_dimension = BLAEQ_Dimensions[i];
		tmp_dimension.BLAEQ_Query_Dimension(min, max, &r_size, &r_idx_device, &r_data_device);

		//if (DEBUG) {
		//	int prev_idx = -1;
		//	for (int i = 0; i < r_size; i++) {
		//		if(r_idx_host[i] < prev_idx) {
		//			std::cerr << "WTF" << std::endl;
		//		}
		//		else {
		//			prev_idx = r_idx_host[i];
		//		}
		//	}
		//}
		
		if (final_size == -1) {
			thrust::device_ptr<int> r_idx_sorted_thrust(r_idx_device);
			//thrust::device_ptr<double> r_data_thurst_sorted(r_data_device);
			thrust::sort(r_idx_sorted_thrust, r_idx_sorted_thrust + r_size);

			if (DEBUG) {
				int* sorted_idx_host = (int*)malloc(r_size * sizeof(int));
				cudaMemcpy(sorted_idx_host, r_idx_sorted_thrust.get(), r_size * sizeof(int), cudaMemcpyDeviceToHost);
				int prev_idx = -1;
				for (int i = 0; i < r_size; i++) {
					if(sorted_idx_host[i] < prev_idx) {
						std::cerr << "WTF" << std::endl;
					}
					else {
						prev_idx = sorted_idx_host[i];
					}
				}
				free(sorted_idx_host);
			}

			final_idx_thrust = r_idx_sorted_thrust;
			//final_result[i] = r_data_thurst_sorted.get();
			final_size = r_size;
		}
		else 
		{
			thrust::device_ptr<int> r_idx_sorted_thrust(r_idx_device);
			//thrust::device_ptr<double> r_data_thurst_sorted(r_data_device);
			thrust::sort(r_idx_sorted_thrust, r_idx_sorted_thrust + r_size);

			int _max_intersection_size = -1;

			if (final_size <= r_size) {
				_max_intersection_size = final_size;
			}
			else {
				_max_intersection_size = r_size;
			}

			int* r_buffer_host;
			cudaMalloc(&r_buffer_host, _max_intersection_size * sizeof(int));
			thrust::device_ptr<int> r_buffer_device_thrust(r_buffer_host);
			//thrust::device_ptr<int> r_buffer_device_thrust(r_buffer_device);
			//thrust::device_ptr<int> final_idx_thrust(final_idx);
			//thrust::device_ptr<int> r_idx_thrust(r_idx_device);
			thrust::device_ptr<int> _intersection_end = thrust::set_intersection(final_idx_thrust, final_idx_thrust + final_size, r_idx_sorted_thrust, r_idx_sorted_thrust + r_size, r_buffer_device_thrust);

			final_size = _intersection_end - r_buffer_device_thrust;
			if (final_size == 0) {
				goto Exit;
			}
			else {
				cudaFree(final_idx_thrust.get());
				//final_idx = (int*)malloc(final_size * sizeof(int));
				//memcpy(final_idx, r_buffer_host, final_size * sizeof(int));
				final_idx_thrust = r_buffer_device_thrust;
			}


			cudaFree(r_idx_sorted_thrust.get());
			cudaFree(r_idx_device);
			cudaFree(r_data_device);

		}
	}
Exit:
	*result_count_ptr = final_size;
	//std::cout << "The size of the query result is: " << final_size << "." << std::endl;
	return;
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





