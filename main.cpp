#include "BLAEQ.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include "debug.h"
#include <cuda_profiler_api.h>



#define BOOST_DISABLE_CURRENT_LOCATION

Multidimensional_Arr Load_Pcd_Data(std::string path);
void Load_Queries(std::string path, int D, int Q_count, double** workload);
void f(std::string path);
std::vector<std::string> splitLine(const std::string& line, char delimiter);



int main() {
	//Load data
	std::cout << "Hello world!" << std::endl;
	std::string input_dir = "data\\synthetic.pcd";
	std::string input_query = "data\\queries.txt";
	Multidimensional_Arr dataset = Load_Pcd_Data(input_dir);

	int K = 100;

	//Load query file
	int Q_count = 10;
	double* workload;

	Load_Queries(input_query, dataset.D, Q_count, &workload);

	BLAEQ BLAEQ_Object = BLAEQ(dataset.N, dataset.D, dataset.data, K);


	/*
	* Perform Query
	*/

	Load_Queries(input_query, 3, Q_count, &workload);


	//Perform query
	cudaProfilerStart();
	BLAEQ_Object.BLAEQ_Query(workload, Q_count);
	cudaProfilerStop();

	return 0;
}


Multidimensional_Arr Load_Pcd_Data(std::string path) {
	const int D = 3; //When using pcl, the dimensionality is bound to be 3.

	pcl::PointCloud<pcl::PointXYZ>* cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1) {
		PCL_ERROR("PCL: Couldn't read the file \n");
	}

	// Print the cloud data
	//std::cerr << "Loaded " << cloud->width * cloud->height << " data points from test_pcd.pcd with the following fields: " << std::endl;
	//for (const auto& point : *cloud) {
	//	std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;
	//}

	Multidimensional_Arr original_dataset = Multidimensional_Arr(cloud->points.size(), D);
	//Loading 3-D data
	pcl::PointCloud<pcl::PointXYZ>::const_iterator ite = cloud->cbegin();
	for (int i = 0; i < original_dataset.N; i++) {
		original_dataset.data[i + 0 * original_dataset.N] = ite->x;
		original_dataset.data[i + 1 * original_dataset.N] = ite->y;
		original_dataset.data[i + 2 * original_dataset.N] = ite->z;
		ite++;
	}
	delete(cloud);

	return original_dataset;
}

void Load_Queries(std::string path, int D, int Q_count, double** workload) {
	std::ifstream file(path);
	std::vector<double> numbers;
	std::string token;
	double value;

	if (!file.is_open()) {
		std::cerr << "Could not open the file - '" << path << "'" << std::endl;
	}

	//double* query_boundaries =nullptr;
	double* query_boundaries = (double*)malloc(Q_count * D * 2 * sizeof(double));

	for (int i = 0; i < Q_count; i++) {
		std::getline(file, token);
		std::vector<std::string> tokens = splitLine(token, ' ');
		std::vector<std::string>::iterator ite = tokens.begin();
		for (int j = 0; j < D * 2; j++) {
			query_boundaries[i * D * 2 + j] = std::stod(*ite);
			ite++;
		}
	}

	*workload = query_boundaries;

	file.close();
}

// Function to split a line into tokens based on a delimiter
std::vector<std::string> splitLine(const std::string& line, char delimiter) {
	std::vector<std::string> tokens;
	std::istringstream tokenStream(line);
	std::string token;

	while (std::getline(tokenStream, token, delimiter)) {
		if (!token.empty()) { // Optionally skip empty tokens
			tokens.push_back(token);
		}
	}

	return tokens;
}

void f(std::string path){
	
}
