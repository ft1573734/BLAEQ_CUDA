#include "BLAEQ.cu"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

void Load_Pcd_Data(std::string path, Multidimensional_Arr* data_set);
void Load_Queries(std::string path, int D, double* workload, int Q_count);
std::vector<std::string> splitLine(const std::string& line, char delimiter);

int main() {
	//Load data
	std::string input_dir = "D:\\raw_data\\synthetic.pcd";
	std::string input_query = "D:\\raw_data\\queries.txt";
	Multidimensional_Arr* dataset;
	Load_Pcd_Data(input_dir, dataset);

	int K = 100;

	//Initialize BLAEQ
	BLAEQ BLAEQ_Object = BLAEQ(dataset-> N, dataset-> D, dataset->data, K);

	/*
	* Perform Query
	*/

	//Load query file
	double* workload;
	int Q_count = 10;
	Load_Queries(input_query, dataset->D, workload, Q_count);


	//Perform query
	BLAEQ_Object.BLAEQ_Query(workload, Q_count);

	return 0;
}


void Load_Pcd_Data(std::string path, Multidimensional_Arr* data_set) {
	pcl::PointCloud<pcl::PointXYZ>* cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1) {
		PCL_ERROR("PCL: Couldn't read the file \n");
	}

	// Print the cloud data
	//std::cerr << "Loaded " << cloud->width * cloud->height << " data points from test_pcd.pcd with the following fields: " << std::endl;
	//for (const auto& point : *cloud) {
	//	std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;
	//}

	Multidimensional_Arr original_dataset = Multidimensional_Arr(cloud->width, cloud->height);
	//Loading 3-D data
	if (original_dataset.D == 3) {
		pcl::PointCloud<pcl::PointXYZ>::const_iterator ite = cloud->cbegin();
		for (int i = 0; i < original_dataset.N; i++) {
			original_dataset.data[i + 0 * original_dataset.N] = ite->x;
			original_dataset.data[i + 1 * original_dataset.N] = ite->y;
			original_dataset.data[i + 2 * original_dataset.N] = ite->z;
			ite++;
		}
	}
	free(cloud);
	data_set = &original_dataset;
}

void Load_Queries(std::string path, int D, double* workload, int Q_count) {
	std::ifstream file(path);
	std::vector<double> numbers;
	std::string line;
	std::string token;
	double value;

	if (!file.is_open()) {
		std::cerr << "Could not open the file - '" << path << "'" << std::endl;
	}

	double* query_boundaries = static_cast<double*>(std::malloc(Q_count * D * 2 * sizeof(double)));

	for (int i = 0; i < Q_count; i++) {
		std::getline(file, token);
		std::vector<std::string> tokens = splitLine(line, ' ');
		std::vector<std::string>::iterator ite = tokens.begin();
		for (int j = 0; j < D * 2; j++) {
			query_boundaries[i * D * 2 + j] = std::stod(*ite);
			ite++;
		}
	}

	workload = query_boundaries;

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