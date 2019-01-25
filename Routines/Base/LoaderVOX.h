#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>

//#include <sys/stat.h>
//#include <sys/types.h>
//#include <sys/file.h>
//#include <sys/fcntl.h>
//#include <unistd.h>
//
//struct LockInfo {
//	int fd;
//	std::string file;
//};
//
//LockInfo lock(std::string filename) {
//	LockInfo li;
//	li.file = filename + ".lock";
//	while(1) {
//		li.fd = open(li.file.c_str(), O_CREAT);
//		flock(li.fd, LOCK_EX);
//		
//		struct stat st0 = {};
//		struct stat st1 = {};
//		fstat(li.fd, &st0);
//		stat(li.file.c_str(), &st1);
//		if(st0.st_ino == st1.st_ino) break;
//
//		close(li.fd);
//	}
//	return li;
//}
//
//void unlock(LockInfo &li) {
//	unlink(li.file.c_str());
//	flock(li.fd, LOCK_UN);
//}
template<typename T, int n_channel>
static void load_vox(std::string filename, Eigen::Vector3i &grid_dims, float &res, Eigen::Matrix4f &grid2world, std::vector<T> &data, bool is_col_major=true) {

	std::ifstream inFile(filename, std::ios::binary);
	assert(inFile.is_open());

	
	inFile.read((char*)grid_dims.data(), 3*sizeof(int32_t));
	inFile.read((char*)&res, sizeof(float));
	inFile.read((char*)grid2world.data(), 16*sizeof(float));
	if (!is_col_major)
		grid2world.transposeInPlace();
	
	int n_elems = grid_dims(0)*grid_dims(1)*grid_dims(2);	
	
	data.resize(n_elems);
	inFile.read((char*)data.data(), n_elems*n_channel*sizeof(T));
	inFile.close();	
	
}

template<typename T0, int n_channel0, typename T1, int n_channel1>
static void load_vox(std::string filename, Eigen::Vector3i &grid_dims, float &res, Eigen::Matrix4f &grid2world, std::vector<T0> &data0, std::vector<T1> &data1) {
	
	std::ifstream inFile(filename, std::ios::binary);
	assert(inFile.is_open());

	inFile.read((char*)grid_dims.data(), 3*sizeof(int32_t));
	inFile.read((char*)&res, sizeof(float));
	inFile.read((char*)grid2world.data(), 16*sizeof(float));
	
	int n_elems = grid_dims(0)*grid_dims(1)*grid_dims(2);	
	
	data0.resize(n_channel0*n_elems);
	data1.resize(n_channel1*n_elems);
	inFile.read((char*)data0.data(), n_elems*n_channel0*sizeof(T0));
	inFile.read((char*)data1.data(), n_elems*n_channel1*sizeof(T1));
	inFile.close();	
	
}

template<typename T, int n_channel>
static void save_vox(std::string filename, Eigen::Vector3i grid_dims, float res, Eigen::Matrix4f grid2world, std::vector<T> &data) {
	std::ofstream outFile;
	outFile.open(filename, std::ofstream::out | std::ios::binary);
	assert(outFile.is_open());
	outFile.write((char*)grid_dims.data(), 3*sizeof(int32_t));
	outFile.write((char*)&res, sizeof(float));
	
	int n_size = n_channel*grid_dims(0)*grid_dims(1)*grid_dims(2);
	outFile.write((char*)grid2world.data(), 16*sizeof(float));
	outFile.write((char*)data.data(), n_channel*n_size*sizeof(float));
	outFile.close();
	
}

template<typename T0, int n_channels0, typename T1, int n_channels1>
static void save_vox(std::string filename, Eigen::Vector3i grid_dims, float res, Eigen::Matrix4f grid2world, std::vector<T0> &data0, std::vector<T1> &data1) {
	std::ofstream outFile;
	outFile.open(filename, std::ofstream::out | std::ios::binary);
	assert(outFile.is_open());
	outFile.write((char*)grid_dims.data(), 3*sizeof(int32_t));
	outFile.write((char*)&res, sizeof(float));
	outFile.write((char*)grid2world.data(), 16*sizeof(float));
	
	int n_size = grid_dims(0)*grid_dims(1)*grid_dims(2);
	outFile.write((char*)data0.data(), n_channels0*n_size*sizeof(T0));
	outFile.write((char*)data1.data(), n_channels1*n_size*sizeof(T1));
	outFile.close();
	
}
