#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>


struct Vox {
	Eigen::Vector3i dims;
	float res;
	Eigen::Matrix4f grid2world;
	std::vector<float> sdf;
	std::vector<float> pdf;
};

inline static Vox load_vox(std::string filename) {
	
	std::ifstream f(filename, std::ios::binary);
	assert(f.is_open());

	bool is_row_major = false;

	std::string extension = filename.substr(filename.find_last_of(".") + 1);
	if (extension == "df")
		is_row_major = true;
	else if (extension == "sdf")
		is_row_major = true;
	
	Vox vox;

	f.read((char*)vox.dims.data(), 3*sizeof(int32_t));
	f.read((char*)&vox.res, sizeof(float));
	f.read((char*)vox.grid2world.data(), 16*sizeof(float));
	if (is_row_major)
		vox.grid2world = vox.grid2world.transpose().eval();
	
	int n_elems = vox.dims(0)*vox.dims(1)*vox.dims(2);	
	
	vox.sdf.resize(n_elems);
	f.read((char*)vox.sdf.data(), n_elems*sizeof(float));

	if(f && f.peek() != EOF) {
		vox.pdf.resize(n_elems);
		f.read((char*)vox.pdf.data(), n_elems*sizeof(float));
	}
	f.close();	

	return vox;
}

inline static void save_vox(std::string filename, Vox &vox) {
	std::ofstream f;
	f.open(filename, std::ofstream::out | std::ios::binary);
	assert(f.is_open());
	f.write((char*)vox.dims.data(), 3*sizeof(int32_t));
	f.write((char*)&vox.res, sizeof(float));
	f.write((char*)vox.grid2world.data(), 16*sizeof(float));
	
	int n_size = vox.dims(0)*vox.dims(1)*vox.dims(2);
	if (vox.sdf.size() > 0)
		f.write((char*)vox.sdf.data(), n_size*sizeof(float));
	if (vox.pdf.size() > 0)
		f.write((char*)vox.pdf.data(), n_size*sizeof(float));
	f.close();
	
}
