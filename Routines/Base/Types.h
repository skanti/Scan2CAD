#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

struct Vox {
	Eigen::Vector3i grid_dims;
	float res;
	Eigen::Matrix4f grid2world;
	std::vector<float> sdf;
	std::vector<float> pdf;
};
