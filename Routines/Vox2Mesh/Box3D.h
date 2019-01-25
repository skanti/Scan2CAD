#pragma once

#include <eigen3/Eigen/Dense>

namespace Box3D {

	void create(Eigen::MatrixXf &vertices0, Eigen::MatrixXf &normals0, Eigen::Matrix<uint32_t, -1, -1> &elements0) {
		// -> vertices
		const float vertices[] = {
			-1.0, -1.0,  1.0,
			1.0, -1.0,  1.0,
			1.0,  1.0,  1.0,
			-1.0,  1.0,  1.0,
			// back
			-1.0, -1.0, -1.0,
			1.0, -1.0, -1.0,
			1.0,  1.0, -1.0,
			-1.0,  1.0, -1.0,
		};
		// <-

		// -> normals
		const float normals[] = {
			-1,  0,  0, -1,  0,  0, -1,  0,  0, // Left Side
			-1,  0,  0, -1,  0,  0, -1,  0,  0, // Left Side
			0,  0, -1,  0,  0, -1,  0,  0, -1, // Back Side
			0,  0, -1,  0,  0, -1,  0,  0, -1, // Back Side
			0, -1,  0,  0, -1,  0,  0, -1,  0, // Bottom Side
			0, -1,  0,  0, -1,  0,  0, -1,  0, // Bottom Side
			0,  0,  1,  0,  0,  1,  0,  0,  1, // front Side
			0,  0,  1,  0,  0,  1,  0,  0,  1, // front Side
			1,  0,  0,  1,  0,  0,  1,  0,  0, // right Side
			1,  0,  0,  1,  0,  0,  1,  0,  0, // right Side
			0,  1,  0,  0,  1,  0,  0,  1,  0, // top Side
			0,  1,  0,  0,  1,  0,  0,  1,  0, // top Side
		};
		// <-

		// -> elements
		const float elements[] = {
			0, 1, 2,
			2, 3, 0,
			// right
			1, 5, 6,
			6, 2, 1,
			// back
			7, 6, 5,
			5, 4, 7,
			// left
			4, 0, 3,
			3, 7, 4,
			// bottom
			4, 5, 1,
			1, 0, 4,
			// top
			3, 2, 6,
			6, 7, 3,
		};
		// <-

		int dim = 3;
		int n_vertices = 8;
		int n_elements = 12;

		vertices0.resize(dim, n_vertices);
		normals0.resize(dim, n_vertices);
		elements0.resize(dim, n_elements);

		std::copy(vertices, vertices + dim*n_vertices, vertices0.data());
		std::copy(normals, normals + dim*n_vertices, normals0.data());
		std::copy(elements, elements + dim*n_elements, elements0.data());

	}
}
