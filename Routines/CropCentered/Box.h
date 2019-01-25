#pragma once

void create_box(Eigen::MatrixXf &verts, Eigen::MatrixXi &elems) {
	float vertices[8*3] = {
		-1, -1,  1,
		 1, -1,  1,
		 1,  1,  1,
		-1,  1,  1,
		-1, -1, -1,
		 1, -1, -1,
		 1,  1, -1,
		-1,  1, -1,
	};

	float elements[12*3] = {
		// front
		0, 1, 2,
		2, 3, 0,
		// top
		1, 5, 6,
		6, 2, 1,
		// back
		7, 6, 5,
		5, 4, 7,
		// bottom
		4, 0, 3,
		3, 7, 4,
		// left
		4, 5, 1,
		1, 0, 4,
		// right
		3, 2, 6,
		6, 7, 3,
	};

	int n_vertices = 8;
	int n_elements = 12;
	verts.resize(3, n_vertices);
	elems.resize(3, n_elements);
	std::copy(vertices, vertices + 3*n_vertices, verts.data());
	std::copy(elements, elements + 3*n_elements, elems.data());
}
