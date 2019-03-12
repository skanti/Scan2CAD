#include <eigen3/Eigen/Dense>
#include <vector>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "LoaderVOX.h"

namespace py = pybind11;


inline static void crop_center(int x, int y, int z, int window_size, float trunc, Vox &vox, Vox &out) {
	out.dims = Eigen::Vector3i::Constant(window_size);
	out.res = vox.res;
	out.grid2world = vox.grid2world;
	out.grid2world.col(3) += vox.grid2world*Eigen::Vector4f(x - window_size/2, y - window_size/2, z - window_size/2, 0);

	int n_elems = window_size*window_size*window_size;
	out.sdf = std::vector<float>(n_elems, trunc);

	int dimx = vox.dims(0);
	int dimy = vox.dims(1);
	int dimz = vox.dims(2);

	for (int k = 0 ; k < window_size; k++){
		for (int j = 0; j <  window_size; j++){
			for (int i = 0; i < window_size; i++){
				int k0 = k + z - std::floor(window_size/2);
				int j0 = j + y - std::floor(window_size/2);
				int i0 = i + x - std::floor(window_size/2);

				if (k0 >= 0 && k0 < dimz && j0 >= 0 && j0 < dimy && i0 >= 0 && i0 < dimx) {
					int index0  = k0*dimy*dimx + j0*dimx + i0;
					int index1  = k*window_size*window_size + j*window_size + i;
					out.sdf[index1] = vox.sdf[index0];
				}
			}
		}
	}
}

void crop_and_save(int window_size, float trunc, py::array_t<float> &kps, std::string filename_scan, std::string customname_out) {
	assert(kps.ndim() == 2 && "Keypoints array must be 2D array!");
	assert(kps.shape(0) == 3 && "Keypoints array must have n_cols = 3!");
	const int n_kps = kps.shape(1);
	assert(n_kps > 0 && "Keypoints array must have n_rows > 0!");
	assert(trunc < 0 && "Truncation must be negative.");
	assert(window_size%2 == 1 && "Window size must be uneven for a symmetric crop.");

	Eigen::MatrixXf kps1 = Eigen::Map<Eigen::MatrixXf>(kps.mutable_data(), 3, n_kps);


	Vox vox = load_vox(filename_scan);
	for (int i = 0; i < kps1.cols(); i++) {
		Vox chunk;
		Eigen::Vector3i p = (vox.grid2world.inverse().eval()*Eigen::Vector4f(kps1(0, i), kps1(1, i), kps1(2, i), 1.0f)).topRows(3).array().round().cast<int>();
		crop_center(p(0), p(1), p(2), window_size, trunc, vox, chunk);

		std::string filename = customname_out + std::to_string(i) + std::string(".vox");
		save_vox(filename, chunk);
	}
}


PYBIND11_MODULE(CropCentered, m) {
	m.def("crop_and_save", &crop_and_save, "crop function");
}
