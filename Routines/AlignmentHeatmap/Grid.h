#pragma once
#include <eigen3/Eigen/Dense>
#include "LoaderVOX.h"

template <typename T>
struct Grid {
	Grid() {
		dims = Eigen::Vector3i(0, 0, 0);
	}

	Grid(Eigen::Vector3i dims_, Eigen::Matrix4d world2grid_) {
		dims = dims_;
		world2grid = world2grid_;
		data.resize(dims(0)*dims(1)*dims(2));
		std::fill(data.begin(), data.end(), T(0));
	}

	Grid(Eigen::Vector3i dims_, Eigen::Matrix4d world2grid_, std::vector<T> &data_) {
		dims = dims_;
		world2grid = world2grid_;
		data = data_;
	}


	T operator()(double x, double y, double z) const {
		int xi = std::floor(x);
		int yi = std::floor(y);
		int zi = std::floor(z);
		double tx = x - xi;
		double ty = y - yi;
		double tz = z - zi;

		const T c000 = (*this)(xi, yi, zi); 
		const T c100 = (*this)(xi + 1, yi, zi); 
		const T c010 = (*this)(xi, yi + 1, zi); 
		const T c001 = (*this)(xi, yi, zi + 1); 
		const T c110 = (*this)(xi + 1, yi + 1, zi); 
		const T c101 = (*this)(xi + 1, yi, zi + 1); 
		const T c011 = (*this)(xi, yi + 1, zi + 1); 
		const T c111 = (*this)(xi + 1, yi + 1, zi + 1); 

		return 
			(T(1) - tx) * (T(1) - ty) * (T(1) - tz) * c000 + 
			tx * (T(1) - ty) * (T(1) - tz) * c100 + 
			(T(1) - tx) * ty * (T(1) - tz) * c010 + 
			tx * ty * (T(1) - tz) * c110 + 
			(T(1) - tx) * (T(1) - ty) * tz * c001 + 
			tx * (T(1) - ty) * tz * c101 + 
			(T(1) - tx) * ty * tz * c011 + 
			tx * ty * tz * c111; 
	}

	T operator()(int x, int y, int z) const {
		if (x >= 0 && x < dims(0) && y >= 0 && y < dims(1) && z >= 0 && z < dims(2))
			return data[z*dims(1)*dims(0) + y*dims(0) + x];
		else
			return 0;
	}
	
	void set(int x, int y, int z, T val) {
		if (x >= 0 && x < dims(0) && y >= 0 && y < dims(1) && z >= 0 && z < dims(2))
			data[z*dims(1)*dims(0) + y*dims(0) + x] = val;
	}

	void pad(T val = 0) {
		Eigen::Vector3i dims1 = dims*2;
		std::vector<T> data1(dims1(0)*dims1(1)*dims1(2), val);

		for (int z = 0; z < dims(2); z++)  {
			for (int y = 0; y < dims(1); y++)  {
				for (int x = 0; x < dims(0); x++)  {
					T v = (*this)(x, y, z);
					int x1 = x + dims(0)/2, y1 = y + dims(1)/2, z1 = z + dims(2)/2;
					data1[z1*dims1(1)*dims1(0) + y1*dims1(0) + x1] = v;
				}
			}
		}

		world2grid.block(0, 3, 3, 1) += dims.cast<double>()/2.0;
		dims = dims1;
		data = data1;
}

	void downsample2x(Grid<T> &g) {
		dims = g.dims/2;
		world2grid = g.world2grid;
		world2grid.block(0, 0, 3, 4) /= 2;
		data.resize(dims(0)*dims(1)*dims(2));
		std::fill(data.begin(), data.end(), T(0));

		for (int z = 0; z < dims(2); z++) {
			for (int y = 0; y < dims(1); y++) {
				for (int x = 0; x < dims(0); x++) {
					T v = (g(2*x, 2*y, 2*z) + g(2*x + 1, 2*y, 2*z) + g(2*x, 2*y + 1, 2*z) +  g(2*x, 2*y, 2*z + 1) + g(2*x + 1, 2*y + 1, 2*z) + g(2*x + 1, 2*y, 2*z + 1) + g(2*x, 2*y + 1, 2*z + 1) + g(2*x + 1, 2*y + 1, 2*z + 1))/8.0;
					set(x, y, z, v);
				}
			}
		}
	}

	void blur(int n_passes) {
		for (int i = 0; i < n_passes; i++) {

			Grid tmp(*this);
			for (int z = 0; z < dims(2); z++)
				for (int y = 0; y < dims(1); y++)
					for (int x = 0; x < dims(0); x++)
						tmp.set(x, y, z, ((*this)(x + 2, y, z) + (*this)(x + 1, y, z) + (*this)(x, y, z) + (*this)(x - 1, y, z) + (*this)(x - 2, y, z) )/5.0);

			std::copy(tmp.data.begin(), tmp.data.end(), data.begin());

			for (int z = 0; z < dims(2); z++)
				for (int y = 0; y < dims(1); y++)
					for (int x = 0; x < dims(0); x++)
						tmp.set(x, y, z, ((*this)(x, y + 2, z) + (*this)(x, y + 1, z) + (*this)(x, y, z) + (*this)(x, y - 1, z) + (*this)(x, y - 2, z) )/5.0);

			std::copy(tmp.data.begin(), tmp.data.end(), data.begin());

			for (int z = 0; z < dims(2); z++)
				for (int y = 0; y < dims(1); y++)
					for (int x = 0; x < dims(0); x++)
						tmp.set(x, y, z, ((*this)(x, y, z + 2) + (*this)(x, y, z + 1) + (*this)(x, y, z) + (*this)(x, y, z - 1) + (*this)(x, y, z - 2) )/5.0);


			std::copy(tmp.data.begin(), tmp.data.end(), data.begin());
			
		}
		double fmax = *std::max_element(data.begin(), data.end());
		for (int z = 0; z < dims(2); z++)
			for (int y = 0; y < dims(1); y++)
				for (int x = 0; x < dims(0); x++)
					(*this).set(x, y, z, (*this)(x, y, z)/fmax);

	}

	static void calc_gradient(Grid<T> &pdf, Grid<T> &gradientx, Grid<T> &gradienty, Grid<T> &gradientz) {
		gradientx = Grid<T>(pdf.dims, pdf.world2grid);
		gradienty = Grid<T>(pdf.dims, pdf.world2grid);
		gradientz = Grid<T>(pdf.dims, pdf.world2grid);

		for (int z = 0; z < pdf.dims(2); z++)  {
			for (int y = 0; y < pdf.dims(1); y++)  {
				for (int x = 0; x < pdf.dims(0); x++)  {
					gradientx.set(x, y, z, 0.5*(pdf(x + 1, y, z) - pdf(x - 1, y, z)));
					gradienty.set(x, y, z, 0.5*(pdf(x, y + 1, z) - pdf(x, y - 1, z)));
					gradientz.set(x, y, z, 0.5*(pdf(x, y, z + 1) - pdf(x, y, z - 1)));
				}
			}
		}
	};

	Eigen::Vector3i dims;
	Eigen::Matrix4d world2grid;
	std::vector<T> data;
};
