#include <eigen3/Eigen/Dense> 
#include <vector>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define IS_IN_RANGE3(x1, y1, z1) ((x1) >= 0 && (x1) < dimx && (y1) >= 0 && (y1) < dimy && (z1) >= 0 && (z1) < dimz)
#define INDEX3(dz, dy, dx) (IS_IN_RANGE3((x + dx), (y + dy), (z + dz)) ? *ptr.data(i, 0, z + dz,  y + dy, x + dx) : 0)
#define CHECK3() \
		  (INDEX3(-1, -1, -1) || INDEX3(-1, -1, 0) || INDEX3(-1, -1, 1) \
		||	INDEX3(-1, 0, -1)  || INDEX3(-1, 0, 0)  || INDEX3(-1, 0, 1) \
		||	INDEX3(-1, 1, -1)  || INDEX3(-1, 1, 0)  || INDEX3(-1, 1, 1) \
		||	INDEX3(0, -1, -1)  || INDEX3(0, -1, 0)  || INDEX3(0, -1, 1) \
		||	INDEX3(0, 0,  -1)   || INDEX3(0, 0, 0)   || INDEX3(0, 0, 1) \
		||	INDEX3(0, 1,  -1)   || INDEX3(0, 1, 0)   || INDEX3(0, 1, 1) \
		||	INDEX3(1, -1, -1) || INDEX3(1, -1, 0) || INDEX3(1, -1, 1) \
		||	INDEX3(1, 0,  -1)  || INDEX3(1, 0, 0)  || INDEX3(1, 0, 1) \
		||	INDEX3(1, 1,  -1)  || INDEX3(1, 1, 0)  || INDEX3(1, 1, 1)) \

#define IS_IN_RANGE1(z1) ((z1) >= 0 && (z1) < dimz)
#define INDEX1(dz) (IS_IN_RANGE1((z + dz)) ? *ptr.data(i, z + dz) : 0)
#define CHECK1() \
		  (INDEX1(-1) || INDEX1(0) || INDEX1(1)) \

int count_tp1(py::array_t<bool> &p, py::array_t<bool> &gt) {
	assert(p.ndim() == 2);
	assert(gt.ndim() == 2);
	const int n_batch = p.shape(0);
	const int dimz = p.shape(1);

	auto& ptr = p;
	int counter = 0;
	for (int i = 0; i < n_batch; i++) 
		for (int z = 0 ; z < dimz ; z++)
			if (*gt.data(i, z)) {
				counter += CHECK1();
			}
	return counter;
}

int count_tp3(py::array_t<bool> &p, py::array_t<bool> &gt) {
	assert(p.ndim() == 5);
	assert(gt.ndim() == 5);
	const int n_batch = p.shape(0);
	assert(p.shape(1) == 1);
	const int dimz = p.shape(2);
	const int dimy = p.shape(3);
	const int dimx = p.shape(4);

	auto& ptr = p;
	int counter = 0;
	for (int i = 0; i < n_batch; i++) 
		for (int z = 0 ; z < dimz ; z++)
			for (int y = 0; y <  dimy; y++)
				for (int x = 0; x < dimx; x++) {
					if (*gt.data(i, 0, z, y, x)) {
						counter += CHECK3();
						//printf("i %d x %d y %d z %d in %d\n", i, x, y, z, res);
					}
				}
	return counter;
}

void extend3(py::array_t<bool> &in, py::array_t<bool> &out) {
	assert(in.ndim() == 5);
	assert(out.ndim() == 5);
	int n_batch = in.shape(0);
	assert(in.shape(1) == 1);
	int dimz = in.shape(2);
	int dimy = in.shape(3);
	int dimx = in.shape(4);

	auto& ptr = in;
	for (int i = 0; i < n_batch; i++) 
		for (int z = 1 ; z < dimz - 1; z++)
			for (int y = 1; y <  dimy - 1; y++)
				for (int x = 1; x < dimx - 1; x++) {
					*out.mutable_data(i, 0, z, y, x) = CHECK3();
				}
}


PYBIND11_MODULE(ExtendBox, m) {
	m.def("count_tp1", &count_tp1, "count tp func");
	m.def("count_tp3", &count_tp3, "count tp func");
	m.def("extend3", &extend3, "extend3 func");
}
