#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <set>
#include <unordered_map>
#include <time.h>
#include <limits>

#include <omp.h>
#include <eigen3/Eigen/Dense>

#include "LoaderVOX.h"
#include "Mat4.h"
#include <ceres/ceres.h>
#include <json/json.h>

#include "args.hxx"

#include "Grid.h"

#include <iostream>


Vox vox_scan;
std::vector<Vox> cads;
std::vector<std::vector<Grid<float>>> pdf, gradientx, gradienty, gradientz;
Eigen::Matrix<double, -1, -1> centers;
Eigen::Matrix<double, -1, -1> scales;
matrix_lib::Mat4<double> M00; // <-- as scan2cad

struct InputArgs {
	std::string json;
	std::string out;
} inargs;

void compose_mat4_from_tqs(Eigen::Vector3d &t, Eigen::Quaterniond &q, Eigen::Vector3d &s, Eigen::Matrix4d &M) {
	Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d S = Eigen::Matrix4d::Identity();
	
	T.block(0, 3, 3, 1) = t;
	R.block(0, 0, 3, 3) = q.toRotationMatrix();
	S.block(0, 0, 3, 3) = (Eigen::Matrix3d)s.asDiagonal();

	M = T*R*S;
}

void decompose_mat4_into_tqs(Eigen::Matrix4d M, Eigen::Vector3d &t, Eigen::Quaterniond &q, Eigen::Vector3d &s) {

	Eigen::Matrix3d rot = M.block(0, 0, 3, 3);
	double sx = rot.col(0).norm();
	double sy = rot.col(1).norm();
	double sz = rot.col(2).norm();
	s = Eigen::Vector3d(sx, sy, sz);

	rot.col(0) /= sx;
	rot.col(1) /= sy;
	rot.col(2) /= sz;
	q = Eigen::Quaterniond(rot);
	q.normalize();

	t = M.col(3).topRows(3);
}

template <typename T>
void calc_gradient(Grid<T> &pdf, Grid<T> &gradientx, Grid<T> &gradienty, Grid<T> &gradientz) {
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
}

double calc_li_confidence(Vox &vox_cad, Vox &vox_scan, Eigen::Matrix4f M_cad2scan) {
	double cost = 0;
	int counter = 0;
	int counter_total = 0;
	for (int k = 0; k < vox_cad.dims(2); k++) {
		for (int j = 0; j < vox_cad.dims(1); j++) {
			for (int i = 0; i < vox_cad.dims(0); i++) {
				if (std::abs(vox_cad.sdf[k*vox_cad.dims(1)*vox_cad.dims(0) + j*vox_cad.dims(0) + i]) <= vox_cad.res) {
					counter_total++; // <-- count CAD surface voxel
					Eigen::Vector4f p_cad = vox_cad.grid2world*Eigen::Vector4f(i, j, k, 1);
					Eigen::Vector4f p_scan = vox_scan.grid2world.inverse().eval()*M_cad2scan*p_cad;
					Eigen::Vector3i p  = p_scan.topRows(3).array().round().cast<int>();
					if ((p.array() >= Eigen::Array3i(0, 0, 0)).all() && (p.array() < vox_scan.dims.array()).all()) {
						float val = vox_scan.sdf[p(2)*vox_scan.dims(1)*vox_scan.dims(0) + p(1)*vox_scan.dims(0) + p(0)];
						bool is_hit = val >= -2*vox_scan.res;
						if (is_hit) {
							cost += val*val;
							counter++;
						}
					}
				}
			}
		}
	}

	if (counter/(double)counter_total < 0.3)
		cost = 1e6;
	else
		cost /= counter;
		
	return cost;
}

void read_json_and_load_models(Eigen::Matrix<double, -1, -1> &centers, Eigen::Matrix<double, -1, -1> &scales, std::vector<Vox> &heatmaps, Vox &vox_scan) {
	std::ifstream file(inargs.json);
	assert(file.is_open() && "Cannot open json file.");
	Json::Value root;
	file >> root;
	std::string filename_vox_scan = root["filename_vox_scan"].asString();
	vox_scan = load_vox(filename_vox_scan);
	Eigen::Vector3d t(0,0,0);
	Eigen::Quaterniond q = Eigen::Quaterniond(root["rot"][0].asDouble(), root["rot"][1].asDouble(), root["rot"][2].asDouble(), root["rot"][3].asDouble());
	Eigen::Vector3d s(1,1,1);

	Eigen::Matrix4d M_cad2scan;
	compose_mat4_from_tqs(t, q, s, M_cad2scan);
	M00 = matrix_lib::Mat4<double>(M_cad2scan.data());
	M00.transpose();
	M00 = M00.getInverse();

	for (Json::Value::ArrayIndex i = 0; i != root["pairs"].size(); i++) {
		Vox heatmap = load_vox(root["pairs"][i]["filename_heatmap"].asString());
		heatmaps.push_back(heatmap);

		Eigen::Vector4d p_center = Eigen::Vector4d(root["pairs"][i]["p_scan"][0].asDouble(), root["pairs"][i]["p_scan"][1].asDouble(), root["pairs"][i]["p_scan"][2].asDouble(), 1);
		centers.conservativeResize(4, centers.cols() + 1);
		centers.col(centers.cols() - 1) = p_center;
		
		Eigen::Vector3d scale = Eigen::Vector3d(root["pairs"][i]["scale"][0].asDouble(), root["pairs"][i]["scale"][1].asDouble(), root["pairs"][i]["scale"][2].asDouble());
		scales.conservativeResize(3, scales.cols() + 1);
		scales.col(scales.cols() - 1) = scale;
	}
}


std::vector<std::pair<Eigen::Vector3d, Eigen::MatrixXd>> create_correspondences(Eigen::MatrixXd &kp0, Eigen::MatrixXd &kp1) {
	std::vector<std::pair<Eigen::Vector3d, Eigen::MatrixXd>> pairs;

	auto update_or_pushback = [&](auto &a, auto &b) {
		for (auto &p : pairs) {
			if ((a.array() == p.first.array()).all()) {
				auto &sec = p.second;
				sec.conservativeResize(3, sec.cols() + 1);
				sec.col(sec.cols() - 1) = b;
				return;
			}
		};
		Eigen::Vector3d first = a;
		Eigen::MatrixXd second(3, 1);
		second.col(0) = b;
		pairs.push_back(std::make_pair(first, second));
	};

	for (int i = 0; i < kp0.cols(); i++) {
		auto a = kp0.col(i);
		auto b = kp1.col(i);
		update_or_pushback(a, b);
	}
	return pairs;
}

Eigen::MatrixXd remove_duplicates(Eigen::MatrixXd &in) {
	Eigen::MatrixXd out(3, 0);
	for (int i = 0; i < in.cols(); i++) {
		bool is_duplicate = false;
		for (int j = 0; j < out.cols(); j++)
			is_duplicate |= (in.col(i).array() == out.col(j).array()).all();
		if (!is_duplicate) {
			out.conservativeResize(3, out.cols() + 1);
			out.col(out.cols() - 1) = in.col(i);
		}
	}
	return out;
}

std::pair<int, double> calc_nn(Eigen::Vector3d a, Eigen::MatrixXd b) {
	int index = -1;
	double dmin = std::numeric_limits<double>::max();

	for (int i = 0; i < b.cols(); i++) {
		double d = (b.col(i) - a).norm();
		if (d < dmin) {
			index = i;
			dmin = d;
		}
	}

	std::pair<int, double> nn(index, dmin);
	return nn;
}


template <typename T>
matrix_lib::Mat4<T> R9toSE3(Eigen::VectorXd &x) {
	matrix_lib::Mat4<T> M00 = matrix_lib::Mat4<T>::R6toSE3(T(x[0]), T(x[1]), T(x[2]), T(x[3]), T(x[4]), T(x[5]));
	matrix_lib::Mat4<T> s00 = matrix_lib::Mat4<T>::diag(T(x[6]), T(x[7]), T(x[8]), T(1));
	M00 = M00*s00;
	return M00;
}


template <typename T>
void SE3toR9(Eigen::Matrix<T, 4, 4> M, Eigen::Matrix<T, 3, 1> &t, Eigen::Matrix<T, 3, 1> &r, Eigen::Matrix<T, 3, 1> &s) {
	Eigen::Matrix<T, 3, 3> R = M.block(0, 0, 3, 3);
	
	T phi = std::acos((R.trace() - 1)*0.5);

	Eigen::Matrix<T, 3, 3> lnR = (phi/(2*std::sin(phi)))*(R - R.transpose());
	r = Eigen::Matrix<T, 3, 1>(lnR(2,1), lnR(0, 2), lnR(1, 0));
	const T theta = r.norm();
	T theta2 = theta*theta;
	T si = std::sin(theta);
	T co = std::cos(theta);
	const T t0 = theta == 0 ? T(1.0) : si/theta;
	const T t1 = theta == 0 ? T(0.5) : (T(1.0) - co)/theta2;
	const T t2 = theta == 0 ? T(1.0/6.0) : (theta - si)/(theta2*theta);
	
	Eigen::Matrix<T, 3, 3> S;
	S << 0, -r(2), r(1), r(2), 0, -r(0), -r(1), r(0), 0;
	Eigen::Matrix<T, 3, 3> V = Eigen::Matrix<T, 3, 3>::Identity() + S*t1 + S*S*t2;
	//Eigen::Matrix<T, 3, 3> Vinv = Eigen::Matrix<T, 3, 3>::Identity() - s*0.5 + (2*);
	
	double sx = R.col(0).norm();
	double sy = R.col(1).norm();
	double sz = R.col(2).norm();
	s = Eigen::Matrix<T, 3, 1>(sx, sy, sz);

	t = V.inverse().eval()*M.block(0, 3, 3, 1);

}

template <typename T>
Eigen::Matrix<T, 4, 4> R6toSE3(Eigen::Matrix<T, 6, 1> a) {
	Eigen::Matrix<T, 3, 3> S;
	S << 0, -a(2), a(1), a(2), 0, -a(0), -a(1), a(0), 0;
	T theta2 = a(0)*a(0) + a(1)*a(1) + a(2)*a(2);
	T theta = std::sqrt(theta2);
	T s = std::sin(theta);
	T c = std::cos(theta);
	const T t0 = theta == 0 ? T(1.0) : s/theta;
	const T t1 = theta == 0 ? T(0.5) : (T(1.0) - c)/theta2;
	const T t2 = theta == 0 ? T(1.0/6.0) : (theta - s)/(theta2*theta);

	Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity() + S*t0 + S*S*t1;
	Eigen::Matrix<T, 3, 3> V = Eigen::Matrix<T, 3, 3>::Identity() + S*t1 + S*S*t2;
	
	Eigen::Matrix<T, 4, 1> t;
	t.topRows(3) = V*a.bottomRows(3);
	t(3) = 1;

	Eigen::Matrix<T, 4, 4> M = Eigen::Matrix<T, 4, 4>::Identity();
	M.block(0, 0, 3, 3) = R;
	M.col(3) = t;
	return M;
}


void make_pyramide(std::vector<Vox> &cads) {
	const int n_level = 3;
	pdf.resize(n_level);
	gradientx.resize(n_level);
	gradienty.resize(n_level);
	gradientz.resize(n_level);
	for (int i = 0; i < n_level; i++) {
		pdf[i].resize(cads.size());
		gradientx[i].resize(cads.size());
		gradienty[i].resize(cads.size());
		gradientz[i].resize(cads.size());
	}

	for (int j = 0; j < (int)cads.size(); j++) {
		pdf[0][j] = Grid<float>(cads[j].dims, cads[j].grid2world.inverse().eval().cast<double>(), cads[j].pdf);
		pdf[0][j].pad();
		pdf[0][j].blur(2);
		Grid<float>::calc_gradient(pdf[0][j], gradientx[0][j], gradienty[0][j], gradientz[0][j]);
	}

	for (int i = 1; i < n_level; i++) {
		for (int j = 0; j < (int)cads.size(); j++) {
			pdf[i][j] = Grid<float>();
			pdf[i][j].downsample2x(pdf[i-1][j]);
			pdf[i][j].blur(1);
			Grid<float>::calc_gradient(pdf[i][j], gradientx[i][j], gradienty[i][j], gradientz[i][j]);
		}
	}
	auto sdftmp0 = Grid<float>(cads[0].dims, cads[0].grid2world.inverse().eval().cast<double>(), cads[0].sdf);
	sdftmp0.pad(5);
	auto sdftmp1 = Grid<float>();
	sdftmp1.downsample2x(sdftmp0);
	auto sdftmp2 = Grid<float>();
	sdftmp2.downsample2x(sdftmp1);
	//save_vox<float, 1, float, 1>("test2.vox2", pdf[2][0].dims, 4, pdf[2][0].world2grid.inverse().eval().cast<float>(), sdftmp2.data, pdf[2][0].data);
	//save_vox<float, 1, float, 1>("test1.vox2", pdf[1][0].dims, 2, pdf[1][0].world2grid.inverse().eval().cast<float>(), sdftmp1.data, pdf[1][0].data);
	//save_vox<float, 1, float, 1>("test0.vox2", pdf[0][0].dims, 1, pdf[0][0].world2grid.inverse().eval().cast<float>(), sdftmp0.data, pdf[0][0].data);
}

void parse_args(int argc, char** argv) {
	args::ArgumentParser parser("This is a test program.", "This goes after the options.");
	args::Group allgroup(parser, "", args::Group::Validators::All);

	args::ValueFlag<std::string> json(allgroup, "bunny.json", "predictions", {"json"});
	args::ValueFlag<std::string> out(allgroup, "bunny.out", "out file", {"out"});

	try {
		parser.ParseCLI(argc, argv);
	} catch (args::ParseError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	} catch (args::ValidationError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	}

	inargs.json = args::get(json);
	inargs.out = args::get(out);
};

template<typename S, typename T>
matrix_lib::Mat4<T> cast(const matrix_lib::Mat4<S> &m) {
	return matrix_lib::Mat4<T>(
			m(0, 0), m(0, 1), m(0, 2), m(0, 3),
			m(1, 0), m(1, 1), m(1, 2), m(1, 3),
			m(2, 0), m(2, 1), m(2, 2), m(2, 3),
			m(3, 0), m(3, 1), m(3, 2), m(3, 3)
			);
};


template<>
matrix_lib::Mat4<double> cast(const matrix_lib::Mat4<ceres::Jet<double, 9>> &m) {
	return matrix_lib::Mat4<double>(
			m(0, 0).a, m(0, 1).a, m(0, 2).a, m(0, 3).a,
			m(1, 0).a, m(1, 1).a, m(1, 2).a, m(1, 3).a,
			m(2, 0).a, m(2, 1).a, m(2, 2).a, m(2, 3).a,
			m(3, 0).a, m(3, 1).a, m(3, 2).a, m(3, 3).a
			);
};

template<>
matrix_lib::Mat4<ceres::Jet<double, 9>> cast(const matrix_lib::Mat4<double> &m) {
	return matrix_lib::Mat4<ceres::Jet<double, 9>>(
			ceres::Jet<double, 9>(m(0, 0)), ceres::Jet<double, 9>(m(0, 1)), ceres::Jet<double, 9>(m(0, 2)), ceres::Jet<double, 9>(m(0, 3)),
			ceres::Jet<double, 9>(m(1, 0)), ceres::Jet<double, 9>(m(1, 1)), ceres::Jet<double, 9>(m(1, 2)), ceres::Jet<double, 9>(m(1, 3)),
			ceres::Jet<double, 9>(m(2, 0)), ceres::Jet<double, 9>(m(2, 1)), ceres::Jet<double, 9>(m(2, 2)), ceres::Jet<double, 9>(m(2, 3)),
			ceres::Jet<double, 9>(m(3, 0)), ceres::Jet<double, 9>(m(3, 1)), ceres::Jet<double, 9>(m(3, 2)), ceres::Jet<double, 9>(m(3, 3))
			);
};

template matrix_lib::Mat4<double> cast<double, double>(const matrix_lib::Mat4<double> &m);
template matrix_lib::Mat4<ceres::Jet<double, 9>> cast<ceres::Jet<double, 9>, ceres::Jet<double, 9>>(const matrix_lib::Mat4<ceres::Jet<double, 9>> &m);


void print_mat(matrix_lib::Mat4<double> &M) {
	std::string result;
	//for (int i = 0; i < 16; i++)
	//	result += std::to_string(M[i]) + " ";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++)
			result += std::to_string(M[i*4 + j]) + " ";
		result += "\n";
	}
	std::cout << result << std::endl;
}

void print_mat(matrix_lib::Mat4<ceres::Jet<double, 9>> &M) {
	std::string result;
	//for (int i = 0; i < 16; i++)
	//	result += std::to_string(M[i].a) + " ";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++)
			result += std::to_string(M[i*4 + j].a) + " ";
		result += "\n";
	}
	std::cout << result << std::endl;
}

void ComputeGridValueAndJacobian(int i_level, int i_cad, const double x, const double y, const double z, double *value, double *jacobian) {
	value[0] = pdf[i_level][i_cad](x, y, z);
	if (jacobian) {
		jacobian[0] = gradientx[i_level][i_cad](x, y, z);
		jacobian[1] = gradienty[i_level][i_cad](x, y, z);
		jacobian[2] = gradientz[i_level][i_cad](x, y, z);
	} 
}

struct GridFunctor : public ceres::SizedCostFunction<1, 3> {
	GridFunctor(int i_level_, int i_cad_) {
		i_level = i_level_;
		i_cad = i_cad_;
	}
	virtual bool Evaluate(double const* const* params, double* residuals, double **jacobians) const {
		
		if (!jacobians)
			ComputeGridValueAndJacobian(i_level, i_cad, params[0][0], params[0][1], params[0][2], residuals, NULL);
		else
			ComputeGridValueAndJacobian(i_level, i_cad, params[0][0], params[0][1], params[0][2], residuals, jacobians[0]);

		return true;
	}
	int i_level, i_cad;
};

struct CostFunctorScale {
	template <typename T>
		bool operator()(const T* params, T* residual) const {

			const T fac = T(3.0);
			residual[0] = fac*(T(1.0) - params[6]);
			residual[1] = fac*(T(1.0) - params[7]);
			residual[2] = fac*(T(1.0) - params[8]);

			return true;
		}
};

struct CostFunctor {
	CostFunctor(int i_level_, int i_cad_) {
		i_level = i_level_;
		i_cad = i_cad_;
		func_grid.reset(new ceres::CostFunctionToFunctor<1, 3>(new GridFunctor(i_level, i_cad)));
	}

	template <typename T>
		bool operator()(const T* params, T* residual) const {

			matrix_lib::Mat4<T> P = cast<T, double>(M00);

			matrix_lib::Mat4<T> M1 = matrix_lib::Mat4<T>::R6toSE3(params[0], params[1], params[2], params[3], params[4], params[5]);
			matrix_lib::Mat4<T> s = matrix_lib::Mat4<T>::diag(params[6], params[7], params[8], T(1));
			matrix_lib::Mat4<T> M = M1*s*P;
			
			//for (int i = 0; i < (int) cads.size(); i++) {
			matrix_lib::Vec4<T> c(T(centers.col(i_cad)(0)), T(centers.col(i_cad)(1)), T(centers.col(i_cad)(2)), T(1));
			Eigen::Matrix<T, 4, 4, Eigen::RowMajor> world2grid0 = pdf[i_level][i_cad].world2grid.cast<T>();
			matrix_lib::Mat4<T> world2grid(world2grid0.data());

			matrix_lib::Vec4<T> a0 = world2grid*M*c;
			const T a[3] = {a0[0], a0[1], a0[2]};

			T f = T(0);
			(*func_grid)(a, &f);
			//if (f >= T(0.1))
				residual[0] = ceres::abs(T(1) -f);
			//else
			//	residual[0] = T(1);

			return true;
		}

	int i_level, i_cad;
	std::unique_ptr<ceres::CostFunctionToFunctor<1, 3> > func_grid;
};

class ParamsUpdateCallback : public ceres::IterationCallback {
	public:
	ParamsUpdateCallback(ceres::Problem *problem_) {
		problem = problem_;

	}
	ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
		//std::vector<double *> parameter_blocks;
		//problem->GetParameterBlocks(&parameter_blocks);
		//double *M = parameter_blocks[0];
		////if (summary.cost_change > 0) {

		//	matrix_lib::Mat4<double> M1 = matrix_lib::Mat4<double>::R6toSE3(M[0], M[1], M[2], M[3], M[4], M[5]);
		//	matrix_lib::Mat4<double> s = matrix_lib::Mat4<double>::diag(M[6], M[7], M[8], double(1));
		//	M00 = M1*s*M00;
		//	//for (int i = 0; i < 9; i++)
		//	//	std::cout << M[i] << " ";
		//	//std::cout << std::endl;
		//	
		//	//printf("M-update\n");
		//	//print_mat(M1);
		//	//exit(0);
		//	Eigen::VectorXd tmp;
		//	tmp << 	0, 0, 0,  0, 0, 0,  1, 1, 1;
		//	problem->SetState(tmp.data());

		////}
		return ceres::SOLVER_CONTINUE;
	}


	private:
		ceres::Problem *problem;
};


void initialize_M00_with_center(Eigen::Vector3d t, Eigen::Vector3d s) {

	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> M00_eigen(M00.getData());

	M00_eigen = M00_eigen.inverse().eval(); // <-- make as scan2cad

	M00_eigen.block(0, 3, 3, 1) = t;
	M00_eigen.col(0).normalize();
	M00_eigen.col(1).normalize();
	M00_eigen.col(2).normalize();
	M00_eigen.block(0, 0, 3, 3) *= s.asDiagonal();

	M00_eigen = M00_eigen.inverse().eval(); // <- make as cad2scan again (for optimization)

	M00 = matrix_lib::Mat4<double>(M00_eigen.data());

}

void get_tqs_from_x_and_M(Eigen::VectorXd &x, matrix_lib::Mat4<double> &M0, Eigen::Vector3d &t, Eigen::Quaterniond &q, Eigen::Vector3d &s) {
	matrix_lib::Mat4<double> Mdelta = R9toSE3<double>(x);
	matrix_lib::Mat4<double> M_scan2cad = Mdelta*M0;
	Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M_cad2scan(M_scan2cad.getData());
	M_cad2scan = M_cad2scan.inverse().eval();
	decompose_mat4_into_tqs(M_cad2scan, t, q, s);
}

typedef std::tuple<double, double, Eigen::VectorXd, matrix_lib::Mat4<double>> ranktype; // <-- optim. cost, li distance, optimal pose, base pose

void sort_rank_by_cost(std::vector<ranktype>& rank) {
	struct {
		bool operator()(ranktype &a, ranktype &b) const {   
			return std::get<0>(a) < std::get<0>(b);
		}   
	} custom_less;
	std::sort (rank.begin(), rank.end(), custom_less);
}

void filter_rank_by_distance(std::vector<ranktype>& rank) {
	double dmax = std::numeric_limits<double>::max();
	for (int i = 0; i < (int)rank.size(); i++) {
		for (int j = i + 1; j < (int)rank.size(); j++) {

			ranktype &a = rank[i];
			ranktype &b = rank[j];
			if (std::get<0>(a) == dmax || std::get<0>(b) == dmax)
				continue;

			Eigen::Vector3d ta, tb, sa, sb;
			Eigen::Quaterniond qa, qb;
			get_tqs_from_x_and_M(std::get<2>(a), std::get<3>(a), ta, qa, sa);
			get_tqs_from_x_and_M(std::get<2>(b), std::get<3>(b), tb, qb, sb);

			double distance = (ta - tb).norm();
			if (distance <= 0.5)
				std::get<0>(b) = dmax;
		}
	}
}

int main(int argc, char** argv) {
	parse_args(argc, argv);

	read_json_and_load_models(centers, scales, cads, vox_scan);
	assert((int)cads.size() >= 4);
	make_pyramide(cads);

	const int n_params = 9;
	Eigen::VectorXd x0(n_params);
	
	
	const int n_restarts = cads.size();


	std::vector<ranktype> cost_rank(n_restarts); 

	printf("starting optimization\n");
	for (int k = 0; k < n_restarts; k++) {
		//int j = dist(mt);
		Eigen::Vector3d c = centers.col(k).topRows(3);
		Eigen::Vector3d s = scales.col(k);
		initialize_M00_with_center(c, s);

		x0 << 	0, 0, 0,  0, 0, 0,  1, 1, 1; // <-- rot, trans, scale
		
		for (int i = pdf.size() - 1; i >= 0; i--) {
			ceres::Problem problem;
			for (int j = 0; j < (int)cads.size(); j++) {
				CostFunctor *ref = new CostFunctor(i, j);
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 9>(ref);
				problem.AddResidualBlock(cost_function, NULL, x0.data());
			}
			ceres::CostFunction* func_reg_scale = new ceres::AutoDiffCostFunction<CostFunctorScale, 3, 9>(new CostFunctorScale());
			problem.AddResidualBlock(func_reg_scale, NULL, x0.data());
			ceres::Solver::Options options;
			//ParamsUpdateCallback callback(&problem);
			options.minimizer_progress_to_stdout = false;
			//options.update_state_every_iteration = true;
			//options.callbacks.push_back(&callback);
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			
			if (i == 0) {
				double cost_final = 0;
				problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost_final, NULL, NULL, NULL);
				std::cout << "k: " << k << " cost: " << cost_final  << std::endl;
				cost_rank[k] = std::make_tuple(cost_final, -1, x0, M00);
			}
		}
	}
	// -> sort alignments by cost
	sort_rank_by_cost(cost_rank);
	// <-

	// -> Calc Li confidence
	for (int i = 0; i < (int)cost_rank.size(); i++) {
		matrix_lib::Mat4<double> Mdelta = R9toSE3<double>(std::get<2>(cost_rank[i]));
		matrix_lib::Mat4<double> M_scan2cad = Mdelta*std::get<3>(cost_rank[i]);
		Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M_cad2scan(M_scan2cad.getData());
		M_cad2scan = M_cad2scan.inverse().eval();
		std::get<1>(cost_rank[i]) = calc_li_confidence(cads[0], vox_scan, M_cad2scan.cast<float>());
		//std::cout << std::get<1>(cost_rank[i]) << std::endl;
	}
	// <-


	////for (int k = -10; k < 11; k++) {
	//	double cost = 0;
	//	auto x1 = x_best;
	//	M00 = M_best;
	//	//x1(2) += k*0.1;
	//	for (int j = 0; j < (int)cads.size(); j++) {
	//		double r = 0;
	//		CostFunctor f = CostFunctor(0, j);
	//		f(x1.data(), &r);
	//		cost += 0.5*r*r;
	//	}
	//	std::cout << "cost: " << cost << std::endl;
	////}

	// -> write out
	std::ofstream file;
	file.open(inargs.out);
	file << "# alignment-energy, li-confidence, tx, ty, tz, qw, qx, qy, qz, sx, sy, sz" << std::endl;
	for (int i = 0; i < (int)cost_rank.size(); i++) {
		ranktype item = cost_rank[i];
		Eigen::Vector3d t, s;
		Eigen::Quaterniond q;
		get_tqs_from_x_and_M(std::get<2>(item), std::get<3>(item), t, q, s);

		file << std::get<0>(item)/cost_rank.size() << "," << std::get<1>(item)/cost_rank.size() << ","  << t(0) << "," << t(1)<< "," << t(2) << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "," << s(0) << "," << s(1) << "," << s(2) << std::endl;
	}
	file.close();
	// <-

	return 0;
}
