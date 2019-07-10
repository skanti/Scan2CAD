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
#include <ceres/ceres.h>
#include <json/json.h>

#include "args.hxx"

#include "Grid.h"

#include <iostream>

const int n_params = 9;

Vox vox_scan;
std::vector<Vox> cads;
std::vector<std::vector<Grid<float>>> pdf, gradientx, gradienty, gradientz;
Eigen::Matrix<double, -1, -1> centers;
Eigen::Matrix<double, -1, -1> scales;
Eigen::Matrix4d M00; // <-- as scan2cad

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

	vox_scan = load_vox (filename_vox_scan);
	Eigen::Quaterniond q = Eigen::Quaterniond(root["rot"][0].asDouble(), root["rot"][1].asDouble(), root["rot"][2].asDouble(), root["rot"][3].asDouble()).normalized();

	M00 = Eigen::Matrix4d::Identity();
	M00.block(0, 0, 3, 3) = q.toRotationMatrix();

	for (Json::Value::ArrayIndex i = 0; i != root["pairs"].size(); i++) {
		Vox heatmap;
		heatmap = load_vox(root["pairs"][i]["filename_heatmap"].asString());
		heatmaps.push_back(heatmap);

		Eigen::Vector4d p_center = Eigen::Vector4d(root["pairs"][i]["p_scan"][0].asDouble(), root["pairs"][i]["p_scan"][1].asDouble(), root["pairs"][i]["p_scan"][2].asDouble(), 1);
		centers.conservativeResize(4, centers.cols() + 1);
		centers.col(centers.cols() - 1) = p_center;
		
		Eigen::Vector3d scale = Eigen::Vector3d(root["pairs"][i]["scale"][0].asDouble(), root["pairs"][i]["scale"][1].asDouble(), root["pairs"][i]["scale"][2].asDouble());
		scales.conservativeResize(3, scales.cols() + 1);
		scales.col(scales.cols() - 1) = scale;
		
	}
}


// -> matrix helper stuff
template <typename T>
Eigen::Matrix<T, 4, 4> R6toSE3(Eigen::Matrix<T, 3, 1> r, Eigen::Matrix<T, 3, 1> t) {
	Eigen::Matrix<T, 3, 3> S;
	S << T(0), -r(2), r(1), r(2), T(0), -r(0), -r(1), r(0), T(0);
	T theta2 = r(0)*r(0) + r(1)*r(1) + r(2)*r(2);
	T theta = ceres::sqrt(theta2);
	T s = ceres::sin(theta);
	T c = ceres::cos(theta);
	const T t0 = theta == T(0) ? T(1.0) : s/theta;
	const T t1 = theta == T(0) ? T(0.5) : (T(1.0) - c)/theta2;
	const T t2 = theta == T(0) ? T(1.0/6.0) : (theta - s)/(theta2*theta);

	Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity() + S*t0 + S*S*t1;
	Eigen::Matrix<T, 3, 3> V = Eigen::Matrix<T, 3, 3>::Identity() + S*t1 + S*S*t2;

	Eigen::Matrix<T, 4, 1> v;
	v.topRows(3) = V*t;
	v(3) = T(1.0);

	Eigen::Matrix<T, 4, 4> M = Eigen::Matrix<T, 4, 4>::Identity();
	M.block(0, 0, 3, 3) = R;
	M.col(3) = v;
	return M;
}

template <typename T>
Eigen::Matrix<T, 4, 4> R9toSE3(Eigen::Matrix<T, -1, 1> &x) {
	Eigen::Matrix<T, 3, 1> r(x(0), x(1), x(2));
	Eigen::Matrix<T, 3, 1> t(x(3), x(4), x(5));
	Eigen::Matrix<T, 3, 1> s(x(6), x(7), x(8));

	Eigen::Matrix<T, 4, 4> M = R6toSE3(r, t);
	Eigen::Matrix<T, 4, 4> S = Eigen::Matrix<T, 4, 1>(T(x(6)), T(x(7)), T(x(8)), T(1)).asDiagonal();
	return M*S;
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

			const T fac = T(16.0);
			residual[0] = fac*(T(1) - params[6]);
			residual[1] = fac*(T(1) - params[7]);
			residual[2] = fac*(T(1) - params[8]);

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

			Eigen::Matrix<T, -1, 1> params1(n_params);
			params1 << params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8];

			Eigen::Matrix<T, 4, 4> M = R9toSE3<T>(params1)*M00.cast<T>();
			
			Eigen::Matrix<T, 4, 1> c(T(centers.col(i_cad)(0)), T(centers.col(i_cad)(1)), T(centers.col(i_cad)(2)), T(1));

			Eigen::Matrix<T, 4, 1> idx0 = pdf[i_level][i_cad].world2grid.cast<T>()*M*c;
			const T idx[3] = {idx0(0), idx0(1), idx0(2)};

			T f = T(0);
			(*func_grid)(idx, &f);
			residual[0] = ceres::abs(T(1) - f);

			return true;
		}

	int i_level, i_cad;
	std::unique_ptr<ceres::CostFunctionToFunctor<1, 3> > func_grid;
};


void initialize_M00_with_center_and_scale(Eigen::Vector3d t, Eigen::Vector3d s) {
	Eigen::Matrix4d tmp = M00.inverse().eval(); // <-- make as scan2cad

	tmp.block(0, 3, 3, 1) = t;
	tmp.col(0).normalize();
	tmp.col(1).normalize();
	tmp.col(2).normalize();
	tmp.block(0, 0, 3, 3) *= s.asDiagonal();

	M00 = tmp.inverse().eval(); // <- make as cad2scan again (for optimization)
}

void get_tqs_from_x_and_M(Eigen::VectorXd &x, Eigen::Matrix4d &M0, Eigen::Vector3d &t, Eigen::Quaterniond &q, Eigen::Vector3d &s) {
	Eigen::Matrix4d Mdelta = R9toSE3<double>(x);
	Eigen::Matrix4d M_scan2cad = Mdelta*M0;
	Eigen::Matrix4d M_cad2scan = M_scan2cad.inverse().eval();
	decompose_mat4_into_tqs(M_cad2scan, t, q, s);
}

typedef std::tuple<double, double, Eigen::VectorXd, Eigen::Matrix4d> ranktype; // <-- optim. cost, li distance, pose delta, base pose

void sort_rank_by_cost(std::vector<ranktype>& rank) {
	struct {
		bool operator()(ranktype &a, ranktype &b) const {   
			return std::get<0>(a) < std::get<0>(b);
		}   
	} custom_less;
	std::sort (rank.begin(), rank.end(), custom_less);
}

int main(int argc, char** argv) {
	parse_args(argc, argv);

	read_json_and_load_models(centers, scales, cads, vox_scan);
	assert((int)cads.size() >= 4);
	make_pyramide(cads);

	Eigen::VectorXd x0(n_params);
	

	const int n_restarts = cads.size();

	std::vector<ranktype> cost_rank(n_restarts); 

	printf("starting optimization\n");
	for (int k = 0; k < n_restarts; k++) {
		Eigen::Vector3d c = centers.col(k).topRows(3);
		Eigen::Vector3d s = scales.col(k);
		initialize_M00_with_center_and_scale(c, s);

		x0 << 	0, 0, 0,  0, 0, 0,  1, 1, 1; // <-- rot, trans, scale
		
		for (int i = pdf.size() - 1; i >= 0; i--) {
			ceres::Problem problem;
			for (int j = 0; j < (int)cads.size(); j++) {
				CostFunctor *ref = new CostFunctor(i, j);
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, n_params>(ref);
				problem.AddResidualBlock(cost_function, NULL, x0.data());
			}
			ceres::CostFunction* func_reg_scale = new ceres::AutoDiffCostFunction<CostFunctorScale, 3, n_params>(new CostFunctorScale());
			//problem.AddResidualBlock(func_reg_scale, NULL, x0.data());
			ceres::Solver::Options options;
			options.minimizer_progress_to_stdout = false;
			ceres::Solver::Summary summary;
			//ceres::Solve(options, &problem, &summary);
			
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
		Eigen::Matrix4d Mdelta = R9toSE3<double>(std::get<2>(cost_rank[i]));
		Eigen::Matrix4d M_scan2cad = Mdelta*std::get<3>(cost_rank[i]);
		Eigen::Matrix4d M_cad2scan = M_scan2cad.inverse().eval();
		std::get<1>(cost_rank[i]) = calc_li_confidence(cads[0], vox_scan, M_cad2scan.cast<float>());
		//std::cout << std::get<1>(cost_rank[i]) << std::endl;
	}
	// <-

	// -> write out
	std::ofstream file;
	file.open(inargs.out);
	file << "# alignment-cost, li-confidence, tx, ty, tz, qw, qx, qy, qz, sx, sy, sz" << std::endl;
	for (int i = 0; i < (int)cost_rank.size(); i++) {
		ranktype item = cost_rank[i];
		Eigen::Vector3d t, s;
		Eigen::Quaterniond q;
		get_tqs_from_x_and_M(std::get<2>(item), std::get<3>(item), t, q, s);

		file << std::get<0>(item)/cost_rank.size() << "," << std::get<1>(item)/cost_rank.size() << "," << t(0) << "," << t(1)<< "," << t(2) << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "," << s(0) << "," << s(1) << "," << s(2) << std::endl;
	}
	file.close();
	// <-

	return 0;
}
