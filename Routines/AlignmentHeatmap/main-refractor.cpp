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

#include "SE3.h"
#include "Grid.h"

#include <iostream>

template<typename T>
struct HeatmapGrid {
	Grid<T> value, gradx, grady, gradz;

	void calc_gradient() {
		Grid<T>::calc_gradient(value, gradx, grady, gradz);
	}
};

struct Correspondence {
	Vox vox;
	std::vector<HeatmapGrid<float>> heatmap;
	Eigen::Vector3d p_scan;
	Eigen::Vector3d scale_network;
	Eigen::Matrix4d M00_cad2scan;
};


struct InputArgs {
	std::string json;
} inargs;

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
Eigen::Matrix<T, 4, 4> R9toSE3(Eigen::Matrix<T, 3, 1> t, Eigen::Matrix<T, 3, 1> r, Eigen::Matrix<T, 3, 1> s) {
	Eigen::Matrix<T, 4, 4> M = R6toSE3(r, t);
	Eigen::Matrix<T, 4, 1> S;
	S << s(0), s(1), s(2), T(1);
	M = S.asDiagonal()*M;
	return M;
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

	t = V.inverse()*M.block(0, 3, 3, 1);

}
// <-

void ComputeGridValueAndJacobian(const double x, const double y, const double z, const auto *grid_data, double *value, double *jacobian) {
	value[0] = grid_data->value(x, y, z);
	if (jacobian) {
		jacobian[0] = grid_data->gradx(x, y, z);
		jacobian[1] = grid_data->grady(x, y, z);
		jacobian[2] = grid_data->gradz(x, y, z);
	} 
}

struct GridFunctor : public ceres::SizedCostFunction<1, 3> {
	GridFunctor(auto *grid_data_) {
		grid_data = grid_data_;
	}

	virtual bool Evaluate(double const* const* params, double* residuals, double **jacobians) const {

		if (!jacobians)
			ComputeGridValueAndJacobian(params[0][0], params[0][1], params[0][2], grid_data, residuals, NULL);
		else
			ComputeGridValueAndJacobian(params[0][0], params[0][1], params[0][2], grid_data, residuals, jacobians[0]);

		return true;
	}

	HeatmapGrid<float> *grid_data;
};

struct CostFunctorScale {
	CostFunctorScale(double reg_) {
		reg = reg_;
	};

	template <typename T>
		bool operator()(const T* params, T* residual) const {

			const T fac = T(reg);
			residual[0] = fac*(T(1.0) - params[6]);
			residual[1] = fac*(T(1.0) - params[7]);
			residual[2] = fac*(T(1.0) - params[8]);

			return true;
		}
	private:
	double reg;
};

struct CostFunctor {
	CostFunctor(Correspondence *corr_) {
		corr = corr_;
		//M = M_;

		func_grid.reset(new ceres::CostFunctionToFunctor<1, 3>(new GridFunctor(&(corr->heatmap))));
	}

	template <typename T>
		bool operator()(const T* params, T* residual) const {


			Eigen::Matrix<T, 3, 1> t(params[0], params[1], params[2]);
			Eigen::Matrix<T, 3, 1> r(params[3], params[4], params[5]);
			Eigen::Matrix<T, 3, 1> s(params[6], params[7], params[8]);
			Eigen::Matrix<T, 4, 4> Mdelta = R9toSE3(t, r, s);

			Eigen::Matrix<T, 4, 1> c(T(corr->p_scan(0)), T(corr->p_scan(1)), T(corr->p_scan(2)), T(1.0));
			Eigen::Matrix<T, 4, 1> idx = corr->vox.grid2world.cast<T>().inverse()*Mdelta*M.cast<T>()*c;

			T f = T(0);
			(*func_grid)(idx, &f);
			residual[0] = ceres::abs(T(1) - f);

			return true;
		}

	Correspondence *corr;
	Eigen::Matrix4d M;
	std::unique_ptr<ceres::CostFunctionToFunctor<1, 3> > func_grid;
};

class World {
	public:
		void read_json_and_load_stuff() {
			std::ifstream file(inargs.json);
			assert(file.is_open() && "Cannot open json file.");
			Json::Value root;
			file >> root;
			
			catid_cad = root["catid_cad"].asString();
			id_cad = root["id_cad"].asString();
			
			std::string filename_vox_scan = root["vox_scan"].asString();
			vox_scan = load_vox (filename_vox_scan);

			Eigen::Vector3d t = Eigen::Vector3d(root["trs"]["trans"][0].asDouble(), root["trs"]["trans"][1].asDouble(), root["trs"]["trans"][2].asDouble());
			Eigen::Quaterniond q = Eigen::Quaterniond(root["trs"]["rot"][0].asDouble(), root["trs"]["rot"][1].asDouble(), root["trs"]["rot"][2].asDouble(), root["trs"]["rot"][3].asDouble());
			Eigen::Vector3d s = Eigen::Vector3d(root["trs"]["scale"][0].asDouble(), root["trs"]["scale"][1].asDouble(), root["trs"]["scale"][2].asDouble());

			for (int i = 0; i < (int)root["pairs"].size(); i++) {
				Json::Value correspondence = root["pairs"][i];
				
				Correspondence corr;

				corr.vox = load_vox(correspondence["heatmap"].asString());
	
				make_pyramide(corr);

				corr.p_scan = Eigen::Vector3d(correspondence["p_scan"][0].asDouble(), correspondence["p_scan"][1].asDouble(), correspondence["p_scan"][2].asDouble());
				
				corr.scale_network = Eigen::Vector3d(correspondence["scale"][0].asDouble(), correspondence["scale"][1].asDouble(), correspondence["scale"][2].asDouble());

				correspondences.push_back(corr);
				
			}

		}

		void make_pyramide(Correspondence &corr) {
			corr.heatmap.resize(n_level);

			corr.heatmap[0].value = Grid<float>(corr.vox.dims, corr.vox.grid2world.inverse().eval().cast<double>(), corr.vox.pdf);
			corr.heatmap[0].value.pad();
			corr.heatmap[0].value.blur(2);
			corr.heatmap[0].calc_gradient();

			for (int i = 1; i < n_level; i++) {
				corr.heatmap[i] = HeatmapGrid<float>();
				corr.heatmap[i].value.downsample2x(corr.heatmap[i - 1].value);
				corr.heatmap[i].value.blur(1);
				corr.heatmap[i].calc_gradient();
			}
			//auto sdftmp0 = Grid<float>(cads[0].dims, cads[0].grid2world.inverse().eval().cast<double>(), cads[0].sdf);
			//sdftmp0.pad(5);
			//auto sdftmp1 = Grid<float>();
			//sdftmp1.downsample2x(sdftmp0);
			//auto sdftmp2 = Grid<float>();
			//sdftmp2.downsample2x(sdftmp1);
			//save_vox<float, 1, float, 1>("test2.vox2", pdf[2][0].dims, 4, pdf[2][0].world2grid.inverse().eval().cast<float>(), sdftmp2.data, pdf[2][0].data);
			//save_vox<float, 1, float, 1>("test1.vox2", pdf[1][0].dims, 2, pdf[1][0].world2grid.inverse().eval().cast<float>(), sdftmp1.data, pdf[1][0].data);
			//save_vox<float, 1, float, 1>("test0.vox2", pdf[0][0].dims, 1, pdf[0][0].world2grid.inverse().eval().cast<float>(), sdftmp0.data, pdf[0][0].data);
		}

		void start() {
			read_json_and_load_stuff();
			optimize();
			save_results();

		}

		void save_results() {
			// -> write out
			std::ofstream file;
			file.open(filename_out);
			file << "# alignment: catid-cad, id-cad, fcost, tx, ty, tz, qw, qx, qy, qz, sx, sy, sz" << std::endl;
			for (int i = 0; i < (int)correspondences.size(); i++) {
				Eigen::Vector3d t, s;
				Eigen::Quaterniond q;
				Eigen::Matrix4d Mdelta = R9toSE3<double>(params[i].segment(0, 3), params[i].segment(3, 3), params[i].segment(6, 3));
				Eigen::Matrix4d Mcad2scan = Mdelta; //(Mdelta*cads[i].M00_cad2scan.inverse()).inverse().eval();
				decompose_mat4(Mcad2scan, t, q, s);

				file << catid_cad << "," << id_cad << "," << 0.0 << "," << t(0) << "," << t(1)<< "," << t(2) << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "," << s(0) << "," << s(1) << "," << s(2) << std::endl;
			}
			file.close();
			// <-
		}

		void optimize() {
			int n_params = 9; // <-- 9DoF

			std::cout << "starting optimization" << std::endl;
			Eigen::VectorXd x(9);
			x << 	0, 0, 0,  0, 0, 0,  1, 1, 1; // <-- rot, trans, scale

			const int n_restarts = correspondences.size();

			//std::vector<ranktype> cost_rank(n_restarts); 

			printf("starting optimization\n");
			for (int k = 0; k < n_restarts; k++) {
				//int j = dist(mt);
				Eigen::Vector3d c = centers.col(k).topRows(3);
				Eigen::Vector3d s = scales.col(k);
				initialize_M00_with_center(c, s);

				x0 << 	0, 0, 0,  0, 0, 0,  1; // <-- rot, trans, scale

				for (int i = pdf.size() - 1; i >= 0; i--) {
					ceres::Problem problem;
					for (int j = 0; j < (int)cads.size(); j++) {
						CostFunctor *ref = new CostFunctor(i, j);
						ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, n_params>(ref);
						problem.AddResidualBlock(cost_function, NULL, x0.data());
					}
					ceres::CostFunction* func_reg_scale = new ceres::AutoDiffCostFunction<CostFunctorScale, 1, n_params>(new CostFunctorScale());
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



			void invert_tqs(Eigen::Vector3d &t, Eigen::Quaterniond &q, Eigen::Vector3d &s) {
			t = (-1.0)*((q.inverse()*t).cwiseQuotient(s));
			s(0) = 1.0/s(0);
			s(1) = 1.0/s(1);
			s(2) = 1.0/s(2);
			q = q.inverse().normalized();
		}

	private:
		int n_level = 3; // <-- number of levels of the scale pyramide
		std::string catid_cad, id_cad;
		std::vector<Eigen::VectorXd> params;  // <- to be optimized over
		Vox vox_scan;
		std::vector<Correspondence> correspondences;

		std::string filename_out;
};



void parse_args(int argc, char** argv) {
	args::ArgumentParser parser("This is a test program.", "This goes after the options.");
	args::Group allgroup(parser, "", args::Group::Validators::All);

	args::ValueFlag<std::string> json(allgroup, "bunny.json", "predictions", {"json"});

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
};

int main(int argc, char** argv) {
	parse_args(argc, argv);

	World world;
	world.start();

	return 0;
}
