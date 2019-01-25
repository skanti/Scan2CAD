#pragma once

#include <eigen3/Eigen/Dense>

template <typename real>
Eigen::Matrix<real, 4, 4> R6toSE3(Eigen::Matrix<real, 6, 1> a) {
	Eigen::Matrix<real, 3, 3> S;
	S << 0, -a(2), a(1), a(2), 0, -a(0),  -a(1), a(0), 0;
	float alpha = a.norm();
	Eigen::Matrix<real, 3, 3> V = Eigen::Matrix<real, 3, 3>::Identity();
	Eigen::Matrix<real, 3, 3> R = Eigen::Matrix<real, 3, 3>::Identity();
	if (alpha != 0) {
		float ia2 = 1.0/(alpha*alpha);
		float s = std::sin(alpha);
		float c = std::cos(alpha);
		R = Eigen::Matrix<real, 3, 3>::Identity() + s/alpha*S + (1.0 - c)*ia2*S*S;
		V = Eigen::Matrix<real, 3, 3>::Identity() + (1.0 - c)*ia2*S + (alpha - s)*ia2/alpha*S*S;
	}
	//Eigen::Matrix<real, 4, 1> t = Eigen::Matrix<real, 4, 1>(V*a1, 1.0);
	Eigen::Matrix<real, 4, 1> t;
	t.topRows(3) = V*a.bottomRows(3);
	t(3) = 1;

	Eigen::Matrix<real, 4, 4> M = Eigen::Matrix<real, 4, 4>::Identity();
	M.block(0, 0, 3, 3) = R;
	M.col(3) = t;
	return M;
}

template <typename T>
void decompose_mat4(Eigen::Matrix<T, 4, 4> M, Eigen::Matrix<T, 3, 1> &t, Eigen::Quaternion<T> &q, Eigen::Matrix<T, 3, 1> &s) {

	Eigen::Matrix<T, 3, 3> rot = M.block(0, 0, 3, 3);
	T sx = rot.col(0).norm();
	T sy = rot.col(1).norm();
	T sz = rot.col(2).norm();
	s = Eigen::Matrix<T, 3, 1>(sx, sy, sz);

	rot.col(0) /= sx;
	rot.col(1) /= sy;
	rot.col(2) /= sz;
	q = Eigen::Quaternion<T>(rot);
	q.normalize();

	t = M.col(3).topRows(3);
}
