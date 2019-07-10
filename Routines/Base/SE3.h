#pragma once

#include <eigen3/Eigen/Dense>

template <typename real>
Eigen::Matrix<real, 4, 4> R6toSE3(Eigen::Matrix<real, 6, 1> a) {
	Eigen::Matrix<real, 3, 3> S;
	S << 0, -a(2), a(1), a(2), 0, -a(0),  -a(1), a(0), 0;
	real alpha = a.norm();
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
Eigen::Matrix<T,3,3> fix_rot( Eigen::Matrix<T, 3, 3> rot ) {
	rot.col(0).normalize();
	rot.col(1).normalize();
	rot.col(2) = rot.col(0).cross(rot.col(1));
	rot.col(2).normalize();
	rot.col(0) = rot.col(1).cross(rot.col(2));
	rot.col(0).normalize();
	return rot;
}

template <typename T>
Eigen::Matrix<T,3,3> quat_to_mat33_ndr(Eigen::Quaternion<T> q) {
	Eigen::Matrix<T,3,3> m;
	T x  = q.x(), y  = q.y(), z  = q.z(), w  = q.w();
	T xx = x*x,  yy = y*y,  zz = z*z,  ww = w*w;
	T tx = 2*x,  ty = 2*y,  tz = 2*z;
	T xy = ty*x, xz = tz*x, yz = ty*z;
	T wx = tx*w, wy = ty*w, wz = tz*w;
	T t0 = ww-zz;
	T t1 = xx-yy;

	m(0,0) = t0+t1;
	m(1,1) = t0-t1;
	m(2,2) = ww-xx-yy+zz;

	m(1,0) = xy+wz; m(0,1) = xy-wz;
	m(2,0) = xz-wy; m(0,2) = xz+wy;
	m(2,1) = yz+wx; m(1,2) = yz-wx;

	return m;
}


template <typename T>
Eigen::Quaternion<T> mat33_2_quat_day(Eigen::Matrix<T,3,3> m) {
	Eigen::Quaternion<T> q;

	T m00=m(0,0), m01=m(0,1), m02=m(0,2);
	T m10=m(1,0), m11=m(1,1), m12=m(1,2);
	T m20=m(2,0), m21=m(2,1), m22=m(2,2);
	T e0;

	if (m22 >= 0) {
		T a = m00+m11;
		T b = m10-m01;
		T c = 1.f+m22;

		if (a >= 0) { 
			e0 = c + a;
			//q = Eigen::Quaternion<T>(m21-m12, m02-m20, b, e0); // w
			q = Eigen::Quaternion<T>(e0, m21-m12, m02-m20, b);
		}
		else { 
			e0 = c - a;
			//q = Eigen::Quaternion<T>(m02+m20, m21+m12, e0, b); // z
			q = Eigen::Quaternion<T>(b, m02+m20, m21+m12, e0);
		}
	}
	else {
		T a = m00-m11;
		T b = m10+m01;
		T c = 1.f-m22;

		if (a >= 0) {
			e0 = c + a;
			//q = Eigen::Quaternion<T>(e0, b, m02+m20, m21-m12); // x
			q = Eigen::Quaternion<T>(m21-m12, e0, b, m02+m20);
		}
		else {
			e0 = c - a;
			//q = Eigen::Quaternion<T>(b, e0, m21+m12, m02-m20); // y
			q = Eigen::Quaternion<T>(m02-m20, b, e0, m21+m12);
		}
	}

	q.w() *= 0.5/std::sqrt(e0);
	q.x() *= 0.5/std::sqrt(e0);
	q.y() *= 0.5/std::sqrt(e0);
	q.z() *= 0.5/std::sqrt(e0);
	return q;
}


template <typename T>
void decompose_mat4(Eigen::Matrix<T, 4, 4> M, Eigen::Matrix<T, 3, 1> &t, Eigen::Quaternion<T> &q, Eigen::Matrix<T, 3, 1> &s) {

	Eigen::Matrix<T, 3, 3> rot = M.block(0, 0, 3, 3);
	T sx = rot.col(0).norm();
	T sy = rot.col(1).norm();
	T sz = rot.col(2).norm();
	s = Eigen::Matrix<T, 3, 1>(sx, sy, sz);

	q = Eigen::Quaternion<T>(fix_rot(rot));

	t = M.col(3).topRows(3);
}

template <typename T>
Eigen::Matrix<T,4,4> compose_mat4_from_tqs(Eigen::Matrix<T, 3, 1> t, Eigen::Quaternion<T> q, Eigen::Matrix<T, 3, 1> s, Eigen::Matrix<T, 3, 1> center = {0,0,0}) {
	Eigen::Matrix<T,4,4> T0 = Eigen::Matrix<T,4,4>::Identity();
	Eigen::Matrix<T,4,4> R0 = Eigen::Matrix<T,4,4>::Identity();
	Eigen::Matrix<T,4,4> S0 = Eigen::Matrix<T,4,4>::Identity();
	Eigen::Matrix<T,4,4> C0 = Eigen::Matrix<T,4,4>::Identity();

	T0.block(0, 3, 3, 1) = t;
	R0.block(0, 0, 3, 3) = fix_rot(q.toRotationMatrix());
	S0.block(0, 0, 3, 3) = (Eigen::Matrix<T,3,3>)s.asDiagonal();
	C0.block(0, 3, 3, 1) = center;

	return T0*R0*S0*C0;
}
