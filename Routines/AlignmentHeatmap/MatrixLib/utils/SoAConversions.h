#pragma once
#include "MatrixLib/matrix_structures/MatrixStructuresInclude.h"
#include "MatrixLib/pose_structures/PoseStructuresInclude.h"

namespace matrix_lib {
	namespace soa {
	
		/**
		 * Methods to load the data structures from SoA and to store data structures
		 * in SoA.
		 */
		 inline Mat3f load(const iAMat3f& soa, unsigned i = 0) {
			return Mat3f{ soa.xrow()[i], soa.yrow()[i], soa.zrow()[i] };
		}
		 inline void store(const Mat3f& m, iAMat3f& soa, unsigned i = 0) {
			soa.xrow()[i] = make_float4(m(0, 0), m(0, 1), m(0, 2), 1.f);
			soa.yrow()[i] = make_float4(m(1, 0), m(1, 1), m(1, 2), 1.f);
			soa.zrow()[i] = make_float4(m(2, 0), m(2, 1), m(2, 2), 1.f);
		}

		 inline Mat4f load(const iAMat4f& soa, unsigned i = 0) {
			return Mat4f{ soa.xrow()[i], soa.yrow()[i], soa.zrow()[i], soa.wrow()[i] };
		}
		 inline void store(const Mat4f& m, iAMat4f& soa, unsigned i = 0) {
			soa.xrow()[i] = make_float4(m(0, 0), m(0, 1), m(0, 2), m(0, 3));
			soa.yrow()[i] = make_float4(m(1, 0), m(1, 1), m(1, 2), m(1, 3));
			soa.zrow()[i] = make_float4(m(2, 0), m(2, 1), m(2, 2), m(2, 3));
			soa.wrow()[i] = make_float4(m(3, 0), m(3, 1), m(3, 2), m(3, 3));
		}

		 inline SE3f load(const iASE3f& soa, unsigned i = 0) {
			return SE3f{ soa.rotation()[i], soa.translation()[i] };
		}
		 inline void store(const SE3f& p, iASE3f& soa, unsigned i = 0) {
			soa.rotation()[i] = make_float4(p.getAxisAngle().getOmega().x(), p.getAxisAngle().getOmega().y(), p.getAxisAngle().getOmega().z(), 1.f);
			soa.translation()[i] = make_float4(p.getTranslation().x(), p.getTranslation().y(), p.getTranslation().z(), 1.f);
		}

		 inline RigidPosef load(const iAQuaternionPosef& soa, unsigned i = 0) {
			return RigidPosef{ soa.rotation()[i], soa.translation()[i] };
		}
		 inline void store(const RigidPosef& p, iAQuaternionPosef& soa, unsigned i = 0) {
			soa.rotation()[i] = make_float4(p.getQuaternion().real(), p.getQuaternion().imag().x(), p.getQuaternion().imag().y(), p.getQuaternion().imag().z());
			soa.translation()[i] = make_float4(p.getTranslation().x(), p.getTranslation().y(), p.getTranslation().z(), 1.f);
		}

		 inline AffinePosef load(const iAAffinePosef& soa, unsigned i = 0) {
			return AffinePosef{ soa.rotation().xrow()[i], soa.rotation().yrow()[i], soa.rotation().zrow()[i], soa.translation()[i] };
		}
		 inline void store(const AffinePosef& p, iAAffinePosef& soa, unsigned i = 0) {
			store(p.getAffineMatrix(), soa.rotation(), i);
			soa.translation()[i] = make_float4(p.getTranslation().x(), p.getTranslation().y(), p.getTranslation().z(), 1.f);
		}

	} // namespace soa
} // namespace matrix_lib
