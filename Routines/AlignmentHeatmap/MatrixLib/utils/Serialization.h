#pragma once

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>

#include "MatrixLib/MatrixLibInclude.h"

namespace boost {
	namespace serialization {

		// Vector classes
		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::Vec2<FloatType>& v, const unsigned int version) {
			ar & v.x();
			ar & v.y();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::Vec3<FloatType>& v, const unsigned int version) {
			ar & v.x();
			ar & v.y();
			ar & v.z();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::Vec4<FloatType>& v, const unsigned int version) {
			ar & v.x();
			ar & v.y();
			ar & v.z();
			ar & v.w();
		}

		template<class Archive, typename FloatType, unsigned s>
		void serialize(Archive& ar, matrix_lib::VecX<FloatType, s>& v, const unsigned int version) {
			for (int i = 0; i < s; ++i) {
				ar & v[i];
			}
		}


		// Matrix classes.
		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::Mat2<FloatType>& m, const unsigned int version) {
			for (int i = 0; i < 4; ++i) {
				ar & m[i];
			}
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::Mat3<FloatType>& m, const unsigned int version) {
			for (int i = 0; i < 9; ++i) {
				ar & m[i];
			}
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::Mat4<FloatType>& m, const unsigned int version) {
			for (int i = 0; i < 16; ++i) {
				ar & m[i];
			}
		}


		// Pose classes.
		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::SO3<FloatType>& p, const unsigned int version) {
			ar & p.getOmega();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::UnitQuaternion<FloatType>& p, const unsigned int version) {
			ar & p.real();
			ar & p.imag();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::RigidPose<FloatType>& p, const unsigned int version) {
			ar & p.getQuaternion();
			ar & p.getTranslation();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::SE3<FloatType>& p, const unsigned int version) {
			ar & p.getAxisAngle();
			ar & p.getTranslation();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::AffinePose<FloatType>& p, const unsigned int version) {
			ar & p.getAffineMatrix();
			ar & p.getTranslation();
		}

		template<class Archive, typename FloatType>
		void serialize(Archive& ar, matrix_lib::AffineIncrement<FloatType>& p, const unsigned int version) {
			ar & p.getAffineMatrix();
			ar & p.getTranslation();
		}

	} // namespace serialization
} // namespace boost