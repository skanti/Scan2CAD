#pragma once
#include "PoseAliases.h"
#include "PoseOperations.h"

namespace matrix_lib {

	/**
	 * Pose-to-pose multiplication.
	 */
	template<typename T>
	 RigidPose<T> operator*(RigidPose<T> lhs, const RigidPose<T>& rhs) {
		lhs *= rhs;
		return lhs;
	}

	template<typename T>
	 AffinePose<T> operator*(AffinePose<T> lhs, const AffinePose<T>& rhs) {
		lhs *= rhs;
		return lhs;
	}


	/**
	 * Increment-to-pose multiplication (left-increment notation).
	 */
	template<typename T, typename U>
	 RigidPose<T> operator*(const SE3<U>& lhs, const RigidPose<T>& rhs) {
		SE3<T> poseIncrement(lhs);
		UnitQuaternion<T> quaternionIncrement = poseIncrement.getAxisAngle().toQuaternion();
		const Vec3<T>& translationIncrement = poseIncrement.getTranslation();

		RigidPose<T> pose{ quaternionIncrement * rhs.getQuaternion(), quaternionIncrement * rhs.getTranslation() + translationIncrement };
		return pose;
	}

	template<typename T, typename U>
	 AffinePose<T> operator*(const AffineIncrement<U>& lhs, const AffinePose<T>& rhs) {
		AffineIncrement<T> poseIncrement{ lhs };
		const Mat3<T>& affineMatrixIncrement = poseIncrement.getAffineMatrix();
		const Vec3<T>& translationIncrement = poseIncrement.getTranslation();

		AffinePose<T> pose{ affineMatrixIncrement * rhs.getAffineMatrix(), affineMatrixIncrement * rhs.getTranslation() + translationIncrement };
		return pose;
	}

} // namespace matrix_lib
