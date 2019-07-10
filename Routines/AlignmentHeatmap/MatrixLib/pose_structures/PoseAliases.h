#pragma once
#include "RigidPose.h"
#include "SE3.h"
#include "AffinePose.h"
#include "AffineIncrement.h"

namespace matrix_lib {

	/**
	 * Alias names for different pose types.
	 */
	// Rigid pose.
	template <typename T>
	using RigidPose = Pose<T, PoseType::QUATERNION>;

	typedef RigidPose<float> RigidPosef;
	typedef RigidPose<double> RigidPosed;

	// Affine pose.
	template <typename T>
	using AffinePose = Pose<T, PoseType::AFFINE>;

	typedef AffinePose<float> AffinePosef;
	typedef AffinePose<double> AffinePosed;

	// SE3 pose increment.
	template <typename T>
	using SE3 = PoseIncrement<T, PoseType::SE3>;

	typedef SE3<float> SE3f;
	typedef SE3<double> SE3d;

	// Affine pose increment.
	template <typename T>
	using AffineIncrement = PoseIncrement<T, PoseType::AFFINE>;

	typedef AffineIncrement<float> AffineIncrementf;
	typedef AffineIncrement<double> AffineIncrementd;

} // namespace matrix_lib
