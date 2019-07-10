#pragma once
#include "MatrixLib/utils/Promotion.h"

namespace matrix_lib {

	/**
	 * The type of pose that is used for modelling deformations.
	 */
	struct PoseType {
		enum {
			SE3 = 0,
			AFFINE = 1,
			QUATERNION = 2
		};
	};


	/**
	 * Abstract pose class.
	 * These methods need to be implemented by all specializations.
	 */
	template<typename T, int PoseType>
	class Pose {
	public:
		/**
		 * Applies the pose on the given point.
		 */
		template<typename U>  Vec3<Promote<T, U>> apply(const Vec3<U>& point) const;
		template<typename U>  Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const;

		/**
		 * Rotates the given point.
		 */
		template<typename U>  Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const;

		/**
		 * Translates the given point.
		 */
		template<typename U>  Vec3<Promote<T, U>> translate(const Vec3<U>& point) const;

		/**
		 * Computes a 4x4 pose matrix. Needs to be generated (therefore slower).
		 */
		 Mat4<T> matrix() const;

		/**
		 * Updates the current pose to match the given pose.
		 */
		 void update(const Pose<T, PoseType>& pose);

		/**
		 * Static constructors. 
		 * The input parameters are given in matrix notation.
		 */
		 static Pose<T, PoseType> identity();
		 static Pose<T, PoseType> translation(const Vec3<T>& translation);
		 static Pose<T, PoseType> rotation(const Mat3<T>& rotation);
		 static Pose<T, PoseType> pose(const Mat4<T>& pose);
		 static Pose<T, PoseType> pose(const Mat3<T>& rotation, const Vec3<T>& translation);

		/**
		 * Interpolates the given poses with given interpolation weights.
		 * The interpolation weights need to sum up to 1.
		 */
		 static Pose<T, PoseType> interpolate(const vector<Pose<T, PoseType>>& poses, const vector<T>& weights);
	};

} // namespace matrix_lib
