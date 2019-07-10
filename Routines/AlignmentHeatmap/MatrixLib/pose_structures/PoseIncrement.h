#pragma once
#include "MatrixLib/utils/Promotion.h"

namespace matrix_lib {

	/**
	 * Abstract pose increment class.
	 * These methods need to be implemented by all specializations.
	 */
	template<typename T, int PoseType>
	class PoseIncrement {
	public:
		/**
		 * Returns the pointer to the raw data.
		 */
		T* getData();

		/**
		 * Applies the pose on the given point.
		 */
		template<typename U> Vec3<Promote<T, U>> apply(const Vec3<U>& point) const;
		template<typename U> Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const;

		/**
		 * Rotates the given point.
		 */
		template<typename U> Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const;

		/**
		* Translates the given point.
		*/
		template<typename U> Vec3<Promote<T, U>> translate(const Vec3<U>& point) const;

		/**
		 * Computes a 4x4 pose matrix. Needs to be generated (therefore slower).
		 */
		Mat4<T> matrix() const;

		/**
		 * Resets the pose increment to identity rotation and zero translation.
		 */
		void reset();

		/**
		 * Static constructors.
		 * The input parameters are given in matrix notation.
		 */
		static PoseIncrement<T, PoseType> identity();
		static PoseIncrement<T, PoseType> translation(const Vec3<T>& translation);
		static PoseIncrement<T, PoseType> rotation(const Mat3<T>& rotation);
		static PoseIncrement<T, PoseType> pose(const Mat4<T>& pose);
		static PoseIncrement<T, PoseType> pose(const Mat3<T>& rotation, const Vec3<T>& translation);
	};

}
