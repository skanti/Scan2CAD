#pragma once

namespace matrix_lib {

	namespace pose_proc {

		/**
		 * Returns a skew-symetric matrix of vector v.
		 */
		template<typename T>
		 Mat3<T> hat(const Vec3<T>& v) {
			Mat3<T> matrix = {
				T(0),	-v.z(),	v.y(),
				v.z(),	T(0),	-v.x(),
				-v.y(),	v.x(),	T(0)
			};
			return matrix;
		}

	} // namespace pose_proc

} // namespace matrix_lib
