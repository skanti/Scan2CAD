#pragma once
#include "PoseProcessing.h"

namespace matrix_lib {

	template <typename T>
	class UnitQuaternion;

	/**
	 * Rotation, represented in the angle-axis (Lie algebra) notation.
	 */
	template<typename T>
	class SO3 {
	public:
		/**
		 * Default constructor.
		 */
		 SO3() = default;

		/**
		 * Constructor from axis-angle vector.
		 */
		 SO3(const Vec3<T>& omega) : m_omega{ omega } { }
		 SO3(const T x, const T y, const T z) : SO3(Vec3<T>{ x, y, z }) { }

		/**
		 * Constructor from 3x3 rotation matrix (needs conversion).
		 */
		 SO3(const Mat3<T>& matrix) {
			(*this) = UnitQuaternion<T>(matrix).toAxisAngle();
		}

		/**
		 * Constructor from parameter array.
		 * Important: The size of data needs to be at least 3 * sizeof(T).
		 */
		 SO3(const T* data) {
			for (int i = 0; i < 3; ++i) m_omega[i] = data[i];
		}

		/**
		 * Initialize with float4.
		 */
		 SO3(const float4& v) : m_omega{ v }  { }

		/**
		 * Copy data from other template type.
		 */
		template<typename U>
		friend class SO3;

		template <class U>
		 SO3(const SO3<U>& other) : m_omega{ other.m_omega } { }

		/**
		 * Applies an axis-angle rotation to a given point.
		 */
		 Vec3<T> apply(const Vec3<T>& point) const {
			const T theta2 = m_omega | m_omega;
			if (theta2 > T(FLT_EPSILON)) {
				// Away from zero, use the Rodrigues' formula
				//
				//   result = point costheta +
				//            (w x point) * sintheta +
				//            w (w . point) (1 - costheta)
				//
				// We want to be careful to only evaluate the square root if the
				// norm of the angle_axis vector is greater than zero. Otherwise
				// we get a division by zero.
				
				const T theta = sqrt(theta2);
				const T costheta = cos(theta);
				const T sintheta = sin(theta);
				const T theta_inverse = T(1.0) / theta;

				const Vec3<T> w = m_omega  * theta_inverse;
				const Vec3<T> wCrossPoint = w ^ point;
				const T tmp = (w | point) * (T(1.0) - costheta);
				
				return Vec3<T>(
					point.x() * costheta + wCrossPoint.x() * sintheta + w.x() * tmp,
					point.y() * costheta + wCrossPoint.y() * sintheta + w.y() * tmp,
					point.z() * costheta + wCrossPoint.z() * sintheta + w.z() * tmp
				);
			}
			else {
				// Near zero, the first order Taylor approximation of the rotation
				// matrix R corresponding to a vector w and angle w is
				//
				//   R = I + hat(w) * sin(theta)
				//
				// But sintheta ~ theta and theta * w = angle_axis, which gives us
				//
				//  R = I + hat(w)
				//
				// and actually performing multiplication with the point pt, gives us
				// R * pt = pt + w x pt.
				//
				// Switching to the Taylor expansion near zero provides meaningful
				// derivatives when evaluated using Jets.

				const Vec3<T> wCrossPoint = m_omega ^ point;

				return Vec3<T>{
					point.x() + wCrossPoint.x(),
					point.y() + wCrossPoint.y(),
					point.z() + wCrossPoint.z()
				};
			}
		}

		 Vec3<T> operator*(const Vec3<T>& point) const {
			return apply(point);
		}

		/**
		 * Sets all three components to the given value.
		 */
		 const SO3<T>& operator=(T value) {
			m_omega = value;
			return *this;
		}

		/**
		 * Converts the rotation to a 3x3 rotation matrix.
		 */
		 Mat3<T> matrix() const {
			// We use the Rodrigues' formula for exponential map.
			// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

			const T theta2 = m_omega | m_omega;
			Mat3<T> omegaHat = pose_proc::hat(m_omega);

			// We need to be careful to not divide by zero. 
			if (theta2 > T(FLT_EPSILON)) {
				// Rodrigues' formula.
				const T theta = sqrt(theta2);
				return Mat3<T>::identity() + omegaHat * (sin(theta) / theta) + omegaHat * omegaHat * ((1 - cos(theta)) / theta2);
			}
			else {
				// If theta squared is too small, we use only a first order approximation of the exponential formula.
				// R = exp(omega) = I + omega_hat + omega_hat^2 / 2! + ...
				return  Mat3<T>::identity() + omegaHat;
			}
		}

		/**
		 * Converts the rotation to a unit quaternion.
		 */
		 UnitQuaternion<T> toQuaternion() const;

		/**
		 * Getters.
		 */
		 const Vec3<T>& getOmega() const { return m_omega; }
		 Vec3<T>& getOmega() { return m_omega; }

		 T* getData() {
			return &m_omega[0];
		}

		/**
		 * Stream output.
		 */
		 friend std::ostream& operator<<(std::ostream& os, const SO3<T>& obj) {
			return os << obj.m_omega;
		}

		/**
		 * Indexing operators.
		 */
		template<unsigned i>
		 T& operator[](I<i>) {
			static_assert(i < 3, "Index out of bounds.");
			return m_omega[I<i>()];
		}

		template<unsigned i>
		 const T& operator[](I<i>) const {
			static_assert(i < 3, "Index out of bounds.");
			return m_omega[I<i>()];
		}

	private:
		Vec3<T> m_omega; // Axis of rotation, scaled by the angle of rotation (in radians).
	};

	using SO3d = SO3<double>;
	using SO3f = SO3<float>;


	/**
	 * Implemention of conversion to unit quaternion.
	 * We forward declared UnitQuaternion before, therefore we need to include it now to define the method.
	 */
	#include "UnitQuaternion.h"

	template <typename T>
	 UnitQuaternion<T> SO3<T>::toQuaternion() const {
		const T theta2 = m_omega | m_omega;

		// For points not at the origin, the full conversion is numerically stable.
		if (theta2 > T(0.0)) {
			const T theta = sqrt(theta2);
			const T halfTheta = theta * T(0.5);
			const T k = sin(halfTheta) / theta;
			return UnitQuaternion<T>{
				cos(halfTheta),
					m_omega.x() * k,
					m_omega.y() * k,
					m_omega.z() * k
			};
		}
		else {
			// At the origin, sqrt() will produce NaN in the derivative since
			// the argument is zero.  By approximating with a Taylor series,
			// and truncating at one term, the value and first derivatives will be
			// computed correctly when Ceres Jets are used.
			const T k{ 0.5 };
			return UnitQuaternion<T>{
				T(1.0),
					m_omega.x() * k,
					m_omega.y() * k,
					m_omega.z() * k
			};
		}
	}

} // namespace matrix_lib
