#pragma once
#include "SO3.h"

namespace matrix_lib {	

	/**
	 * Unit quaternions are a 4-parameter representation of rotation.
	 */
	template <typename T> 
	class UnitQuaternion {
	public:
		/**
		 * Constructor for identity rotation.
		 */
		 UnitQuaternion() : m_real{ T(1) }, m_imag{ T(0) } {}

		/**
		 * Explicit constructor.
		 */
		 UnitQuaternion(T r, T i, T j, T k) {
			m_real = r;
			m_imag = Vec3<T>(i, j, k);
			normalize();
		}

		 UnitQuaternion(T real, const Vec3<T>& imag) : m_real(real), m_imag(imag) {
			normalize();
		}

		/**
		 * Constructor from 3x3 rotation matrix (needs conversion).
		 */
		 UnitQuaternion(const Mat3<T>& matrix) {
			constructFromMatrix(matrix);
		}

		/**
		 * Initialize with float4.
		 */
		 UnitQuaternion(const float4& v) :
			m_real{ v.x },
			m_imag{ v.y, v.z, v.w }
		{ }

		/**
		 * Copy data from other template type.
		 */
		template<typename U>
		friend class UnitQuaternion;

		template <class U>
		 UnitQuaternion(const UnitQuaternion<U>& other) :
			m_real{ other.m_real },
			m_imag{ other.m_imag } 
		{ }

		/**
		 * Getters.
		 */
		 T real() const { return m_real; }
		 T& real() { return m_real; }
		 const Vec3<T>& imag() const { return m_imag; }
		 Vec3<T>& imag() { return m_imag; }

		 T* getData() {
			return &m_real;
		}

		/**
		 * Multiplies two unit quaternions. 
		 */
		 UnitQuaternion<T> operator*(const UnitQuaternion<T>& other) const {
			// Derivation: https://en.wikipedia.org/wiki/Quaternion
			return UnitQuaternion<T>{
				m_real * other.m_real - m_imag.x() * other.m_imag.x() - m_imag.y() * other.m_imag.y() - m_imag.z() * other.m_imag.z(),
				m_real * other.m_imag.x() + m_imag.x() * other.m_real + m_imag.y() * other.m_imag.z() - m_imag.z() * other.m_imag.y(),
				m_real * other.m_imag.y() - m_imag.x() * other.m_imag.z() + m_imag.y() * other.m_real + m_imag.z() * other.m_imag.x(),
				m_real * other.m_imag.z() + m_imag.x() * other.m_imag.y() - m_imag.y() * other.m_imag.x() + m_imag.z() * other.m_real
			};
		}

		 UnitQuaternion<T> operator*=(const UnitQuaternion<T>& other) {
			(*this) = (*this) * other;
			return *this;
		}

		/**
		 * Applies a unit quaternion to a 3D point.
		 */
		 Vec3<T> apply(const Vec3<T>& point) const {
			// p = 0 + px * i + py * j + pz * k
			// pRotated = q * p * qInverse

			const T t2 = m_real * m_imag.x();
			const T t3 = m_real * m_imag.y();
			const T t4 = m_real * m_imag.z();
			const T t5 = -m_imag.x() * m_imag.x();
			const T t6 = m_imag.x() * m_imag.y();
			const T t7 = m_imag.x() * m_imag.z();
			const T t8 = -m_imag.y() * m_imag.y();
			const T t9 = m_imag.y() * m_imag.z();
			const T t1 = -m_imag.z() * m_imag.z();
			
			return Vec3<T>{
				T(2) * ((t8 + t1) * point.x() + (t6 - t4) * point.y() + (t3 + t7) * point.z()) + point.x(),
				T(2) * ((t4 + t6) * point.x() + (t5 + t1) * point.y() + (t9 - t2) * point.z()) + point.y(),
				T(2) * ((t7 - t3) * point.x() + (t2 + t9) * point.y() + (t5 + t8) * point.z()) + point.z()
			};
		}

		 Vec3<T> operator*(const Vec3<T>& point) const {
			return apply(point);
		}

		/**
		 * Returns the unit quaternion that performs the inverse rotation.
		 */
		 UnitQuaternion getInverse() const {
			// Since we have a unit quaternion, we only need to conjugate the quaternion to get inverse.
			return getConjugate();
		}
		
		/**
		 * Transforms a quaternion into a 3x3 rotation matrix.
		 */
		 Mat3<T> matrix() const {
			// Assumption: Quaternion is normalized.
			Mat3<T> m;
			m(0, 0) = m_real*m_real + m_imag[0] * m_imag[0] - m_imag[1] * m_imag[1] - m_imag[2] * m_imag[2];
			m(0, 1) = T(2.0) * (m_imag[0] * m_imag[1] - m_real*m_imag[2]);
			m(0, 2) = T(2.0) * (m_imag[0] * m_imag[2] + m_real*m_imag[1]);

			m(1, 0) = T(2.0) * (m_imag[0] * m_imag[1] + m_real*m_imag[2]);
			m(1, 1) = m_real*m_real - m_imag[0] * m_imag[0] + m_imag[1] * m_imag[1] - m_imag[2] * m_imag[2];
			m(1, 2) = T(2.0) * (m_imag[1] * m_imag[2] - m_real*m_imag[0]);

			m(2, 0) = T(2.0) * (m_imag[0] * m_imag[2] - m_real*m_imag[1]);
			m(2, 1) = T(2.0) * (m_imag[1] * m_imag[2] + m_real*m_imag[0]);
			m(2, 2) = m_real*m_real - m_imag[0] * m_imag[0] - m_imag[1] * m_imag[1] + m_imag[2] * m_imag[2];

			return m;
		}

		 SO3<T> toAxisAngle() const;

		/**
		 * Interpolates the given unit quaternions with given interpolation weights.
		 * The interpolation weights need to sum up to 1.
		 */
		 static UnitQuaternion<T> interpolate(const vector<UnitQuaternion<T>>& quaternions, const vector<T>& weights) {
			runtime_assert(quaternions.size() == weights.size(), "Number of quaternions should equal the number of weights.");
			Vec4<T> interpolatedVector{ T(0) };
			const unsigned nQuaternions = quaternions.size();

			for (int i = 0; i < nQuaternions; ++i) {
				const auto& quaternion = quaternions[i];
				const T& weight = weights[i];
				interpolatedVector.x() += weight * quaternion.m_real;
				interpolatedVector.y() += weight * quaternion.m_imag[0];
				interpolatedVector.z() += weight * quaternion.m_imag[1];
				interpolatedVector.w() += weight * quaternion.m_imag[2];
			}

			// Unit quaternion constructor automatically normalizes the 4D vector. Since we don't 
			// want to normalize twice, we only call the constructor.
			return UnitQuaternion{ interpolatedVector.x(), interpolatedVector.y(), interpolatedVector.z(), interpolatedVector.w() };
		}

		/**
		 * Indexing operators.
		 */
		 T& operator[](I<0>) { return m_real; }
		 const T& operator[](I<0>) const { return m_real; }

		 T& operator[](I<1>) { return m_imag[I<0>()]; }
		 const T& operator[](I<1>) const { return m_imag[I<0>()]; }

		 T& operator[](I<2>) { return m_imag[I<1>()]; }
		 const T& operator[](I<2>) const { return m_imag[I<1>()]; }

		 T& operator[](I<3>) { return m_imag[I<2>()]; }
		 const T& operator[](I<3>) const { return m_imag[I<2>()]; }

	private:
		T m_real;		// The real part of the quaternion
		Vec3<T> m_imag;	// The imaginary part of the quaternion

		/**
		 * Computes the (squared) length of quaternion (should be always 1, except perhaps at initialization).
		 */
		 T squaredLength() const { return m_real * m_real + m_imag[0] * m_imag[0] + m_imag[1] * m_imag[1] + m_imag[2] * m_imag[2]; }
		 T length() const { return sqrt(squaredLength()); }

		/**
		 * Normalizes the quaternion (needed only at initialization).
		 */
		 void normalize() {
			static T eps = T(0.00001);
			T l = length();
			if (l > eps) {
				m_real /= l;
				m_imag /= l;
			}
			else {
				m_real = 1;
				m_imag = Vec3<T>(0, 0, 0);
			}
		}
		
		/**
		 * Returns a conjugated quaternion.
		 */
		 UnitQuaternion<T> getConjugate() const {
			return UnitQuaternion(m_real, Vec3<T>(-m_imag[0], -m_imag[1], -m_imag[2]));
		}

		/**
		 * Constructs a unit quaternion from 3x3 rotation matrix.
		 */
		 void constructFromMatrix(const Mat3<T>& m) {
			T m00 = m(0, 0);	T m01 = m(0, 1);	T m02 = m(0, 2);
			T m10 = m(1, 0);	T m11 = m(1, 1);	T m12 = m(1, 2);
			T m20 = m(2, 0);	T m21 = m(2, 1);	T m22 = m(2, 2);

			T tr = m00 + m11 + m22;

			T qw, qx, qy, qz;
			if (tr > 0) {
				T S = sqrt(tr + (T)1.0) * 2; // S=4*qw 
				qw = (T)0.25 * S;
				qx = (m21 - m12) / S;
				qy = (m02 - m20) / S;
				qz = (m10 - m01) / S;
			}
			else if ((m00 > m11)&(m00 > m22)) {
				T S = sqrt((T)1.0 + m00 - m11 - m22) * (T)2; // S=4*qx 
				qw = (m21 - m12) / S;
				qx = (T)0.25 * S;
				qy = (m01 + m10) / S;
				qz = (m02 + m20) / S;
			}
			else if (m11 > m22) {
				T S = sqrt((T)1.0 + m11 - m00 - m22) * (T)2; // S=4*qy
				qw = (m02 - m20) / S;
				qx = (m01 + m10) / S;
				qy = (T)0.25 * S;
				qz = (m12 + m21) / S;
			}
			else {
				T S = sqrt((T)1.0 + m22 - m00 - m11) * (T)2; // S=4*qz
				qw = (m10 - m01) / S;
				qx = (m02 + m20) / S;
				qy = (m12 + m21) / S;
				qz = (T)0.25 * S;
			}
			m_real = qw;
			m_imag = Vec3<T>(qx, qy, qz);
			normalize();
		}

		/**
		 * Read a quaternion from a stream.
		 */
		 friend std::istream& operator>> (std::istream& s, UnitQuaternion<T>& q);
	};

	/**
	 * Implementation of stream operations.
	 */
	template<typename T>
	 std::istream& operator>> (std::istream& s, UnitQuaternion<T>& q) { return (s >> q.m_real >> q.m_imag); }
	
	template<typename T>
	 std::ostream& operator<< (std::ostream& s, const UnitQuaternion<T>& q) { return (s << q.real() << " " << q.imag()); }

	using Quatd = UnitQuaternion<double>;
	using Quatf = UnitQuaternion<float>;


	/**
	 * Implementation of conversion to SO3.
	 */
	template <typename T>
	 SO3<T> UnitQuaternion<T>::toAxisAngle() const  {
		const T& q1 = m_imag.x();
		const T& q2 = m_imag.y();
		const T& q3 = m_imag.z();
		const T sinSquaredTheta = q1 * q1 + q2 * q2 + q3 * q3;

		// For quaternions representing non-zero rotation, the conversion
		// is numerically stable.
		if (sinSquaredTheta > T(0.0)) {
			const T sinTheta = sqrt(sinSquaredTheta);
			const T& cosTheta = m_real;

			// If cosTheta is negative, theta is greater than pi/2, which
			// means that angle for the angleAxis vector which is 2 * theta
			// would be greater than pi.
			//
			// While this will result in the correct rotation, it does not
			// result in a normalized angle-axis vector.
			//
			// In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
			// which is equivalent saying
			//
			//   theta - pi = atan(sin(theta - pi), cos(theta - pi))
			//              = atan(-sin(theta), -cos(theta))
			//
			const T twoTheta =
				T(2.0) * ((cosTheta < 0.0)
				? atan2(-sinTheta, -cosTheta)
				: atan2(sinTheta, cosTheta));
			const T k = twoTheta / sinTheta;

			return SO3<T>{
				q1 * k,
					q2 * k,
					q3 * k
			};
		}
		else {
			// For zero rotation, sqrt() will produce NaN in the derivative since
			// the argument is zero.  By approximating with a Taylor series,
			// and truncating at one term, the value and first derivatives will be
			// computed correctly when Ceres Jets are used.
			const T k(2.0);
			return SO3<T>{
				q1 * k,
				q2 * k,
				q3 * k
			};
		}
	}

} // namespace matrix_lib
