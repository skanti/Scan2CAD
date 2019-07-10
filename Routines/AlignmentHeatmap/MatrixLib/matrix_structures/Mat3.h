#pragma once
#include "Vec3.h"

namespace matrix_lib {

	/**
	 * 3x3 matrix.
	 * The arrangement of the matrix is row-like.
	 * The index of a specific position is:
	 * 0  1  2
	 * 3  4  5
	 * 6  7  8
	 */
	template <class T>
	class Mat3 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		// An uninitialized matrix.
		 Mat3() {
			setZero();
		}

		// Initialize with values stored in an array.
		 Mat3(const T* values) {
			for (unsigned int i = 0; i < 9; i++) {
				m_matrix[i] = values[i];
			}
		}

		// Initialize from 3 row vectors.
		 Mat3(const Vec3<T>& v0, const Vec3<T>& v1, const Vec3<T>& v2) {
			m_matrix[0] = v0.x();
			m_matrix[1] = v0.y();
			m_matrix[2] = v0.z();
			m_matrix[3] = v1.x();
			m_matrix[4] = v1.y();
			m_matrix[5] = v1.z();
			m_matrix[6] = v2.x();
			m_matrix[7] = v2.y();
			m_matrix[8] = v2.z();
		}

		// Initializes the matrix row wise.
		 Mat3(const T& m00, const T& m01, const T& m02,
		          const T& m10, const T& m11, const T& m12,
		          const T& m20, const T& m21, const T& m22) {
			m_matrix[0] = m00;
			m_matrix[1] = m01;
			m_matrix[2] = m02;
			m_matrix[3] = m10;
			m_matrix[4] = m11;
			m_matrix[5] = m12;
			m_matrix[6] = m20;
			m_matrix[7] = m21;
			m_matrix[8] = m22;
		}


		// Initialize with a matrix from another type.
		template <class U>
		 Mat3(const Mat3<U>& other) {
			for (unsigned int i = 0; i < 9; i++) {
				m_matrix[i] = T(other.getData()[i]);
			}
		}

		// Overwrite the matrix with an identity-matrix.
		 void setIdentity() {
			setScale(T(1), T(1), T(1));
		}

		 static Mat3 identity() {
			Mat3 res;
			res.setIdentity();
			return res;
		}

		// Sets the matrix zero (or a specified value).
		 void setZero(T v = T(0)) {
			m_matrix[0] = m_matrix[1] = m_matrix[2] = v;
			m_matrix[3] = m_matrix[4] = m_matrix[5] = v;
			m_matrix[6] = m_matrix[7] = m_matrix[8] = v;
		}

		 static Mat3 zero(T v = T(0)) {
			Mat3 res;
			res.setZero(v);
			return res;
		}


		 static Mat3 rotationX(T angle) {
			Mat3 res;
			res.setRotationX(angle);
			return res;
		}


		 static Mat3 rotationY(T angle) {
			Mat3 res;
			res.setRotationY(angle);
			return res;
		}


		 static Mat3 rotationZ(T angle) {
			Mat3 res;
			res.setRotationZ(angle);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle in degrees, ccw).
		 void setRotation(T yaw, T pitch, T roll) {
			*this = rotationY(yaw) * rotationX(pitch) * rotationZ(roll);
		}

		 static Mat3 rotation(T yaw, T pitch, T roll) {
			Mat3 res;
			res.setRotation(yaw, pitch, roll);
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		 void setScale(T x, T y, T z) {
			m_matrix[0] = x;
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = T(0);
			m_matrix[4] = y;
			m_matrix[5] = T(0);
			m_matrix[6] = T(0);
			m_matrix[7] = T(0);
			m_matrix[8] = z;
		}

		 static Mat3 scale(T x, T y, T z) {
			Mat3 res;
			res.setScale(x, y, z);
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		 void setScale(T s) {
			setScale(s, s, s);
		}

		 static Mat3 scale(T s) {
			Mat3 res;
			res.setScale(s);
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		 void setScale(const Vec3<T>& v) {
			m_matrix[0] = v.x();
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = T(0);
			m_matrix[4] = v.y();
			m_matrix[5] = T(0);
			m_matrix[6] = T(0);
			m_matrix[7] = T(0);
			m_matrix[8] = v.z();
		}

		 static Mat3 scale(const Vec3<T>& v) {
			Mat3 res;
			res.setScale(v);
			return res;
		}

		// Overwrite the matrix with a diagonal matrix.
		 void setDiag(T x, T y, T z) {
			setScale(x, y, z);
		}

		 static Mat3 diag(T x, T y, T z) {
			Mat3 res;
			res.setDiag(x, y, z);
			return res;
		}

		/**
		 * Basic operations.
		 */
		// Equal operator.
		 bool operator==(const Mat3<T>& other) const {
			for (unsigned i = 0; i < 9; i++) {
				if (m_matrix[i] != other[i]) return false;
			}
			return true;
		}

		// Not equal operator.
		 bool operator!=(const Mat3<T>& other) const {
			return !(*this == other);
		}

		 T trace() const {
			return (m_matrix[0] + m_matrix[4] + m_matrix[8]);
		}

		// Return the product of the operand with matrix.
		 Mat3 operator*(const Mat3& other) const {
			Mat3<T> result;
			//TODO unroll the loop
			for (unsigned char i = 0; i < 3; i++) {
				for (unsigned char j = 0; j < 3; j++) {
					result.at(i, j) =
						this->at(i, 0) * other.at(0, j) +
						this->at(i, 1) * other.at(1, j) +
						this->at(i, 2) * other.at(2, j);
				}
			}
			return result;
		}

		// Multiply operand with matrix b.
		 Mat3& operator*=(const Mat3& other) {
			Mat3<T> prod = (*this) * other;
			*this = prod;
			return *this;
		}

		// Multiply each element in the matrix with a scalar factor.
		 Mat3 operator*(T r) const {
			Mat3<T> result;
			for (unsigned int i = 0; i < 9; i++) {
				result[i] = m_matrix[i] * r;
			}
			return result;
		}

		// Multiply each element in the matrix with a scalar factor.
		 Mat3& operator*=(T r) {
			for (unsigned int i = 0; i < 9; i++) {
				m_matrix[i] *= r;
			}
			return *this;
		}

		// Divide the matrix by a scalar factor.
		 Mat3 operator/(T r) const {
			Mat3<T> result;
			for (unsigned int i = 0; i < 9; i++) {
				result[i] = m_matrix[i] / r;
			}
			return result;
		}

		// Divide each element in the matrix with a scalar factor.
		 Mat3& operator/=(T r) {
			for (unsigned int i = 0; i < 9; i++) {
				m_matrix[i] /= r;
			}
			return *this;
		}

		// Transform a 3D-Vector with the matrix.
		 Vec3<T> operator*(const Vec3<T>& v) const {
			return Vec3<T>(
				m_matrix[0] * v[0] + m_matrix[1] * v[1] + m_matrix[2] * v[2],
				m_matrix[3] * v[0] + m_matrix[4] * v[1] + m_matrix[5] * v[2],
				m_matrix[6] * v[0] + m_matrix[7] * v[1] + m_matrix[8] * v[2]
			);
		}

		// Return the sum of the operand with matrix b.
		 Mat3 operator+(const Mat3& other) const {
			Mat3<T> result;
			for (unsigned int i = 0; i < 9; i++) {
				result[i] = m_matrix[i] + other[i];
			}
			return result;
		}

		// Add matrix other to the operand.
		 Mat3& operator+=(const Mat3& other) {
			for (unsigned int i = 0; i < 9; i++) {
				m_matrix[i] += other[i];
			}
			return *this;
		}

		// Return the difference of the operand with matrix b.
		 Mat3 operator-(const Mat3& other) const {
			Mat3<T> result;
			for (unsigned int i = 0; i < 9; i++) {
				result.m_matrix[i] = m_matrix[i] - other[i];
			}
			return result;
		}

		// Subtract matrix other from the operand.
		 Mat3 operator-=(const Mat3& other) {
			for (unsigned int i = 0; i < 9; i++) {
				m_matrix[i] -= other[i];
			}
			return *this;
		}

		// Return the determinant of the matrix.
		 T det() const {
			return det3x3();
		}

		/**
		 * Indexing operators.
		 */
		// Access element of matrix at the given row and column for constant access.
		 T at(unsigned char row, unsigned char col) const {
			return m_matrix[col + row * 3];
		}

		// Access element of matrix at the given row and column.
		 T& at(unsigned char row, unsigned char col) {
			return m_matrix[col + row * 3];
		}

		// Access element of matrix at the given row and column for constant access.
		 T operator()(unsigned int row, unsigned int col) const {
			return m_matrix[col + row * 3];
		}

		// Access element of matrix at the given row and column.
		 T& operator()(unsigned int row, unsigned int col) {
			return m_matrix[col + row * 3];
		}

		// Access i-th element of the matrix for constant access.
		 const T& operator[](unsigned int i) const {
			return m_matrix[i];
		}

		// Access i-th element of the matrix.
		 T& operator[](unsigned int i) {
			return m_matrix[i];
		}

		/**
		 * Getters/setters.
		 */
		// Get the x column out of the matrix.
		 Vec3<T> xcol() const {
			return Vec3<T>(m_matrix[0], m_matrix[3], m_matrix[6]);
		}

		// Get the y column out of the matrix.
		 Vec3<T> ycol() const {
			return Vec3<T>(m_matrix[1], m_matrix[4], m_matrix[7]);
		}

		// Get the y column out of the matrix.
		 Vec3<T> zcol() const {
			return Vec3<T>(m_matrix[2], m_matrix[5], m_matrix[8]);
		}

		// Get the x row out of the matrix.
		 Vec3<T> xrow() const {
			return Vec3<T>(m_matrix[0], m_matrix[1], m_matrix[2]);
		}

		// Get the y row out of the matrix.
		 Vec3<T> yrow() const {
			return Vec3<T>(m_matrix[3], m_matrix[4], m_matrix[5]);
		}

		// Get the y row out of the matrix.
		 Vec3<T> zrow() const {
			return Vec3<T>(m_matrix[6], m_matrix[7], m_matrix[8]);
		}

		 std::string toString(const std::string& seperator = ",") const {
			std::string result;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++) {
					result += to_string(m_matrix[i + j * 3]);
					if (i != 3 || j != 3)
						result += seperator;
				}
			return result;
		}

		 const T* getData() const {
			return &m_matrix[0];
		}

		 T* getData() {
			return &m_matrix[0];
		}

		// Return the inverse matrix; but does not change the current matrix.
		 Mat3 getInverse() const {
			T inv[9];

			inv[0] = m_matrix[4] * m_matrix[8] - m_matrix[5] * m_matrix[7];
			inv[1] = -m_matrix[1] * m_matrix[8] + m_matrix[2] * m_matrix[7];
			inv[2] = m_matrix[1] * m_matrix[5] - m_matrix[2] * m_matrix[4];

			inv[3] = -m_matrix[3] * m_matrix[8] + m_matrix[5] * m_matrix[6];
			inv[4] = m_matrix[0] * m_matrix[8] - m_matrix[2] * m_matrix[6];
			inv[5] = -m_matrix[0] * m_matrix[5] + m_matrix[2] * m_matrix[3];

			inv[6] = m_matrix[3] * m_matrix[7] - m_matrix[4] * m_matrix[6];
			inv[7] = -m_matrix[0] * m_matrix[7] + m_matrix[1] * m_matrix[6];
			inv[8] = m_matrix[0] * m_matrix[4] - m_matrix[1] * m_matrix[3];

			T matrixDet = det();

			T matrixDetr = T(1.0) / matrixDet;

			Mat3<T> res;
			for (unsigned int i = 0; i < 9; i++) {
				res[i] = inv[i] * matrixDetr;
			}
			return res;

		}

		// Overwrite the current matrix with its inverse.
		 void invert() {
			*this = getInverse();
		}

		// Return the transposed matrix.
		 Mat3 getTranspose() const {
			Mat3<T> result;
			for (unsigned char x = 0; x < 3; x++) {
				result.at(x, 0) = at(0, x);
				result.at(x, 1) = at(1, x);
				result.at(x, 2) = at(2, x);
			}
			return result;
		}

		// Transpose the matrix in place
		 void transpose() {
			*this = getTranspose();
		}

		// Compute the matrix representation of the cross-product.
		 static Mat3 skewSymmetric(const Vec3<T>& v) {
			return Mat3(
				0, -v.z(), v.y(),
				v.z(), 0, -v.x(),
				-v.y(), v.x(), 0
			);
		}

		// Save matrix to .mat file.
		 void saveMatrixToFile(const std::string& name) const {
			std::ofstream out(name.c_str());
			out << 3 << " " << 3 << "\n";
			for (unsigned int i = 0; i < 3; i++) {
				for (unsigned int j = 0; j < 3; j++) {
					out << (*this)(i, j) << " ";
				}
				out << "\n";
			}
		}

		// Load matrix from .mat file.
		 void loadMatrixFromFile(const std::string& name) {
			std::ifstream in(name.c_str());
			unsigned int height, width;
			in >> height >> width;
			for (unsigned int i = 0; i < 3; i++) {
				for (unsigned int j = 0; j < 3; j++) {
					in >> (*this)(i, j);
				}
			}
		}

	private:
		T m_matrix[9];

		// Calculate determinant of a 3x3 sub-matrix given by the indices of the rows and columns.
		 T det3x3(unsigned int i0 = 0, unsigned int i1 = 1, unsigned int i2 = 2, unsigned int j0 = 0,
		                 unsigned int j1 = 1, unsigned int j2 = 2) const {
			return
				((*this)(i0, j0) * (*this)(i1, j1) * (*this)(i2, j2))
				+ ((*this)(i0, j1) * (*this)(i1, j2) * (*this)(i2, j0))
				+ ((*this)(i0, j2) * (*this)(i1, j0) * (*this)(i2, j1))
				- ((*this)(i2, j0) * (*this)(i1, j1) * (*this)(i0, j2))
				- ((*this)(i2, j1) * (*this)(i1, j2) * (*this)(i0, j0))
				- ((*this)(i2, j2) * (*this)(i1, j0) * (*this)(i0, j1));
		}

	};

	/**
	 * Math operations.
	 */
	template <class T>
	 Mat3<T> operator*(T s, const Mat3<T>& m) {
		return m * s;
	}

	template <class T>
	 Mat3<T> operator/(T s, const Mat3<T>& m) {
		return Mat3<T>(
			s / m(0, 0), s / m(0, 1), s / m(0, 2),
			s / m(1, 0), s / m(1, 1), s / m(1, 2),
			s / m(2, 0), s / m(2, 1), s / m(2, 2)
		);
	}

	template <class T>
	 Mat3<T> operator+(T s, const Mat3<T>& m) {
		return m + s;
	}

	template <class T>
	 Mat3<T> operator-(T s, const Mat3<T>& m) {
		return -m + s;
	}

	// Writes to a stream.
	template <class T>
	 std::ostream& operator<<(std::ostream& s, const Mat3<T>& m) {
		return (
			s <<
			m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << std::endl <<
			m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << std::endl <<
			m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << std::endl
		);
	}

	// Reads from a stream.
	template <class T>
	 std::istream& operator>>(std::istream& s, const Mat3<T>& m) {
		return (
			s >>
			m(0, 0) >> m(0, 1) >> m(0, 2) >>
			m(1, 0) >> m(1, 1) >> m(1, 2) >>
			m(2, 0) >> m(2, 1) >> m(2, 2)
		);
	}

	typedef Mat3<int> Mat3i;
	typedef Mat3<int> Mat3u;
	typedef Mat3<float> Mat3f;
	typedef Mat3<double> Mat3d;

} // namespace matrix_lib
