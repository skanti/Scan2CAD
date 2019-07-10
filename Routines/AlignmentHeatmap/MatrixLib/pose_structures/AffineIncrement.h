#pragma once
#include "PoseIncrement.h"

namespace matrix_lib {

	/**
	 * Affine pose increment, where affine matrix is a 3x3 matrix and translation is 3x1 vector.
	 * This pose increment is used for optimization of affine poses.
	 */
 	template<typename T>
	class PoseIncrement<T, PoseType::AFFINE> {
	public:
		enum { Type = PoseType::AFFINE };

		/**
		 * Default constructor (identity transformation).
		 */
		 PoseIncrement() { reset(); }

		/**
		 * Constructor from a parameter array.
		 * Important: The size of data needs to be at least 12 * sizeof(T).
		 */
		 PoseIncrement(const T* data) :
			m_affineMatrix{ data },
			m_translation{ data + 9 }
		{ }

		/**
		 * Explicit constructor.
		 */
		 PoseIncrement(const Mat4<T>& matrix) :
			m_affineMatrix{ matrix.getMatrix3x3() },
			m_translation{ matrix.getTranslation() }
		{ }

		 PoseIncrement(const Mat3<T>& affineMatrix, const Vec3<T>& translation) :
			m_affineMatrix{ affineMatrix },
			m_translation{ translation }
		{ }

		/**
		 * Initialize with float4.
		 */
		 PoseIncrement(const float4& matrixXRow, const float4& matrixYRow, const float4& matrixZRow, const float4& translation) :
			m_affineMatrix{ matrixXRow, matrixYRow, matrixZRow },
			m_translation{ translation }
		{ }

		/**
		 * Copy data from other template type.
		 */
		template<typename U, int PoseType>
		friend class PoseIncrement;

		template <class U>
		 PoseIncrement(const PoseIncrement<U, PoseType::AFFINE>& other) :
			m_affineMatrix{ other.m_affineMatrix },
			m_translation{ other.m_translation }
		{ }

		/**
		 * Interface implementation.
		 */
		 T* getData() {
			// The data structure is packed, therefore the translation follows continuously in the memory.
			return m_affineMatrix.getData();
		}

		template <typename U>
		 Vec3<Promote<T, U>> apply(const Vec3<U>& point) const {
			return m_affineMatrix * point + m_translation;
		}

		template <typename U>
		 Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const {
			return apply(point);
		}

		template <typename U>
		 Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const {
			return m_affineMatrix * point;
		}

		template <typename U>
		 Vec3<Promote<T, U>> translate(const Vec3<U>& point) const {
			return point + m_translation;
		}

		 Mat4<T> matrix() const {
			return Mat4<T>{ m_affineMatrix, m_translation };
		}

		 void reset() { m_affineMatrix.setIdentity(); m_translation = T(0); }

		 static PoseIncrement<T, PoseType::AFFINE> identity() { return PoseIncrement<T, PoseType::AFFINE>{}; }
		 static PoseIncrement<T, PoseType::AFFINE> translation(const Vec3<T>& translation) { return PoseIncrement<T, PoseType::AFFINE>{ Mat3<T>::identity(), translation }; }
		 static PoseIncrement<T, PoseType::AFFINE> rotation(const Mat3<T>& rotation) { return PoseIncrement<T, PoseType::AFFINE>{ rotation, Vec3<T>{} }; }
		 static PoseIncrement<T, PoseType::AFFINE> pose(const Mat4<T>& pose) { return PoseIncrement<T, PoseType::AFFINE>{ pose }; }
		 static PoseIncrement<T, PoseType::AFFINE> pose(const Mat3<T>& rotation, const Vec3<T>& translation) { return PoseIncrement<T, PoseType::AFFINE>{ rotation, translation }; }

		/**
		 * Stream output.
		 */
		 friend std::ostream& operator<<(std::ostream& os, const PoseIncrement<T, PoseType::AFFINE>& obj) {
			return os << "affineMatrix = " << obj.m_affineMatrix << ", translation = " << obj.m_translation;
		}

		/**
		 * Getters.
		 */
		 const Mat3<T>& getAffineMatrix() const { return m_affineMatrix; }
		 Mat3<T>& getAffineMatrix() { return m_affineMatrix; }
		 const Vec3<T>& getTranslation() const { return m_translation; }
		 Vec3<T>& getTranslation() { return m_translation; }

		/**
		 * Indexing operators.
		 */
		 Mat3<T>& at(I<0>) { return m_affineMatrix; }
		 const Mat3<T>& at(I<0>) const { return m_affineMatrix; }
		 Vec3<T>& at(I<1>) { return m_translation; }
		 const Vec3<T>& at(I<1>) const { return m_translation; }

		template<unsigned i>
		 T& operator[](I<i>) {
			static_assert(i < 16, "Index out of bounds.");
			return at(I<i / 12>())[I<i % 12>()];
		}

		template<unsigned i>
		 const T& operator[](I<i>) const {
			static_assert(i < 16, "Index out of bounds.");
			return at(I<i / 12>())[I<i % 12>()];
		}

	private:
		Mat3<T> m_affineMatrix;
		Vec3<T> m_translation;
	};

}
