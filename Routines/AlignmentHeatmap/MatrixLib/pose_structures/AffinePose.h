#pragma once
#include "Pose.h"
#include "PoseArrays.h"

namespace matrix_lib {

	/**
	 * Pose (A, t), where A is a 3x3 affine matrix and t is 3D translation.
	 */
	template<typename T> class Pose<T, PoseType::AFFINE> {
	public:
		enum { Type = PoseType::AFFINE };

		 Pose() :
			m_affineMatrix{ Mat3<T>::identity() },
			m_translation{ Vec3<T>{} }
		{ }

		/**
		 * Constructor.
		 * @param	affineMatrix	3x3 affine matrix (all 9 elements are any real numbers)
		 * @param	translation		Translation vector
		 */
		 Pose(const Mat3<T>& affineMatrix, const Vec3<T>& translation) :
			m_affineMatrix{ affineMatrix },
			m_translation{ translation }
		{ }

		/**
		 * Constructor.
		 * @param	matrix			4x4 affine matrix (3x3 affine matrix and 3x1 translation)
		 */
		 Pose(const Mat4<T>& matrix) :
			m_affineMatrix{ matrix.getMatrix3x3() },
			m_translation{ matrix.getTranslation() }
		{ }

		/**
		 * Initialize with float4.
		 */
		 Pose(const float4& matrixXRow, const float4& matrixYRow, const float4& matrixZRow, const float4& translation) :
			m_affineMatrix{ matrixXRow, matrixYRow, matrixZRow },
			m_translation{ translation }
		{ }

		/**
		 * Copy data from other template type.
		 */
		template<typename U, int PoseType>
		friend class Pose;

		template <class U>
		 Pose(const Pose<U, PoseType::AFFINE>& other) :
			m_affineMatrix{ other.m_affineMatrix },
			m_translation{ other.m_translation }
		{ }

		/**
		 * Interface implementation.
		 */
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

		 void update(const Pose<T, PoseType::AFFINE>& pose) {
			m_affineMatrix = pose.getAffineMatrix();
			m_translation = pose.getTranslation();
		}

		 static Pose<T, PoseType::AFFINE> identity() {
			return Pose<T, PoseType::AFFINE>{};
		}

		 static Pose<T, PoseType::AFFINE> translation(const Vec3<T>& translation) {
			return Pose<T, PoseType::AFFINE>{ Mat3<T>::identity(), translation };
		}

		 static Pose<T, PoseType::AFFINE> rotation(const Mat3<T>& rotation) {
			return Pose<T, PoseType::AFFINE>{ rotation, Vec3<T>{ 0, 0, 0 } };
		}

		 static Pose<T, PoseType::AFFINE> pose(const Mat4<T>& pose) {
			return Pose<T, PoseType::AFFINE>{ pose };
		}

		 static Pose<T, PoseType::AFFINE> pose(const Mat3<T>& rotation, const Vec3<T>& translation) {
			return Pose<T, PoseType::AFFINE >{ rotation, translation };
		}

		 static Pose<T, PoseType::AFFINE> interpolate(const vector<Pose<T, PoseType::AFFINE>>& poses, const vector<T>& weights) {
			runtime_assert(poses.size() == weights.size(), "Number of poses and weights should be the same.");
			Pose<T, PoseType::AFFINE> interpolatedPose{ Mat3<T>::zero(), Vec3<T>{ 0, 0, 0 } };

			const unsigned nPoses = poses.size();
			for (auto i = 0; i < nPoses; ++i) {
				interpolatedPose.m_affineMatrix += weights[i] * poses[i].m_affineMatrix;
				interpolatedPose.m_translation += weights[i] * poses[i].m_translation;
			}

			return interpolatedPose;
		}

		/**
		 * Multiplies this pose with another pose (from the right).
		 * The general multiplication of two poses is defined outside the class.
		 */
		 Pose& operator*=(const Pose& rhs) {
			m_translation = m_affineMatrix * rhs.m_translation + m_translation;
			m_affineMatrix = m_affineMatrix * rhs.m_affineMatrix;
			return *this;
		}

		/**
		 * Getters.
		 */
		 const Mat3<T>& getAffineMatrix() const { return m_affineMatrix; }
		 Mat3<T>& getAffineMatrix() { return m_affineMatrix; }
		 const Vec3<T>& getTranslation() const {	return m_translation; }
		 Vec3<T>& getTranslation() {	return m_translation; }

		/**
		 * Methods to save and load a pose, represented as a 4x4 matrix.
		 * @param	filename	Location of a matrix file
		 */
		void saveMatrixToFile(const std::string& filename) const {
			matrix().saveMatrixToFile(filename);
		}

		void loadMatrixFromFile(const std::string& filename) {
			Mat4<T> matrix;
			matrix.loadMatrixFromFile(filename);
			m_affineMatrix = matrix.getRotation();
			m_translation = matrix.getTranslation();
		}

		/**
		 * Stream ouput.
		 */
		  friend std::ostream& operator<<(std::ostream& os, const Pose<T, PoseType::AFFINE>& obj) {
			return os << "translation: " << obj.getTranslation() << ", rotation: " << obj.getAffineMatrix();
		}

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

} // namespace matrix_lib
