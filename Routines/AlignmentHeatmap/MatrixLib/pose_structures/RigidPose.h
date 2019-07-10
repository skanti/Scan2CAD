#pragma once
#include "Pose.h"
#include "UnitQuaternion.h"
#include "SO3.h"
#include "PoseArrays.h"

namespace matrix_lib {
	
	/**
	 * Pose (R, t), where R is rotation and t is translation.
	 * Rotations are represented as quaternions.
	 */
	template<typename T> 
	class Pose<T, PoseType::QUATERNION> {
	public:
		enum { Type = PoseType::QUATERNION };
		
		 Pose() = default;

		/**
		 * Constructor from quaternion notation.
		 * @param   rotation    Rotation given in quaternion notation
		 * @param   translation Translation vector
		 */
		 Pose(const UnitQuaternion<T>& rotation, const Vec3<T>& translation) :
			m_rotation{ rotation },
			m_translation{ translation } 
		{ }

		/**
		 * Constructors from matrix notation (needs conversion).
		 */
		 Pose(const Mat3<T>& rotation, const Vec3<T>& translation) :
			m_rotation{ rotation },
			m_translation{ translation }
		{ }

		 Pose(const Mat4<T>& pose) : Pose(pose.getRotation(), pose.getTranslation()) { }

		/**
	     * Constructors from axis-angle notation (needs conversion).
	     * Pose given in SE(3) notation (w, t), where w is axis-angle representation
		 * for rotation and t is a translation vector.
		 */
		 Pose(const Vec3<T>& rotation, const Vec3<T>& translation) :
			m_rotation{ SO3<T>(rotation).toQuaternion() },
			m_translation{ translation } 
		{ }

		 Pose(const Vec6<T>& pose) : Pose(Vec3<T>{ pose[0], pose[1], pose[2] }, Vec3<T>{ pose[3], pose[4], pose[5] }) { }

		/**
		 * Initialize with float4.
		 */
		 Pose(const float4& rotation, const float4& translation) :
			m_rotation{ rotation },
			m_translation{ translation }
		{ }

		/**
		 * Copy data from other template type.
		 */
		template<typename U, int PoseType>
		friend class Pose;

		template <class U>
		 Pose(const Pose<U, PoseType::QUATERNION>& other) :
			m_rotation{ other.m_rotation },
			m_translation{ other.m_translation }
		{ }
	
		/**
		 * Interface implementation.
		 */
		template <typename U> 
		 Vec3<Promote<T, U>> apply(const Vec3<U>& point) const {
			return m_rotation * point + m_translation;
		}

		template <typename U> 
		 Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const {
			return apply(point);
		}

		template <typename U> 
		 Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const {
			return m_rotation * point;
		}

		template <typename U> 
		 Vec3<Promote<T, U>> translate(const Vec3<U>& point) const {
			return point + m_translation;
		}

		 Mat4<T> matrix() const {
			return Mat4<T>{ m_rotation.matrix(), m_translation };
		}

		 void update(const Pose<T, PoseType::QUATERNION>& pose) {
			m_translation = pose.getTranslation();
			m_rotation = pose.getQuaternion();
		}

		 static Pose<T, PoseType::QUATERNION> identity() {
			return Pose<T, PoseType::QUATERNION>{};
		}

		 static Pose<T, PoseType::QUATERNION> translation(const Vec3<T>& translation) {
			return Pose<T, PoseType::QUATERNION>{ UnitQuaternion<T>{ 1, 0, 0, 0 }, translation };
		}

		 static Pose<T, PoseType::QUATERNION> rotation(const Mat3<T>& rotation) {
			return Pose<T, PoseType::QUATERNION>{ rotation, Vec3<T>{ 0, 0, 0 } };
		}

		 static Pose<T, PoseType::QUATERNION> pose(const Mat4<T>& pose) {
			return Pose<T, PoseType::QUATERNION>{ pose };
		}

		 static Pose<T, PoseType::QUATERNION> pose(const Mat3<T>& rotation, const Vec3<T>& translation) {
			return Pose<T, PoseType::QUATERNION >{ rotation, translation };
		}

		 static Pose<T, PoseType::QUATERNION> interpolate(const vector<Pose<T, PoseType::QUATERNION>>& poses, const vector<T>& weights) {
			runtime_assert(poses.size() == weights.size(), "Number of poses should equal the number of weights.");
			Pose<T, PoseType::QUATERNION> interpolatedPose{ UnitQuaternion<T>{ 1, 0, 0, 0 }, Vec3<T>{ T(0) } };
			const unsigned nPoses = poses.size();

			vector<UnitQuaternion<T>> quaternions;
			quaternions.reserve(nPoses);
			for (auto i = 0; i < nPoses; ++i) {
				quaternions.emplace_back(poses[i].m_rotation);
				interpolatedPose.m_translation += weights[i] * poses[i].m_translation;
			}
			interpolatedPose.m_rotation = UnitQuaternion<T>::interpolate(quaternions, weights);

			return interpolatedPose;
		}


		/**
		 * Multiplies this pose with another pose (from the right).
		 * The general multiplication of two poses is defined outside the class.
		 */
		 Pose& operator*=(const Pose& rhs) {
			m_translation = m_rotation * rhs.m_translation + m_translation;
			m_rotation = m_rotation * rhs.m_rotation;
			return *this;
		}

		/**
		 * Getters.
		 */
		 const UnitQuaternion<T>& getQuaternion() const { return m_rotation; }
		 UnitQuaternion<T>& getQuaternion() { return m_rotation; }
		 const Vec3<T>& getTranslation() const {	return m_translation; }
		 Vec3<T>& getTranslation() {	return m_translation; }

		/**
		 * Setters.
		 */
		 void setQuaternion(const UnitQuaternion<T>& quaternion) { m_rotation = quaternion; }
		 void setTranslation(const Vec3<T>& translation) { m_translation = translation; }

		/**
		 * Returns the inverse pose.
		 */
		 Pose<T, PoseType::QUATERNION> getInverse() const {
			const UnitQuaternion<T> rotationInverse = m_rotation.getInverse();
			const Vec3<T> translationInverse = rotationInverse * (-m_translation);
			return Pose<T, PoseType::QUATERNION>{ rotationInverse, translationInverse };
		}

		/**
		 * Computes the projection matrix by composing intrinsic matrix and camera pose matrix.
		 */
		 Mat4<T> getVisionProjection(const Mat3<T>& intrinsicMatrix) const {
			Mat4<T> projection = Mat4<T>::identity();
			projection.setRotationMatrix(intrinsicMatrix * m_rotation.matrix());
			projection.setTranslationVector(intrinsicMatrix * m_translation);
			return projection;
		}

		/**
		 * Computes the projection matrix by composing intrinsic matrix and camera pose matrix.
		 * Transformed the points from [0, width] x [0, height] x [depthMin, depthMax] to [-1, 1] x [-1, 1] x [0, 1].
		 */
		 Mat4<T> getGraphicsProjection(const Mat3<T>& intrinsicMatrix, unsigned width, unsigned height, float depthMin, float depthMax) const {
			Mat4<T> projection = Mat4<T>::identity();
			projection.setMatrix3x3(intrinsicMatrix);

			Mat4<T> pixelToNDC{
				T(2) / T(width), T(0), T(1) / T(width) - T(1), T(0),
				T(0), T(-2) / T(height), T(1) - T(1) / T(height), T(0),
				T(0), T(0), T(depthMax / (depthMax - depthMin)), T(depthMax * depthMin / (depthMin - depthMax)),
				T(0), T(0), T(1), T(0)
			};

			return pixelToNDC * projection * matrix();
		}

		/**
		 * Additional static constructors.
		 */
		 static Pose<T, PoseType::QUATERNION> rotation(const UnitQuaternion<T>& rotation) {
			Pose<T, PoseType::QUATERNION> pose{ rotation, Vec3<T>{ 0, 0, 0 } };
			return pose;
		}		

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
			m_rotation = UnitQuaternion<T>{ matrix.getRotation() };
			m_translation = matrix.getTranslation();
		}

		/**
		 * Stream ouput.
		 */
		 friend std::ostream& operator<<(std::ostream& os, const Pose<T, PoseType::QUATERNION>& obj) {
			return os << "translation: " << obj.getTranslation() << ", rotation: " << obj.getQuaternion();
		}

		/**
		 * Indexing operators.
		 */
		 UnitQuaternion<T>& at(I<0>) { return m_rotation; }
		 const UnitQuaternion<T>& at(I<0>) const { return m_rotation; }
		 Vec3<T>& at(I<1>) { return m_translation; }
		 const Vec3<T>& at(I<1>) const { return m_translation; }

		template<unsigned i>
		 T& operator[](I<i>) {
			static_assert(i < 7, "Index out of bounds.");
			return at(I<i / 4>())[I<i % 4>()];
		}

		template<unsigned i>
		 const T& operator[](I<i>) const {
			static_assert(i < 7, "Index out of bounds.");
			return at(I<i / 4>())[I<i % 4>()];
		}

	private:
		UnitQuaternion<T> m_rotation;
		Vec3<T> m_translation;
	};

} // namespace matrix_lib
