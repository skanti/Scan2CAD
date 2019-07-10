#pragma once
#include "PoseIncrement.h"
#include "UnitQuaternion.h"

namespace matrix_lib {

	/**
	 * Rigid pose increment, represented as a 6 dimensional vector: axis-angle rotation and translation.
	 * This pose increment is used for optimization of rigid poses on Lie algebra.
	 */
	template<typename T>
	class PoseIncrement<T, PoseType::SE3> {
	public:
		enum { Type = PoseType::SE3 };

		/**
		 * Default constructor.
		 */
		 PoseIncrement() { reset(); }

		/**
		 * Constructor from parameter array.
		 * Important: The size of data needs to be at least 6 * sizeof(T).
		 */
		 PoseIncrement(const T* data) :
			m_rotation{ data }, 
			m_translation{ data + 3 }
		{ }
			
		/**
		 * Explicit constructor.
		 */
		 PoseIncrement(const Vec3<T>& rotation, const Vec3<T>& translation) :
			m_rotation{ rotation },
			m_translation{ translation } 
		{ }
		
		 PoseIncrement(const Vec6<T>& twist) :
			PoseIncrement(Vec3<T>{ twist[0], twist[1], twist[2] }, Vec3<T>{ twist[3], twist[4], twist[5] })
		{ }

		/**
		 * Initialize with float4.
		 */
		 PoseIncrement(const float4& rotation, const float4& translation) :
			m_rotation{ rotation },
			m_translation{ translation }
		{ }

		/**
		 * Constructor from 4x4 pose matrix (needs conversion).
		 */
		 PoseIncrement(const Mat4<T>& matrix) :
			m_rotation{ matrix.getRotation() },
			m_translation{ matrix.getTranslation() }
		{ }

		 PoseIncrement(const Mat3<T>& rotation, const Vec3<T>& translation) :
			m_rotation{ rotation },
			m_translation{ translation }
		{ }

		/**
		 * Copy data from other template type.
		 */
		template<typename U, int PoseType>
		friend class PoseIncrement;

		template <class U>
		 PoseIncrement(const PoseIncrement<U, PoseType::SE3>& other) :
			m_rotation{ other.m_rotation },
			m_translation{ other.m_translation }
		{ }

		/**
		 * Interface implementation.
		 */
		 T* getData() {
			// The data structure is packed, therefore the translation follows continuously in the memory.
			return m_rotation.getData();
		}
		
		template <typename U> 
		 Vec3<Promote<T, U>> apply(const Vec3<U>& point) const {
			// TODO: Implement the real SE3 (with different translation) => going to be slower.
			return m_rotation * point + m_translation;
		}

		template <typename U>
		 Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const {
			return apply(point);
		}

		template <typename U>
		 Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const {
			// TODO: Implement the real SE3 (with different translation) => going to be slower.
			return m_rotation * point;
		}

		template <typename U>
		 Vec3<Promote<T, U>> translate(const Vec3<U>& point) const {
			// TODO: Implement the real SE3 (with different translation) => going to be slower.
			return point + m_translation;
		}

		 Mat4<T> matrix() const {
			return Mat4<T>{ m_rotation.matrix(), m_translation };
		}

		 void reset() {
			m_rotation = T(0); 
			m_translation = T(0);
		}

		 static PoseIncrement<T, PoseType::SE3> identity() { return PoseIncrement<T, PoseType::SE3>{}; }
		 static PoseIncrement<T, PoseType::SE3> translation(const Vec3<T>& translation) { return PoseIncrement<T, PoseType::SE3>{ Mat3<T>::identity(), translation }; }
		 static PoseIncrement<T, PoseType::SE3> rotation(const Mat3<T>& rotation) { return PoseIncrement<T, PoseType::SE3>{ rotation, Vec3<T>{} }; }
		 static PoseIncrement<T, PoseType::SE3> pose(const Mat4<T>& pose) { return PoseIncrement<T, PoseType::SE3>{ pose }; }
		 static PoseIncrement<T, PoseType::SE3> pose(const Mat3<T>& rotation, const Vec3<T>& translation) { return PoseIncrement<T, PoseType::SE3>{ rotation, translation }; }

		/**
		 * Stream output.
		 */
		 friend std::ostream& operator<<(std::ostream& os, const PoseIncrement<T, PoseType::SE3>& obj) {
			return os << "rotation = " << obj.m_rotation << ", translation = " << obj.m_translation;
		}

		/**
		 * Getters.
		 */
		 const SO3<T>& getAxisAngle() const { return m_rotation; }
		 SO3<T>& getAxisAngle() { return m_rotation; }
		 const Vec3<T>& getTranslation() const { return m_translation; }
		 Vec3<T>& getTranslation() { return m_translation; }

		/**
		 * Indexing operators.
		 */
		 SO3<T>& at(I<0>) { return m_rotation; }
		 const SO3<T>& at(I<0>) const { return m_rotation; }
		 Vec3<T>& at(I<1>) { return m_translation; }
		 const Vec3<T>& at(I<1>) const { return m_translation; }

		template<unsigned i>
		 T& operator[](I<i>) {
			static_assert(i < 6, "Index out of bounds.");
			return at(I<i / 3>())[I<i % 3>()];
		}

		template<unsigned i>
		 const T& operator[](I<i>) const {
			static_assert(i < 6, "Index out of bounds.");
			return at(I<i / 3>())[I<i % 3>()];
		}

	private:
		SO3<T> m_rotation;
		Vec3<T> m_translation;
	};

}
