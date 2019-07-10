#pragma once
#include <Solo/meta_structures/TypeFinder.h>

#include "MatrixLib/pose_structures/PoseStructuresInclude.h"
#include "MatrixLib/matrix_structures/MatrixStructuresInclude.h"

// Implementation of Solo methods for BaseDeform classes.
template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Vec2<T>> {
	using type = typename BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Vec3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Vec4<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T, unsigned N>
struct solo::BaseTypeHelper<matrix_lib::VecX<T, N>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Mat2<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Mat3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Mat4<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::UnitQuaternion<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::SO3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::RigidPose<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::SE3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::AffinePose<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::AffineIncrement<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};


template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Vec2<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Vec3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Vec4<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T, unsigned N>
struct solo::ResultTypeHelper<matrix_lib::VecX<T, N>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Mat2<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Mat3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Mat4<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::UnitQuaternion<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::SO3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::RigidPose<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::SE3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::AffinePose<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::AffineIncrement<T>> {
	using type = typename ResultTypeHelper<T>::type;
};