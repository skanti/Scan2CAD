#pragma once
#include "MatrixLib/matrix_structures/VecArrays.h"

namespace matrix_lib {
	
	/**
	 * SE3 pose (SO3 rotation + translation).
	 */
	template<bool isInterface>
	struct ASE3f : public SoA<
		isInterface,
		typename TL<
			float4,	// Axis-angle (SO3) rotation
			float4	// Translation
		>::type
	> {
		 float4* rotation() { return t()[I<0>()]; }
		 const float4* rotation() const { return t()[I<0>()]; }
		 float4* translation() { return t()[I<1>()]; }
		 const float4* translation() const { return t()[I<1>()]; }
	};

	using iASE3f = ASE3f<true>;
	using sASE3f = ASE3f<false>;


	/**
	 * Quaternion pose (unit quaternion rotation + translation).
	 */
	template<bool isInterface>
	struct AQuaternionPosef : public SoA<
		isInterface,
		typename TL<
			float4,	// Unit quaternion rotation
			float4	// Translation
		>::type
	> {
		 float4* rotation() { return t()[I<0>()]; }
		 const float4* rotation() const { return t()[I<0>()]; }
		 float4* translation() { return t()[I<1>()]; }
		 const float4* translation() const { return t()[I<1>()]; }
	};

	using iAQuaternionPosef = AQuaternionPosef<true>;
	using sAQuaternionPosef = AQuaternionPosef<false>;


	/**
	 * Affine pose (3x3 affine matrix + translation).
	 */
	template<bool isInterface>
	struct AAffinePosef : public SoA<
		isInterface,
		typename TL<
			AMat3f<isInterface>,	// Affine 3x3 matrix
			float4					// Translation
		>::type
	> {
		 AMat3f<isInterface>& rotation() { return t()[I<0>()]; }
		 const AMat3f<isInterface>& rotation() const { return t()[I<0>()]; }
		 float4* translation() { return t()[I<1>()]; }
		 const float4* translation() const { return t()[I<1>()]; }
	};

	using iAAffinePosef = AAffinePosef<true>;
	using sAAffinePosef = AAffinePosef<false>;

} // namespace matrix_lib
