#pragma once
#include "MatrixLib/utils/SoloSupport.h"

namespace matrix_lib {

	// This file defines basic vector and matrix operations. For matrices row-wise
	// storage is assumed.

	/**
	 * Element-wise vector addition.
	 */
	template<typename A, typename B>
	 auto add2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] + b[I<0>()],
			a[I<1>()] + b[I<1>()]
		);
	}

	template<typename A, typename B>
	 auto add3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] + b[I<0>()],
			a[I<1>()] + b[I<1>()],
			a[I<2>()] + b[I<2>()]
		);
	}

	template<typename A, typename B>
	 auto add4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] + b[I<0>()],
			a[I<1>()] + b[I<1>()],
			a[I<2>()] + b[I<2>()],
			a[I<3>()] + b[I<3>()]
		);
	}

	/**
	 * Element-wise vector subtraction.
	 */
	template<typename A, typename B>
	 auto sub2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] - b[I<0>()],
			a[I<1>()] - b[I<1>()]
		);
	}

	template<typename A, typename B>
	 auto sub3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] - b[I<0>()],
			a[I<1>()] - b[I<1>()],
			a[I<2>()] - b[I<2>()]
		);
	}

	template<typename A, typename B>
	 auto sub4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] - b[I<0>()],
			a[I<1>()] - b[I<1>()],
			a[I<2>()] - b[I<2>()],
			a[I<3>()] - b[I<3>()]
		);
	}

	/**
	 * Element-wise vector multiplication.
	 */
	template<typename A, typename B>
	 auto mul2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] * b[I<0>()],
			a[I<1>()] * b[I<1>()]
		);
	}

	template<typename A, typename B>
	 auto mul3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] * b[I<0>()],
			a[I<1>()] * b[I<1>()],
			a[I<2>()] * b[I<2>()]
		);
	}

	template<typename A, typename B>
	 auto mul4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] * b[I<0>()],
			a[I<1>()] * b[I<1>()],
			a[I<2>()] * b[I<2>()],
			a[I<3>()] * b[I<3>()]
		);
	}

	/**
	 * Element-wise vector division.
	 */
	template<typename A, typename B>
	 auto div2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] / b[I<0>()],
			a[I<1>()] / b[I<1>()]
		);
	}

	template<typename A, typename B>
	 auto div3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] / b[I<0>()],
			a[I<1>()] / b[I<1>()],
			a[I<2>()] / b[I<2>()]
		);
	}

	template<typename A, typename B>
	 auto div4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] / b[I<0>()],
			a[I<1>()] / b[I<1>()],
			a[I<2>()] / b[I<2>()],
			a[I<3>()] / b[I<3>()]
		);
	}

	/**
	 * Element-wise scalar addition.
	 */
	template<typename V, typename C>
	 auto shift2(const V& v, const C& c) {
		return makeTuple(
			v[I<0>()] + c,
			v[I<1>()] + c
		);
	}

	template<typename V, typename C>
	 auto shift3(const V& v, const C& c) {
		return makeTuple(
			v[I<0>()] + c,
			v[I<1>()] + c,
			v[I<2>()] + c
		);
	}

	template<typename V, typename C>
	 auto shift4(const V& v, const C& c) {
		return makeTuple(
			v[I<0>()] + c,
			v[I<1>()] + c,
			v[I<2>()] + c,
			v[I<3>()] + c
		);
	}

	/**
	 * Element-wise scalar multiplication.
	 */
	template<typename V, typename C>
	 auto scale2(const V& v, const C& c) {
		return makeTuple(
			v[I<0>()] * c,
			v[I<1>()] * c
		);
	}

	template<typename V, typename C>
	 auto scale3(const V& v, const C& c) {
		return makeTuple(
			v[I<0>()] * c,
			v[I<1>()] * c,
			v[I<2>()] * c
		);
	}

	template<typename V, typename C>
	 auto scale4(const V& v, const C& c) {
		return makeTuple(
			v[I<0>()] * c,
			v[I<1>()] * c,
			v[I<2>()] * c,
			v[I<3>()] * c
		);
	}

	/**
	 * Dot product.
	 */
	template<typename A, typename B>
	 auto dot2(const A& a, const B& b) {
		return a[I<0>()] * b[I<0>()] + a[I<1>()] * b[I<1>()];
	}

	template<typename A, typename B>
	 auto dot3(const A& a, const B& b) {
		return a[I<0>()] * b[I<0>()] + a[I<1>()] * b[I<1>()] + a[I<2>()] * b[I<2>()];
	}

	template<typename A, typename B>
	 auto dot4(const A& a, const B& b) {
		return a[I<0>()] * b[I<0>()] + a[I<1>()] * b[I<1>()] + a[I<2>()] * b[I<2>()] + a[I<3>()] * b[I<3>()];
	}

	/**
	 * Cross product.
	 */
	template<typename A, typename B>
	 auto cross3(const A& a, const B& b) {
		const auto a0 = a[I<0>()];
		const auto a1 = a[I<1>()];
		const auto a2 = a[I<2>()];
		const auto b0 = b[I<0>()];
		const auto b1 = b[I<1>()];
		const auto b2 = b[I<2>()];

		return makeTuple(
			a1 * b2 - a2 * b1,
			a2 * b0 - a0 * b2,
			a0 * b1 - a1 * b0
		);
	}

	/**
	 * Element-wise matrix addition.
	 */
	template<typename A, typename B>
	 auto add2x2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] + b[I<0>()],
			a[I<1>()] + b[I<1>()],
			a[I<2>()] + b[I<2>()],
			a[I<3>()] + b[I<3>()]
		);
	}

	template<typename A, typename B>
	 auto add3x3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] + b[I<0>()],
			a[I<1>()] + b[I<1>()],
			a[I<2>()] + b[I<2>()],
			a[I<3>()] + b[I<3>()],
			a[I<4>()] + b[I<4>()],
			a[I<5>()] + b[I<5>()],
			a[I<6>()] + b[I<6>()],
			a[I<7>()] + b[I<7>()],
			a[I<8>()] + b[I<8>()]
		);
	}

	template<typename A, typename B>
	 auto add4x4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] + b[I<0>()],
			a[I<1>()] + b[I<1>()],
			a[I<2>()] + b[I<2>()],
			a[I<3>()] + b[I<3>()],
			a[I<4>()] + b[I<4>()],
			a[I<5>()] + b[I<5>()],
			a[I<6>()] + b[I<6>()],
			a[I<7>()] + b[I<7>()],
			a[I<8>()] + b[I<8>()],
			a[I<9>()] + b[I<9>()],
			a[I<10>()] + b[I<10>()],
			a[I<11>()] + b[I<11>()],
			a[I<12>()] + b[I<12>()],
			a[I<13>()] + b[I<13>()],
			a[I<14>()] + b[I<14>()],
			a[I<15>()] + b[I<15>()]
		);
	}

	/**
	 * Element-wise matrix subtraction.
	 */
	template<typename A, typename B>
	 auto sub2x2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] - b[I<0>()],
			a[I<1>()] - b[I<1>()],
			a[I<2>()] - b[I<2>()],
			a[I<3>()] - b[I<3>()]
		);
	}

	template<typename A, typename B>
	 auto sub3x3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] - b[I<0>()],
			a[I<1>()] - b[I<1>()],
			a[I<2>()] - b[I<2>()],
			a[I<3>()] - b[I<3>()],
			a[I<4>()] - b[I<4>()],
			a[I<5>()] - b[I<5>()],
			a[I<6>()] - b[I<6>()],
			a[I<7>()] - b[I<7>()],
			a[I<8>()] - b[I<8>()]
		);
	}

	template<typename A, typename B>
	 auto sub4x4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] - b[I<0>()],
			a[I<1>()] - b[I<1>()],
			a[I<2>()] - b[I<2>()],
			a[I<3>()] - b[I<3>()],
			a[I<4>()] - b[I<4>()],
			a[I<5>()] - b[I<5>()],
			a[I<6>()] - b[I<6>()],
			a[I<7>()] - b[I<7>()],
			a[I<8>()] - b[I<8>()],
			a[I<9>()] - b[I<9>()],
			a[I<10>()] - b[I<10>()],
			a[I<11>()] - b[I<11>()],
			a[I<12>()] - b[I<12>()],
			a[I<13>()] - b[I<13>()],
			a[I<14>()] - b[I<14>()],
			a[I<15>()] - b[I<15>()]
		);
	}

	/**
	 * Element-wise matrix multiplication.
	 */
	template<typename A, typename B>
	 auto mul2x2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] * b[I<0>()],
			a[I<1>()] * b[I<1>()],
			a[I<2>()] * b[I<2>()],
			a[I<3>()] * b[I<3>()]
		);
	}

	template<typename A, typename B>
	 auto mul3x3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] * b[I<0>()],
			a[I<1>()] * b[I<1>()],
			a[I<2>()] * b[I<2>()],
			a[I<3>()] * b[I<3>()],
			a[I<4>()] * b[I<4>()],
			a[I<5>()] * b[I<5>()],
			a[I<6>()] * b[I<6>()],
			a[I<7>()] * b[I<7>()],
			a[I<8>()] * b[I<8>()]
		);
	}

	template<typename A, typename B>
	 auto mul4x4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] * b[I<0>()],
			a[I<1>()] * b[I<1>()],
			a[I<2>()] * b[I<2>()],
			a[I<3>()] * b[I<3>()],
			a[I<4>()] * b[I<4>()],
			a[I<5>()] * b[I<5>()],
			a[I<6>()] * b[I<6>()],
			a[I<7>()] * b[I<7>()],
			a[I<8>()] * b[I<8>()],
			a[I<9>()] * b[I<9>()],
			a[I<10>()] * b[I<10>()],
			a[I<11>()] * b[I<11>()],
			a[I<12>()] * b[I<12>()],
			a[I<13>()] * b[I<13>()],
			a[I<14>()] * b[I<14>()],
			a[I<15>()] * b[I<15>()]
		);
	}

	/**
	 * Element-wise matrix division.
	 */
	template<typename A, typename B>
	 auto div2x2(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] / b[I<0>()],
			a[I<1>()] / b[I<1>()],
			a[I<2>()] / b[I<2>()],
			a[I<3>()] / b[I<3>()]
		);
	}

	template<typename A, typename B>
	 auto div3x3(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] / b[I<0>()],
			a[I<1>()] / b[I<1>()],
			a[I<2>()] / b[I<2>()],
			a[I<3>()] / b[I<3>()],
			a[I<4>()] / b[I<4>()],
			a[I<5>()] / b[I<5>()],
			a[I<6>()] / b[I<6>()],
			a[I<7>()] / b[I<7>()],
			a[I<8>()] / b[I<8>()]
		);
	}

	template<typename A, typename B>
	 auto div4x4(const A& a, const B& b) {
		return makeTuple(
			a[I<0>()] / b[I<0>()],
			a[I<1>()] / b[I<1>()],
			a[I<2>()] / b[I<2>()],
			a[I<3>()] / b[I<3>()],
			a[I<4>()] / b[I<4>()],
			a[I<5>()] / b[I<5>()],
			a[I<6>()] / b[I<6>()],
			a[I<7>()] / b[I<7>()],
			a[I<8>()] / b[I<8>()],
			a[I<9>()] / b[I<9>()],
			a[I<10>()] / b[I<10>()],
			a[I<11>()] / b[I<11>()],
			a[I<12>()] / b[I<12>()],
			a[I<13>()] / b[I<13>()],
			a[I<14>()] / b[I<14>()],
			a[I<15>()] / b[I<15>()]
		);
	}

	/**
	 * Element-wise scalar addition.
	 */
	template<typename M, typename C>
	 auto shift2x2(const M& m, const C& c) {
		return makeTuple(
			m[I<0>()] + c,
			m[I<1>()] + c,
			m[I<2>()] + c,
			m[I<3>()] + c
		);
	}

	template<typename M, typename C>
	 auto shift3x3(const M& m, const C& c) {
		return makeTuple(
			m[I<0>()] + c,
			m[I<1>()] + c,
			m[I<2>()] + c,
			m[I<3>()] + c,
			m[I<4>()] + c,
			m[I<5>()] + c,
			m[I<6>()] + c,
			m[I<7>()] + c,
			m[I<8>()] + c
		);
	}

	template<typename M, typename C>
	 auto shift4x4(const M& m, const C& c) {
		return makeTuple(
			m[I<0>()] + c,
			m[I<1>()] + c,
			m[I<2>()] + c,
			m[I<3>()] + c,
			m[I<4>()] + c,
			m[I<5>()] + c,
			m[I<6>()] + c,
			m[I<7>()] + c,
			m[I<8>()] + c,
			m[I<9>()] + c,
			m[I<10>()] + c,
			m[I<11>()] + c,
			m[I<12>()] + c,
			m[I<13>()] + c,
			m[I<14>()] + c,
			m[I<15>()] + c
		);
	}

	/**
	 * Element-wise scalar multiplication.
	 */
	template<typename M, typename C>
	 auto scale2x2(const M& m, const C& c) {
		return makeTuple(
			m[I<0>()] * c,
			m[I<1>()] * c,
			m[I<2>()] * c,
			m[I<3>()] * c
		);
	}

	template<typename M, typename C>
	 auto scale3x3(const M& m, const C& c) {
		return makeTuple(
			m[I<0>()] * c,
			m[I<1>()] * c,
			m[I<2>()] * c,
			m[I<3>()] * c,
			m[I<4>()] * c,
			m[I<5>()] * c,
			m[I<6>()] * c,
			m[I<7>()] * c,
			m[I<8>()] * c
		);
	}

	template<typename M, typename C>
	 auto scale4x4(const M& m, const C& c) {
		return makeTuple(
			m[I<0>()] * c,
			m[I<1>()] * c,
			m[I<2>()] * c,
			m[I<3>()] * c,
			m[I<4>()] * c,
			m[I<5>()] * c,
			m[I<6>()] * c,
			m[I<7>()] * c,
			m[I<8>()] * c,
			m[I<9>()] * c,
			m[I<10>()] * c,
			m[I<11>()] * c,
			m[I<12>()] * c,
			m[I<13>()] * c,
			m[I<14>()] * c,
			m[I<15>()] * c
		);
	}

	/**
	 * Matrix-vector multiplication.
	 */
	template<typename A, typename B>
	 auto mv2x2(const A& m, const B& v) {
		const auto v0 = v[I<0>()];
		const auto v1 = v[I<1>()];

		return makeTuple(
			m[I<0>()] * v0 + m[I<1>()] * v1,
			m[I<2>()] * v0 + m[I<3>()] * v1
		);
	}

	template<typename A, typename B>
	 auto mv3x3(const A& m, const B& v) {
		const auto v0 = v[I<0>()];
		const auto v1 = v[I<1>()];
		const auto v2 = v[I<2>()];

		return makeTuple(
			m[I<0>()] * v0 + m[I<1>()] * v1 + m[I<2>()] * v2,
			m[I<3>()] * v0 + m[I<4>()] * v1 + m[I<5>()] * v2,
			m[I<6>()] * v0 + m[I<7>()] * v1 + m[I<8>()] * v2
		);
	}

	template<typename A, typename B>
	 auto mv4x4(const A& m, const B& v) {
		const auto v0 = v[I<0>()];
		const auto v1 = v[I<1>()];
		const auto v2 = v[I<2>()];
		const auto v3 = v[I<3>()];

		return makeTuple(
			m[I<0>()] * v0 + m[I<1>()] * v1 + m[I<2>()] * v2 + m[I<3>()] * v3,
			m[I<4>()] * v0 + m[I<5>()] * v1 + m[I<6>()] * v2 + m[I<7>()] * v3,
			m[I<8>()] * v0 + m[I<9>()] * v1 + m[I<10>()] * v2 + m[I<11>()] * v3,
			m[I<12>()] * v0 + m[I<13>()] * v1 + m[I<14>()] * v2 + m[I<15>()] * v3
		);
	}

	/**
	 * Matrix-matrix multiplication.
	 */
	template<typename A, typename B>
	 auto mm2x2(const A& a, const B& b) {
		const auto a00 = a[I<0>()]; const auto a01 = a[I<1>()];
		const auto a10 = a[I<2>()]; const auto a11 = a[I<3>()];

		const auto b00 = b[I<0>()]; const auto b01 = b[I<1>()];
		const auto b10 = b[I<2>()]; const auto b11 = b[I<3>()];

		return makeTuple(
			a00 * b00 + a01 * b10, // c00
			a00 * b01 + a01 * b11, // c01
			a10 * b00 + a11 * b10, // c10
			a10 * b01 + a11 * b11  // c11
		);
	}

	template<typename A, typename B>
	 auto mm3x3(const A& a, const B& b) {
		const auto a00 = a[I<0>()]; const auto a01 = a[I<1>()]; const auto a02 = a[I<2>()];
		const auto a10 = a[I<3>()]; const auto a11 = a[I<4>()]; const auto a12 = a[I<5>()];
		const auto a20 = a[I<6>()]; const auto a21 = a[I<7>()]; const auto a22 = a[I<8>()];

		const auto b00 = b[I<0>()]; const auto b01 = b[I<1>()]; const auto b02 = b[I<2>()];
		const auto b10 = b[I<3>()]; const auto b11 = b[I<4>()]; const auto b12 = b[I<5>()];
		const auto b20 = b[I<6>()]; const auto b21 = b[I<7>()]; const auto b22 = b[I<8>()];

		return makeTuple(
			a00 * b00 + a01 * b10 + a02 * b20, // c00
			a00 * b01 + a01 * b11 + a02 * b21, // c01
			a00 * b02 + a01 * b12 + a02 * b22, // c02
			a10 * b00 + a11 * b10 + a12 * b20, // c10
			a10 * b01 + a11 * b11 + a12 * b21, // c11
			a10 * b02 + a11 * b12 + a12 * b22, // c12
			a20 * b00 + a21 * b10 + a22 * b20, // c20
			a20 * b01 + a21 * b11 + a22 * b21, // c21
			a20 * b02 + a21 * b12 + a22 * b22  // c22
		);
	}

	template<typename A, typename B>
	 auto mm4x4(const A& a, const B& b) {
		const auto a00 = a[I<0>()];  const auto a01 = a[I<1>()];  const auto a02 = a[I<2>()];  const auto a03 = a[I<3>()];
		const auto a10 = a[I<4>()];  const auto a11 = a[I<5>()];  const auto a12 = a[I<6>()];  const auto a13 = a[I<7>()];
		const auto a20 = a[I<8>()];  const auto a21 = a[I<9>()];  const auto a22 = a[I<10>()]; const auto a23 = a[I<11>()];
		const auto a30 = a[I<12>()]; const auto a31 = a[I<13>()]; const auto a32 = a[I<14>()]; const auto a33 = a[I<15>()];

		const auto b00 = b[I<0>()];  const auto b01 = b[I<1>()];  const auto b02 = b[I<2>()];  const auto b03 = b[I<3>()];
		const auto b10 = b[I<4>()];  const auto b11 = b[I<5>()];  const auto b12 = b[I<6>()];  const auto b13 = b[I<7>()];
		const auto b20 = b[I<8>()];  const auto b21 = b[I<9>()];  const auto b22 = b[I<10>()]; const auto b23 = b[I<11>()];
		const auto b30 = b[I<12>()]; const auto b31 = b[I<13>()]; const auto b32 = b[I<14>()]; const auto b33 = b[I<15>()];

		return makeTuple(
			a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30, // c00
			a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31, // c01
			a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32, // c02
			a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33, // c03
			a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30, // c10
			a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31, // c11
			a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32, // c12
			a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33, // c13
			a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30, // c20
			a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31, // c21
			a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32, // c22
			a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33, // c23
			a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30, // c30
			a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31, // c31
			a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32, // c32
			a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33  // c33
		);
	}


	/**
	 * Computes the determinant of a matrix.
	 */
	template<typename M>
	 auto det2x2(const M& m) {
		return m[I<0>()] * m[I<3>()] - m[I<1>()] * m[I<2>()];
	}

	template<typename M>
	 auto det3x3(const M& m) {
		const auto m00 = m[I<0>()]; const auto m01 = m[I<1>()]; const auto m02 = m[I<2>()];
		const auto m10 = m[I<3>()]; const auto m11 = m[I<4>()]; const auto m12 = m[I<5>()];
		const auto m20 = m[I<6>()]; const auto m21 = m[I<7>()]; const auto m22 = m[I<8>()];

		return (m00 * m11 * m22 - m00 * m12 * m21) -
			(m01 * m10 * m22 - m01 * m12 * m20) +
			(m02 * m10 * m21 - m02 * m11 * m20);
	}

	template<typename M>
	 auto det4x4(const M& m) {
		const auto m00 = m[I<0>()];  const auto m01 = m[I<1>()];  const auto m02 = m[I<2>()];  const auto m03 = m[I<3>()];
		const auto m10 = m[I<4>()];  const auto m11 = m[I<5>()];  const auto m12 = m[I<6>()];  const auto m13 = m[I<7>()];
		const auto m20 = m[I<8>()];  const auto m21 = m[I<9>()];  const auto m22 = m[I<10>()]; const auto m23 = m[I<11>()];
		const auto m30 = m[I<12>()]; const auto m31 = m[I<13>()]; const auto m32 = m[I<14>()]; const auto m33 = m[I<15>()];

		const auto subDet00 = (m11 * m22 * m33 - m11 * m23 * m32) -
			(m12 * m21 * m33 - m12 * m23 * m31) +
			(m13 * m21 * m32 - m13 * m22 * m31);

		const auto subDet01 = (m10 * m22 * m33 - m10 * m23 * m32) -
			(m12 * m20 * m33 - m12 * m23 * m30) +
			(m13 * m20 * m32 - m13 * m22 * m30);

		const auto subDet02 = (m10 * m21 * m33 - m10 * m23 * m31) -
			(m11 * m20 * m33 - m11 * m23 * m30) +
			(m13 * m20 * m31 - m13 * m21 * m30);

		const auto subDet03 = (m10 * m21 * m32 - m10 * m22 * m31) -
			(m11 * m20 * m32 - m11 * m22 * m30) +
			(m12 * m20 * m31 - m12 * m21 * m30);

		return m00 * subDet00 - m01 * subDet01 + m02 * subDet02 - m03 * subDet03;
	}


	/**
	 * Computes the inverse of a matrix.
	 * Important: The matrix should already be in register memory, otherwise unnecessary global 
	 * memory calls are made.
	 */
	template<typename M>
	 auto invert2x2(const M& m) {
		using T = typename BaseType<M>::type;
		auto matrixDetInv = T(1) / det2x2(m);

		return makeTuple(
			m[I<3>()] * matrixDetInv,
			-m[I<1>()] * matrixDetInv,
			-m[I<2>()] * matrixDetInv,
			m[I<0>()] * matrixDetInv
		);
	}

	template<typename M>
	 auto invert3x3(const M& m) {
		using T = typename BaseType<M>::type;
		auto matrixDetInv = T(1) / det3x3(m);

		return makeTuple(
			matrixDetInv * (m[I<4>()] * m[I<8>()] - m[I<5>()] * m[I<7>()]),
			matrixDetInv * (-m[I<1>()] * m[I<8>()] + m[I<2>()] * m[I<7>()]),
			matrixDetInv * (m[I<1>()] * m[I<5>()] - m[I<2>()] * m[I<4>()]),
			matrixDetInv * (-m[I<3>()] * m[I<8>()] + m[I<5>()] * m[I<6>()]),
			matrixDetInv * (m[I<0>()] * m[I<8>()] - m[I<2>()] * m[I<6>()]),
			matrixDetInv * (-m[I<0>()] * m[I<5>()] + m[I<2>()] * m[I<3>()]),
			matrixDetInv * (m[I<3>()] * m[I<7>()] - m[I<4>()] * m[I<6>()]),
			matrixDetInv * (-m[I<0>()] * m[I<7>()] + m[I<1>()] * m[I<6>()]),
			matrixDetInv * (m[I<0>()] * m[I<4>()] - m[I<1>()] * m[I<3>()])
		);
	}

	template<typename M>
	 auto invert4x4(const M& m) {
		using T = typename BaseType<M>::type;
		auto matrixDetInv = T(1) / det4x4(m);

		return makeTuple(
			(m[I<5>()] * m[I<10>()] * m[I<15>()] -
			m[I<5>()] * m[I<11>()] * m[I<14>()] -
			m[I<9>()] * m[I<6>()] * m[I<15>()] +
			m[I<9>()] * m[I<7>()] * m[I<14>()] +
			m[I<13>()] * m[I<6>()] * m[I<11>()] -
			m[I<13>()] * m[I<7>()] * m[I<10>()]) * matrixDetInv,
			(-m[I<1>()] * m[I<10>()] * m[I<15>()] +
			m[I<1>()] * m[I<11>()] * m[I<14>()] +
			m[I<9>()] * m[I<2>()] * m[I<15>()] -
			m[I<9>()] * m[I<3>()] * m[I<14>()] -
			m[I<13>()] * m[I<2>()] * m[I<11>()] +
			m[I<13>()] * m[I<3>()] * m[I<10>()]) * matrixDetInv,
			(m[I<1>()] * m[I<6>()] * m[I<15>()] -
			m[I<1>()] * m[I<7>()] * m[I<14>()] -
			m[I<5>()] * m[I<2>()] * m[I<15>()] +
			m[I<5>()] * m[I<3>()] * m[I<14>()] +
			m[I<13>()] * m[I<2>()] * m[I<7>()] -
			m[I<13>()] * m[I<3>()] * m[I<6>()]) * matrixDetInv,
			(-m[I<1>()] * m[I<6>()] * m[I<11>()] +
			m[I<1>()] * m[I<7>()] * m[I<10>()] +
			m[I<5>()] * m[I<2>()] * m[I<11>()] -
			m[I<5>()] * m[I<3>()] * m[I<10>()] -
			m[I<9>()] * m[I<2>()] * m[I<7>()] +
			m[I<9>()] * m[I<3>()] * m[I<6>()]) * matrixDetInv,
			(-m[I<4>()] * m[I<10>()] * m[I<15>()] +
			m[I<4>()] * m[I<11>()] * m[I<14>()] +
			m[I<8>()] * m[I<6>()] * m[I<15>()] -
			m[I<8>()] * m[I<7>()] * m[I<14>()] -
			m[I<12>()] * m[I<6>()] * m[I<11>()] +
			m[I<12>()] * m[I<7>()] * m[I<10>()]) * matrixDetInv,
			(m[I<0>()] * m[I<10>()] * m[I<15>()] -
			m[I<0>()] * m[I<11>()] * m[I<14>()] -
			m[I<8>()] * m[I<2>()] * m[I<15>()] +
			m[I<8>()] * m[I<3>()] * m[I<14>()] +
			m[I<12>()] * m[I<2>()] * m[I<11>()] -
			m[I<12>()] * m[I<3>()] * m[I<10>()]) * matrixDetInv,
			(-m[I<0>()] * m[I<6>()] * m[I<15>()] +
			m[I<0>()] * m[I<7>()] * m[I<14>()] +
			m[I<4>()] * m[I<2>()] * m[I<15>()] -
			m[I<4>()] * m[I<3>()] * m[I<14>()] -
			m[I<12>()] * m[I<2>()] * m[I<7>()] +
			m[I<12>()] * m[I<3>()] * m[I<6>()]) * matrixDetInv,
			(m[I<0>()] * m[I<6>()] * m[I<11>()] -
			m[I<0>()] * m[I<7>()] * m[I<10>()] -
			m[I<4>()] * m[I<2>()] * m[I<11>()] +
			m[I<4>()] * m[I<3>()] * m[I<10>()] +
			m[I<8>()] * m[I<2>()] * m[I<7>()] -
			m[I<8>()] * m[I<3>()] * m[I<6>()]) * matrixDetInv,
			(m[I<4>()] * m[I<9>()] * m[I<15>()] -
			m[I<4>()] * m[I<11>()] * m[I<13>()] -
			m[I<8>()] * m[I<5>()] * m[I<15>()] +
			m[I<8>()] * m[I<7>()] * m[I<13>()] +
			m[I<12>()] * m[I<5>()] * m[I<11>()] -
			m[I<12>()] * m[I<7>()] * m[I<9>()]) * matrixDetInv,
			(-m[I<0>()] * m[I<9>()] * m[I<15>()] +
			m[I<0>()] * m[I<11>()] * m[I<13>()] +
			m[I<8>()] * m[I<1>()] * m[I<15>()] -
			m[I<8>()] * m[I<3>()] * m[I<13>()] -
			m[I<12>()] * m[I<1>()] * m[I<11>()] +
			m[I<12>()] * m[I<3>()] * m[I<9>()]) * matrixDetInv,
			(m[I<0>()] * m[I<5>()] * m[I<15>()] -
			m[I<0>()] * m[I<7>()] * m[I<13>()] -
			m[I<4>()] * m[I<1>()] * m[I<15>()] +
			m[I<4>()] * m[I<3>()] * m[I<13>()] +
			m[I<12>()] * m[I<1>()] * m[I<7>()] -
			m[I<12>()] * m[I<3>()] * m[I<5>()]) * matrixDetInv,
			(-m[I<0>()] * m[I<5>()] * m[I<11>()] +
			m[I<0>()] * m[I<7>()] * m[I<9>()] +
			m[I<4>()] * m[I<1>()] * m[I<11>()] -
			m[I<4>()] * m[I<3>()] * m[I<9>()] -
			m[I<8>()] * m[I<1>()] * m[I<7>()] +
			m[I<8>()] * m[I<3>()] * m[I<5>()]) * matrixDetInv,
			(-m[I<4>()] * m[I<9>()] * m[I<14>()] +
			m[I<4>()] * m[I<10>()] * m[I<13>()] +
			m[I<8>()] * m[I<5>()] * m[I<14>()] -
			m[I<8>()] * m[I<6>()] * m[I<13>()] -
			m[I<12>()] * m[I<5>()] * m[I<10>()] +
			m[I<12>()] * m[I<6>()] * m[I<9>()]) * matrixDetInv,
			(m[I<0>()] * m[I<9>()] * m[I<14>()] -
			m[I<0>()] * m[I<10>()] * m[I<13>()] -
			m[I<8>()] * m[I<1>()] * m[I<14>()] +
			m[I<8>()] * m[I<2>()] * m[I<13>()] +
			m[I<12>()] * m[I<1>()] * m[I<10>()] -
			m[I<12>()] * m[I<2>()] * m[I<9>()]) * matrixDetInv,
			(-m[I<0>()] * m[I<5>()] * m[I<14>()] +
			m[I<0>()] * m[I<6>()] * m[I<13>()] +
			m[I<4>()] * m[I<1>()] * m[I<14>()] -
			m[I<4>()] * m[I<2>()] * m[I<13>()] -
			m[I<12>()] * m[I<1>()] * m[I<6>()] +
			m[I<12>()] * m[I<2>()] * m[I<5>()]) * matrixDetInv,
			(m[I<0>()] * m[I<5>()] * m[I<10>()] -
			m[I<0>()] * m[I<6>()] * m[I<9>()] -
			m[I<4>()] * m[I<1>()] * m[I<10>()] +
			m[I<4>()] * m[I<2>()] * m[I<9>()] +
			m[I<8>()] * m[I<1>()] * m[I<6>()] -
			m[I<8>()] * m[I<2>()] * m[I<5>()]) * matrixDetInv
		);
	}


	/**
	 * Vector/matrix assignment methods.
	 */
	template<typename In, typename Out>
	 void assign2(const In& input, Out& output) {
		output[I<0>()] = input[I<0>()];
		output[I<1>()] = input[I<1>()];
	}

	template<typename In, typename Out>
	 void assign3(const In& input, Out& output) {
		output[I<0>()] = input[I<0>()];
		output[I<1>()] = input[I<1>()];
		output[I<2>()] = input[I<2>()];
	}

	template<typename In, typename Out>
	 void assign4(const In& input, Out& output) {
		output[I<0>()] = input[I<0>()];
		output[I<1>()] = input[I<1>()];
		output[I<2>()] = input[I<2>()];
		output[I<3>()] = input[I<3>()];
	}

	template<typename In, typename Out>
	 void assign2x2(const In& input, Out& output) {
		output[I<0>()] = input[I<0>()];
		output[I<1>()] = input[I<1>()];
		output[I<2>()] = input[I<2>()];
		output[I<3>()] = input[I<3>()];
	}

	template<typename In, typename Out>
	 void assign3x3(const In& input, Out& output) {
		output[I<0>()] = input[I<0>()];
		output[I<1>()] = input[I<1>()];
		output[I<2>()] = input[I<2>()];
		output[I<3>()] = input[I<3>()];
		output[I<4>()] = input[I<4>()];
		output[I<5>()] = input[I<5>()];
		output[I<6>()] = input[I<6>()];
		output[I<7>()] = input[I<7>()];
		output[I<8>()] = input[I<8>()];
	}

	template<typename In, typename Out>
	 void assign4x4(const In& input, Out& output) {
		output[I<0>()]  = input[I<0>()];
		output[I<1>()]  = input[I<1>()];
		output[I<2>()]  = input[I<2>()];
		output[I<3>()]  = input[I<3>()];
		output[I<4>()]  = input[I<4>()];
		output[I<5>()]  = input[I<5>()];
		output[I<6>()]  = input[I<6>()];
		output[I<7>()]  = input[I<7>()];
		output[I<8>()]  = input[I<8>()];
		output[I<9>()]  = input[I<9>()];
		output[I<10>()] = input[I<10>()];
		output[I<11>()] = input[I<11>()];
		output[I<12>()] = input[I<12>()];
		output[I<13>()] = input[I<13>()];
		output[I<14>()] = input[I<14>()];
		output[I<15>()] = input[I<15>()];
	}

} // namespace matrix_lib