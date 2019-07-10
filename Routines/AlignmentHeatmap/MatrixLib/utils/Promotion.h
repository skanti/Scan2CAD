#pragma once

namespace matrix_lib {
	
	template<typename T1, typename T2>
	struct Promotion {
		typedef decltype(T1() + T2()) Type;
	};

	template<typename T1, typename T2>
	using Promote = typename Promotion<T1, T2>::Type;

} // namespace matrix_lib