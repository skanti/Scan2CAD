#pragma once

namespace matrix_lib {
	namespace math_proc {
		
		static const double PI = 3.1415926535897932384626433832795028842;
		static const float PIf = 3.14159265358979323846f;

		 inline float degreesToRadians(float x) {
			return x * (PIf / 180.0f);
		}

		 inline float radiansToDegrees(float x) {
			return x * (180.0f / PIf);
		}

		 inline double degreesToRadians(double x) {
			return x * (PI / 180.0);
		}

		 inline double radiansToDegrees(double x) {
			return x * (180.0 / PI);
		}

		template<class T>
		 long int floor(T x) {
			return (long int)std::floor(x);
		}

		template<class T>
		 long int ceil(T x) {
			return (long int)std::ceil(x);
		}

		using std::round;
		using std::abs;
		using std::sqrt;
		using std::max;
		using std::min;

	} // namespace math_proc
} // namespace matrix_lib