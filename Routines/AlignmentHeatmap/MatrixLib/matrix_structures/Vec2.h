#pragma once
#include "utils/Math.h"
#include "utils/VectorMath.h"

namespace matrix_lib {

	/**
	 * 2D vector.
	 */
	template <class T>
	class Vec2 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		 explicit Vec2(T v) {
			m_array[0] = m_array[1] = v;
		}

		// Default constructor (we only set values to zero if we are dealing
		// with fundamental types, otherwise we just use default constructor of
		// a compound type).
		 void reset(std::true_type) {
			m_array[0] = m_array[1] = T(0);
		}

		 void reset(std::false_type) { }

		 Vec2() {
			reset(std::is_fundamental<T>::type());
		}

		 Vec2(T x, T y) {
			m_array[0] = x;
			m_array[1] = y;
		}

		// Initialize with tuple.
		 Vec2(const Tuple<typename TL<T, T>::type>& t) {
			m_array[0] = t[I<0>()];
			m_array[1] = t[I<1>()];
		}

		// Initialize with float4.
		 Vec2(const float4& v) {
			m_array[0] = v.x;
			m_array[1] = v.y;
		}

		// Initialize with raw array.
		 Vec2(const T* other) {
			m_array[0] = other[0];
			m_array[1] = other[1];
		}

		// Copy data from other template type.
		template <class U>
		 Vec2(const Vec2<U>& other) {
			m_array[0] = T(other[0]);
			m_array[1] = T(other[1]);
		}

		// Copy constructor.
		 Vec2(const Vec2& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
		}

		// Move constructor.
		 Vec2(Vec2&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
		}

		// Copy assignment.
		 Vec2<T>& operator=(const Vec2& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			return *this;
		}

		// Move assignment.
		 Vec2<T>& operator=(Vec2&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			return *this;
		}

		 Vec2<T>& operator=(T other) {
			m_array[0] = other;
			m_array[1] = other;
			return *this;
		}

		// Destructor.
		 ~Vec2() = default;

		/**
		 * Basic operations.
		 */
		 Vec2<T> operator-() const {
			return Vec2<T>(-m_array[0], -m_array[1]);
		}

		 Vec2<T> operator+(const Vec2& other) const {
			return Vec2<T>(m_array[0] + other.m_array[0], m_array[1] + other.m_array[1]);
		}

		 Vec2<T> operator+(T val) const {
			return Vec2<T>(m_array[0] + val, m_array[1] + val);
		}

		 void operator+=(const Vec2& other) {
			m_array[0] += other.m_array[0];
			m_array[1] += other.m_array[1];
		}

		 void operator-=(const Vec2& other) {
			m_array[0] -= other.m_array[0];
			m_array[1] -= other.m_array[1];
		}

		 void operator+=(T val) {
			m_array[0] += val;
			m_array[1] += val;
		}

		 void operator-=(T val) {
			m_array[0] -= val;
			m_array[1] -= val;
		}

		 void operator*=(T val) {
			m_array[0] *= val;
			m_array[1] *= val;
		}

		 void operator/=(T val) {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				m_array[0] *= inv;
				m_array[1] *= inv;
			}
			else {
				m_array[0] /= val;
				m_array[1] /= val;
			}
		}

		 Vec2<T> operator*(T val) const {
			return Vec2<T>(m_array[0] * val, m_array[1] * val);
		}

		 Vec2<T> operator/(T val) const {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				return Vec2<T>(m_array[0] * inv, m_array[1] * inv);
			}
			else {
				return Vec2<T>(m_array[0] / val, m_array[1] / val);
			}
		}

		 Vec2<T> operator-(const Vec2& other) const {
			return Vec2<T>(m_array[0] - other.m_array[0], m_array[1] - other.m_array[1]);
		}

		 Vec2<T> operator-(T val) const {
			return Vec2<T>(m_array[0] - val, m_array[1] - val);
		}

		 bool operator==(const Vec2& other) const {
			if ((m_array[0] == other.m_array[0]) && (m_array[1] == other.m_array[1]))
				return true;

			return false;
		}

		 bool operator!=(const Vec2& other) const {
			return !(*this == other);
		}

		// Dot product
		 T operator|(const Vec2& other) const {
			return (m_array[0] * other.m_array[0] + m_array[1] * other.m_array[1]);
		}

		 static inline T dot(const Vec2& l, const Vec2& r) {
			return(l.m_array[0] * r.m_array[0] + l.m_array[1] * r.m_array[1]);
		}

		 bool operator<(const Vec2& other) const {
			if ((x < other.x()) && (y < other.y()))
				return true;

			return false;
		}

		 bool isValid() const {
			return (m_array[0] == m_array[0] && m_array[1] == m_array[1]);
		}

		 bool isFinite() const {
			return std::isfinite(m_array[0]) && std::isfinite(m_array[1]);
		}

		 T lengthSq() const {
			return (m_array[0] * m_array[0] + m_array[1] * m_array[1]);
		}

		 T length() const {
			return sqrt(m_array[0] * m_array[0] + m_array[1] * m_array[1]);
		}

		 static T distSq(const Vec2& v0, const Vec2& v1) {
			return ((v0.m_array[0] - v1.m_array[0])*(v0.m_array[0] - v1.m_array[0]) + (v0.m_array[1] - v1.m_array[1])*(v0.m_array[1] - v1.m_array[1]));
		}

		 static T dist(const Vec2& v0, const Vec2& v1) {
			return sqrt((v0.m_array[0] - v1.m_array[0])*(v0.m_array[0] - v1.m_array[0]) + (v0.m_array[1] - v1.m_array[1])*(v0.m_array[1] - v1.m_array[1]));
		}

		 static inline Vec2<T> normalize(const Vec2<T>& v) {
			return v.getNormalized();
		}

		 void normalize() {
			T val = (T)1.0 / length();
			m_array[0] *= val;
			m_array[1] *= val;
		}

		 Vec2<T> getNormalized() const {
			T val = (T)1.0 / length();
			return Vec2<T>(m_array[0] * val, m_array[1] * val);
		}

		/**
		 * Indexing operators.
		 */
		 T& operator[](unsigned int i) {
			runtime_assert(i < 2, "Index out of bounds.");
			return m_array[i];
		}

		 const T& operator[](unsigned int i) const {
			runtime_assert(i < 2, "Index out of bounds.");
			return m_array[i];
		}

		template<unsigned i>
		 T& operator[](I<i>) {
			static_assert(i < 2, "Index out of bounds.");
			return m_array[i];
		}

		template<unsigned i>
		 const T& operator[](I<i>) const {
			static_assert(i < 2, "Index out of bounds.");
			return m_array[i];
		}

		/**
		 * Getters/setters.
		 */
		 const T& x() const { return m_array[0]; }
		 T& x() { return m_array[0]; }
		 const T& y() const { return m_array[1]; }
		 T& y() { return m_array[1]; }

		 void print() const {
			std::cout << "(" << m_array[0] << " / " << m_array[1] << ")" << std::endl;
		}

		 T* getData() {
			return &m_array[0];
		}

		 const T* getData() const {
			return &m_array[0];
		}

		 inline std::string toString(char separator = ' ') const {
			return toString(std::string(1, separator));
		}

		 inline std::string toString(const std::string &separator) const {
			return std::to_string(x()) + separator + std::to_string(y());
		}
	private:
		T m_array[2];
	};

	/**
	 * Math operators.
	 */
	template <class T>
	 Vec2<T> operator*(T s, const Vec2<T>& v) {
		return v * s;
	}

	template <class T>
	 Vec2<T> operator/(T s, const Vec2<T>& v) {
		return Vec2<T>(s / v.x(), s / v.y());
	}

	template <class T>
	 Vec2<T> operator+(T s, const Vec2<T>& v) {
		return v + s;
	}

	template <class T>
	 Vec2<T> operator-(T s, const Vec2<T>& v) {
		return Vec2<T>(s - v.x(), s - v.y());
	}

	// Write a vec2 to a stream.
	template <class T>
	 std::ostream& operator<<(std::ostream& s, const Vec2<T>& v) {
		return (s << v[0] << " " << v[1]);
	}

	// Read a vec2 from a stream.
	template <class T>
	 std::istream& operator>>(std::istream& s, Vec2<T>& v) {
		return (s >> v[0] >> v[1]);
	}

	/**
	 * Comparison operators.
	 */
	template<class T>  bool operator==(const Vec2<T>& lhs, const Vec2<T>& rhs) { return lhs[0] == rhs[0] && lhs[1] == rhs[1]; }
	template<class T>  bool operator!=(const Vec2<T>& lhs, const Vec2<T>& rhs) { return !operator==(lhs, rhs); }
	template<class T>  bool operator< (const Vec2<T>& lhs, const Vec2<T>& rhs) { return std::tie(lhs[0], lhs[1]) < std::tie(rhs[0], rhs[1]); }
	template<class T>  bool operator> (const Vec2<T>& lhs, const Vec2<T>& rhs) { return  operator< (rhs, lhs); }
	template<class T>  bool operator<=(const Vec2<T>& lhs, const Vec2<T>& rhs) { return !operator> (lhs, rhs); }
	template<class T>  bool operator>=(const Vec2<T>& lhs, const Vec2<T>& rhs) { return !operator< (lhs, rhs); }

	typedef Vec2<double> Vec2d;
	typedef Vec2<float> Vec2f;
	typedef Vec2<int> Vec2i;
	typedef Vec2<short> Vec2s;
	typedef Vec2<short> Vec2us;
	typedef Vec2<unsigned int> Vec2ui;
	typedef Vec2<unsigned char> Vec2uc;
	typedef Vec2<unsigned long long> Vec2ul;
	typedef Vec2<long long> Vec2l;

	/**
	 * Math operations.
	 */
	namespace math_proc {
		template<class T>
		 Vec2i round(const Vec2<T>& f) {
			return Vec2i(round(f.x()), round(f.y()));
		}
		template<class T>
		 Vec2i ceil(const Vec2<T>& f) {
			return Vec2i(ceil(f.x()), ceil(f.y()));
		}
		template<class T>
		 Vec2i floor(const Vec2<T>& f) {
			return Vec2i(floor(f.x()), floor(f.y()));
		}
		template<class T>
		 Vec2<T> abs(const Vec2<T>& p) {
			return Vec2<T>(abs(p.x()), abs(p.y()));
		}
		template<class T>
		 Vec2<T> sqrt(const Vec2<T>& p) {
			return Vec2<T>(sqrt(p.x()), sqrt(p.y()));
		}
		template<class T>
		 Vec2<T> max(const Vec2<T>& p, T v) {
			return Vec2<T>(max(p.x(), v),
				max(p.y(), v));
		}
		template<class T>
		 Vec2<T> max(const Vec2<T>& p, const Vec2<T>& v) {
			return Vec2<T>(
				max(p.x(), v.x()),
				max(p.y(), v.y()));
		}
		template<class T>
		 Vec2<T> min(const Vec2<T>& p, T v) {
			return Vec2<T>(min(p.x(), v),
				min(p.y(), v));
		}
		template<class T>
		 Vec2<T> min(const Vec2<T>& p, const Vec2<T>& v) {
			return Vec2<T>(
				min(p.x(), v.x()),
				min(p.y(), v.y()));
		}
	} // namespace math_proc

} // namespace matrix_lib
