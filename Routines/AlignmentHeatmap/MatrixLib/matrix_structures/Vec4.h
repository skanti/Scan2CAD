#pragma once

namespace matrix_lib {

	/**
	 * 4D vector.
	 */
	template <class T>
	class Vec4 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		 explicit Vec4(T v) {
			m_array[0] = m_array[1] = m_array[2] = m_array[3] = v;
		}

		// Default constructor (we only set values to zero if we are dealing
		// with fundamental types, otherwise we just use default constructor of
		// a compound type).
		 void reset(std::true_type) {
			m_array[0] = m_array[1] = m_array[2] = m_array[3] = T(0);
		}

		 void reset(std::false_type) { }

		 Vec4() {
			reset(std::is_fundamental<T>::type());
		}

		 Vec4(T x, T y, T z, T w) {
			m_array[0] = x;
			m_array[1] = y;
			m_array[2] = z;
			m_array[3] = w;
		}

		// Initialize with raw array.
		 Vec4(const T* other) {
			m_array[0] = other[0];
			m_array[1] = other[1];
			m_array[2] = other[2];
			m_array[3] = other[3];
		}

		// Copy data from other template type.
		template <class U>
		 Vec4(const Vec4<U>& other) {
			m_array[0] = T(other[0]);
			m_array[1] = T(other[1]);
			m_array[2] = T(other[2]);
			m_array[3] = T(other[3]);
		}

		// Copy constructor.
		 Vec4(const Vec4& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			m_array[2] = other.m_array[2];
			m_array[3] = other.m_array[3];
		}

		// Move constructor.
		 Vec4(Vec4&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			m_array[2] = std::move(other.m_array[2]);
			m_array[3] = std::move(other.m_array[3]);
		}

		// Copy assignment.
		 Vec4<T>& operator=(const Vec4& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			m_array[2] = other.m_array[2];
			m_array[3] = other.m_array[3];
			return *this;
		}

		// Move assignment.
		 Vec4<T>& operator=(Vec4&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			m_array[2] = std::move(other.m_array[2]);
			m_array[3] = std::move(other.m_array[3]);
			return *this;
		}

		// Destructor.
		 ~Vec4() = default;

		/**
		 * Basic operations.
		 */
		 Vec4<T> operator-() const {
			return Vec4<T>(-m_array[0], -m_array[1], -m_array[2], -m_array[3]);
		}

		 Vec4<T> operator+(const Vec4& other) const {
			return Vec4<T>(m_array[0] + other.m_array[0], m_array[1] + other.m_array[1],
			               m_array[2] + other.m_array[2], m_array[3] + other.m_array[3]);
		}

		 Vec4<T> operator+(T val) const {
			return Vec4<T>(m_array[0] + val, m_array[1] + val, m_array[2] + val, m_array[3] + val);
		}

		 void operator+=(const Vec4& other) {
			m_array[0] += other.m_array[0];
			m_array[1] += other.m_array[1];
			m_array[2] += other.m_array[2];
			m_array[3] += other.m_array[3];
		}

		 void operator-=(const Vec4& other) {
			m_array[0] -= other.m_array[0];
			m_array[1] -= other.m_array[1];
			m_array[2] -= other.m_array[2];
			m_array[3] -= other.m_array[3];
		}

		 void operator+=(T val) {
			m_array[0] += val;
			m_array[1] += val;
			m_array[2] += val;
			m_array[3] += val;
		}

		 void operator-=(T val) {
			m_array[0] -= val;
			m_array[1] -= val;
			m_array[2] -= val;
			m_array[3] -= val;
		}

		 void operator*=(T val) {
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
			m_array[3] *= val;
		}

		 void operator/=(T val) {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				m_array[0] *= inv;
				m_array[1] *= inv;
				m_array[2] *= inv;
				m_array[3] *= inv;
			}
			else {
				m_array[0] /= val;
				m_array[1] /= val;
				m_array[2] /= val;
				m_array[3] /= val;
			}
		}

		 Vec4<T> operator*(T val) const {
			return Vec4<T>(m_array[0] * val, m_array[1] * val, m_array[2] * val, m_array[3] * val);
		}

		 Vec4<T> operator/(T val) const {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				return Vec4<T>(m_array[0] * inv, m_array[1] * inv, m_array[2] * inv, m_array[3] * inv);
			}
			else {
				return Vec4<T>(m_array[0] / val, m_array[1] / val, m_array[2] / val, m_array[3] / val);
			}
		}

		// Cross product (of .xyz).
		 Vec4<T> operator^(const Vec4& other) const {
			return Vec4<T>(m_array[1] * other.m_array[2] - m_array[2] * other.m_array[1],
			               m_array[2] * other.m_array[0] - m_array[0] * other.m_array[2],
			               m_array[0] * other.m_array[1] - m_array[1] * other.m_array[0], T(1));
		}

		// Dot product.
		 T operator|(const Vec4& other) const {
			return (m_array[0] * other.m_array[0] + m_array[1] * other.m_array[1] + m_array[2] *
				other.m_array[2] + m_array[3] * other.m_array[3]);
		}

		 Vec4<T> operator-(const Vec4& other) const {
			return Vec4<T>(m_array[0] - other.m_array[0], m_array[1] - other.m_array[1], m_array[2] - other.m_array[2],
			               m_array[3] - other.m_array[3]);
		}

		 Vec4<T> operator-(T val) const {
			return Vec4<T>(m_array[0] - val, m_array[1] - val, m_array[2] - val, m_array[3] - val);
		}

		 bool operator==(const Vec4& other) const {
			if ((m_array[0] == other.m_array[0]) && (m_array[1] == other.m_array[1]) &&
				(m_array[2] == other.m_array[2]) && (m_array[3] == other.m_array[3])) {
				return true;
			}

			return false;
		}

		 bool operator!=(const Vec4& other) const {
			return !(*this == other);
		}

		 bool isValid() const {
			return (m_array[0] == m_array[0] && m_array[1] == m_array[1] && m_array[2] == m_array[2] && m_array[3] == m_array[3]);
		}

		 bool isFinite() const {
			return std::isfinite(m_array[0]) && std::isfinite(m_array[1]) && std::isfinite(m_array[2]) && std::isfinite(m_array[3]);
		}

		 T lengthSq() const {
			return (m_array[0] * m_array[0] + m_array[1] * m_array[1] + m_array[2] * m_array[2] + m_array[3] * m_array[3]);
		}

		 T length() const {
			return sqrt(lengthSq());
		}

		 static inline Vec4<T> normalize(const Vec4<T>& v) {
			return v.getNormalized();
		}

		 static T distSq(const Vec4& v0, const Vec4& v1) {
			return (
				(v0.m_array[0] - v1.m_array[0]) * (v0.m_array[0] - v1.m_array[0]) +
				(v0.m_array[1] - v1.m_array[1]) * (v0.m_array[1] - v1.m_array[1]) +
				(v0.m_array[2] - v1.m_array[2]) * (v0.m_array[2] - v1.m_array[2]) +
				(v0.m_array[3] - v1.m_array[3]) * (v0.m_array[3] - v1.m_array[3])
			);
		}

		 static T dist(const Vec4& v0, const Vec4& v1) {
			return (v0 - v1).length();
		}

		 void normalize() {
			T val = (T)1.0 / length();
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
			m_array[3] *= val;
		}

		 Vec4<T> getNormalized() const {
			T val = (T)1.0 / length();
			return Vec4<T>(m_array[0] * val, m_array[1] * val, m_array[2] * val,
				m_array[3] * val);
		}

		 void dehomogenize() {
			m_array[0] /= m_array[3];
			m_array[1] /= m_array[3];
			m_array[2] /= m_array[3];
			m_array[3] /= m_array[3];
		}


		 bool isLinearDependent(const Vec4& other) const {
			T factor = x() / other.x();

			if ((std::fabs(x() / factor - other.x()) + std::fabs(y() / factor - other.y()) +
				std::fabs(z() / factor - other.z()) + std::fabs(w() / factor - other.w())) < 0.00001) {
				return true;
			}
			else {
				return false;
			}
		}

		 T* getData() {
			return &m_array[0];
		}

		 const T* getData() const {
			return &m_array[0];
		}

		/**
		 * Indexing operators.
		 */
		 const T& operator[](int i) const {
			return m_array[i];
		}

		 T& operator[](int i) {
			return m_array[i];
		}

		/**
		 * Getters/setters.
		 */
		 const T& x() const { return m_array[0]; }
		 T& x() { return m_array[0]; }
		 const T& y() const { return m_array[1]; }
		 T& y() { return m_array[1]; }
		 const T& z() const { return m_array[2]; }
		 T& z() { return m_array[2]; }
		 const T& w() const { return m_array[3]; }
		 T& w() { return m_array[3]; }

		 std::string toString(char separator = ' ') const {
			return toString(std::string(1, separator));
		}

		 std::string toString(const std::string& separator) const {
			return std::to_string(x()) + separator + std::to_string(y()) + separator + std::to_string(z()) + separator + std::
				to_string(w());
		}

		 void print() const {
			std::cout << "(" << m_array[0] << " / " << m_array[1] << " / " << m_array[2] <<
				" / " << m_array[3] << " ) " << std::endl;
		}

	private:
		T m_array[4];
	};

	/**
	 * Math operators.
	 */
	template <class T>
	 Vec4<T> operator*(T s, const Vec4<T>& v) {
		return v * s;
	}

	template <class T>
	 Vec4<T> operator/(T s, const Vec4<T>& v) {
		return Vec4<T>(s / v.x(), s / v.y(), s / v.z(), s / v.w());
	}

	template <class T>
	 Vec4<T> operator+(T s, const Vec4<T>& v) {
		return v + s;
	}

	template <class T>
	 Vec4<T> operator-(T s, const Vec4<T>& v) {
		return Vec4<T>(s - v.x(), s - v.y(), s - v.z(), s - v.w());
	}

	// Write a Vec4 to a stream.
	template <class T>
	 std::ostream& operator<<(std::ostream& s, const Vec4<T>& v) {
		return (s << v[0] << " " << v[1] << " " << v[2] << " " << v[3]);
	}

	// Read a Vec4 from a stream.
	template <class T>
	 std::istream& operator>>(std::istream& s, Vec4<T>& v) {
		return (s >> v[0] >> v[1] >> v[2] >> v[3]);
	}

	/**
	 * Comparison operators.
	 */
	template<class T>  bool operator==(const Vec4<T>& lhs, const Vec4<T>& rhs) { return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3]; }
	template<class T>  bool operator!=(const Vec4<T>& lhs, const Vec4<T>& rhs) { return !operator==(lhs, rhs); }

	typedef Vec4<double> Vec4d;
	typedef Vec4<float> Vec4f;
	typedef Vec4<int> Vec4i;
	typedef Vec4<short> Vec4s;
	typedef Vec4<short> Vec4us;
	typedef Vec4<unsigned int> Vec4ui;
	typedef Vec4<unsigned char> Vec4uc;
	typedef Vec4<unsigned long long> Vec4ul;
	typedef Vec4<long long> Vec4l;

	/**
	 * Math operations.
	 */
	namespace math_proc {
		template<class T>
		 Vec4i round(const Vec4<T>& f) {
			return Vec4i(round(f.x()), round(f.y()), round(f.z()), round(f.w()));
		}
		template<class T>
		 Vec4i ceil(const Vec4<T>& f) {
			return Vec4i(ceil(f.x()), ceil(f.y()), ceil(f.z()), ceil(f.w()));
		}
		template<class T>
		 Vec4i floor(const Vec4<T>& f) {
			return Vec4i(floor(f.x()), floor(f.y()), floor(f.z()), floor(f.w()));
		}
		template<class T>
		 Vec4<T> abs(const Vec4<T>& p) {
			return Vec4<T>(abs(p.x()), abs(p.y()), abs(p.z()), abs(p.w()));
		}
		template<class T>
		 Vec4<T> sqrt(const Vec4<T>& p) {
			return Vec4<T>(sqrt(p.x()), sqrt(p.y()), sqrt(p.z()), sqrt(p.w()));
		}
		template<class T>
		 Vec4<T> max(const Vec4<T>& p, T v) {
			return Vec4<T>(
				max(p.x(), v),
				max(p.y(), v),
				max(p.z(), v),
				max(p.w(), v));
		}
		template<class T>
		 Vec4<T> max(const Vec4<T>& p, const Vec4<T>& v) {
			return Vec4<T>(
				max(p.x(), v.x()),
				max(p.y(), v.y()),
				max(p.z(), v.z()),
				max(p.w(), v.w()));
		}
		template<class T>
		 Vec4<T> min(const Vec4<T>& p, T v) {
			return Vec4<T>(
				min(p.x(), v),
				min(p.y(), v),
				min(p.z(), v),
				min(p.w(), v));
		}
		template<class T>
		 Vec4<T> min(const Vec4<T>& p, const Vec4<T>& v) {
			return Vec4<T>(
				min(p.x(), v.x()),
				min(p.y(), v.y()),
				min(p.z(), v.z()),
				min(p.w(), v.w()));
		}
	} // namespace math_proc

} // namespace matrix_lib
