#pragma once

namespace matrix_lib {

	/**
	 * 3D vector.
	 */
	template <class T>
	class Vec3 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		 explicit Vec3(T v) {
			m_array[0] = m_array[1] = m_array[2] = v;
		}

		// Default constructor (we only set values to zero if we are dealing
		// with fundamental types, otherwise we just use default constructor of
		// a compound type).
		 void reset(std::true_type) {
			m_array[0] = m_array[1] = m_array[2] = T(0);
		}

		 void reset(std::false_type) { }

		 Vec3() {
			reset(std::is_fundamental<T>::type());
		}

		// Initialize with values stored in an array.
		 Vec3(const T* values) {
			for (unsigned int i = 0; i < 3; i++) m_array[i] = values[i];
		}

		 Vec3(T x, T y, T z) {
			m_array[0] = x;
			m_array[1] = y;
			m_array[2] = z;
		}


		// Copy data from other template type.
		template <class U>
		 Vec3(const Vec3<U>& other) {
			m_array[0] = T(other[0]);
			m_array[1] = T(other[1]);
			m_array[2] = T(other[2]);
		}


		// Copy constructor.
		 Vec3(const Vec3& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			m_array[2] = other.m_array[2];
		}

		// Move constructor.
		 Vec3(Vec3&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			m_array[2] = std::move(other.m_array[2]);
		}

		// Copy assignment.
		 Vec3<T>& operator=(const Vec3& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			m_array[2] = other.m_array[2];
			return *this;
		}

		// Move assignment.
		 Vec3<T>& operator=(Vec3&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			m_array[2] = std::move(other.m_array[2]);
			return *this;
		}

		 Vec3<T>& operator=(T other) {
			m_array[0] = other;
			m_array[1] = other;
			m_array[2] = other;
			return *this;
		}

		// Destructor.
		 ~Vec3() = default;

		/**
		 * Basic operations.
		 */
		 Vec3<T> operator-() const {
			return Vec3<T>(-m_array[0], -m_array[1], -m_array[2]);
		}

		 Vec3<T> operator+(const Vec3& other) const {
			return Vec3<T>(m_array[0] + other.m_array[0], m_array[1] + other.m_array[1], m_array[2] + other.m_array[2]);
		}

		 Vec3<T> operator+(T val) const {
			return Vec3<T>(m_array[0] + val, m_array[1] + val, m_array[2] + val);
		}

		 void operator+=(const Vec3& other) {
			m_array[0] += other.m_array[0];
			m_array[1] += other.m_array[1];
			m_array[2] += other.m_array[2];
		}

		 void operator-=(const Vec3& other) {
			m_array[0] -= other.m_array[0];
			m_array[1] -= other.m_array[1];
			m_array[2] -= other.m_array[2];
		}

		 void operator+=(T val) {
			m_array[0] += val;
			m_array[1] += val;
			m_array[2] += val;
		}

		 void operator-=(T val) {
			m_array[0] -= val;
			m_array[1] -= val;
			m_array[2] -= val;
		}

		 void operator*=(T val) {
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
		}

		 void operator/=(T val) {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = T(1) / val;
				m_array[0] *= inv;
				m_array[1] *= inv;
				m_array[2] *= inv;
			}
			else {
				m_array[0] /= val;
				m_array[1] /= val;
				m_array[2] /= val;
			}
		}

		 Vec3<T> operator*(T val) const {
			return Vec3<T>(m_array[0] * val, m_array[1] * val, m_array[2] * val);
		}

		 Vec3<T> operator/(T val) const {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				return Vec3<T>(m_array[0] * inv, m_array[1] * inv, m_array[2] * inv);
			}
			else {
				return Vec3<T>(m_array[0] / val, m_array[1] / val, m_array[2] / val);
			}
		}

		// Cross product
		 Vec3<T> operator^(const Vec3& other) const {
			return Vec3<T>(m_array[1] * other.m_array[2] - m_array[2] * other.m_array[1],
			               m_array[2] * other.m_array[0] - m_array[0] * other.m_array[2],
			               m_array[0] * other.m_array[1] - m_array[1] * other.m_array[0]);
		}

		// Dot product
		 T operator|(const Vec3& other) const {
			return (m_array[0] * other.m_array[0] + m_array[1] * other.m_array[1] + m_array[2] * other.m_array[2]);
		}

		 static inline T dot(const Vec3& l, const Vec3& r) {
			return (l.m_array[0] * r.m_array[0] + l.m_array[1] * r.m_array[1] + l.m_array[2] * r.m_array[2]);
		}

		 static inline Vec3 cross(const Vec3& l, const Vec3& r) {
			return (l ^ r);
		}

		 Vec3<T> operator-(const Vec3& other) const {
			return Vec3<T>(m_array[0] - other.m_array[0], m_array[1] - other.m_array[1], m_array[2] - other.m_array[2]);
		}

		 Vec3<T> operator-(T val) const {
			return Vec3<T>(m_array[0] - val, m_array[1] - val, m_array[2] - val);
		}

		 bool operator==(const Vec3& other) const {
			if ((m_array[0] == other.m_array[0]) && (m_array[1] == other.m_array[1]) && (m_array[2] == other.m_array[2]))
				return true;

			return false;
		}

		 bool operator!=(const Vec3& other) const {
			return !(*this == other);
		}

		 bool isValid() const {
			return (m_array[0] == m_array[0] && m_array[1] == m_array[1] && m_array[2] == m_array[2]);
		}

		 bool isFinite() const {
			return std::isfinite(m_array[0]) && std::isfinite(m_array[1]) && std::isfinite(m_array[2]);
		}

		 T lengthSq() const {
			return (m_array[0] * m_array[0] + m_array[1] * m_array[1] + m_array[2] * m_array[2]);
		}

		 T length() const {
			return std::sqrt(lengthSq());
		}

		 static T distSq(const Vec3& v0, const Vec3& v1) {
			return ((v0.m_array[0] - v1.m_array[0]) * (v0.m_array[0] - v1.m_array[0]) + (v0.m_array[1] - v1.m_array[1]) * (v0.m_array[1] - v1.
				m_array[1]) + (v0.m_array[2] - v1.m_array[2]) * (v0.m_array[2] - v1.m_array[2]));
		}

		 static T dist(const Vec3& v0, const Vec3& v1) {
			return std::sqrt(
				(v0.m_array[0] - v1.m_array[0]) * (v0.m_array[0] - v1.m_array[0]) + (v0.m_array[1] - v1.m_array[1]) * (v0.m_array[1] - v1.m_array[1])
				+ (v0.m_array[2] - v1.m_array[2]) * (v0.m_array[2] - v1.m_array[2])
			);
		}

		 operator T*() {
			return m_array;
		}

		 operator const T*() const {
			return m_array;
		}

		 void print() const {
			std::cout << "(" << m_array[0] << " / " << m_array[1] << " / " << m_array[2] << ")" << std::endl;
		}

		 static inline Vec3<T> normalize(const Vec3<T>& v) {
			return v.getNormalized();
		}

		 inline void normalize() {
			T val = (T)1.0 / length();
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
		}

		// If this Vec3 is non-zero, then normalize it, else return.
		 inline void normalizeIfNonzero() {
			const T l = length();
			if (l == static_cast<T>(0)) { return; } // TODO: Better to check against epsilon tolerance
			const T val = static_cast<T>(1) / l;
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
		}

		 inline Vec3<T> getNormalized() const {
			T val = (T)1.0 / length();
			return Vec3<T>(m_array[0] * val, m_array[1] * val, m_array[2] * val);
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

		 inline T* getData() {
			return &m_array[0];
		}

		 inline const T* getData() const {
			return &m_array[0];
		}

		 inline std::string toString(char separator = ' ') const {
			return toString(std::string(1, separator));
		}

		 inline std::string toString(const std::string& separator) const {
			return std::to_string(x()) + separator + std::to_string(y()) + separator + std::to_string(z());
		}

	private:
		T m_array[3];

	};

	/**
	 * Math operators.
	 */
	template <class T>
	 Vec3<T> operator*(T s, const Vec3<T>& v) {
		return v * s;
	}

	template <class T>
	 Vec3<T> operator/(T s, const Vec3<T>& v) {
		return Vec3<T>(s / v.x(), s / v.y(), s / v.z());
	}

	template <class T>
	 Vec3<T> operator+(T s, const Vec3<T>& v) {
		return v + s;
	}

	template <class T>
	 Vec3<T> operator-(T s, const Vec3<T>& v) {
		return Vec3<T>(s - v.x(), s - v.y(), s - v.z());
	}

	// Write a Vec3 to a stream.
	template <class T>
	 std::ostream& operator<<(std::ostream& s, const Vec3<T>& v) {
		return (s << v[0] << " " << v[1] << " " << v[2]);
	}

	// Read a Vec3 from a stream.
	template <class T>
	 std::istream& operator>>(std::istream& s, Vec3<T>& v) {
		return (s >> v[0] >> v[1] >> v[2]);
	}

	/**
	 * Comparison operators.
	 */
	template<class T>  bool operator==(const Vec3<T>& lhs, const Vec3<T>& rhs) { return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2]; }
	template<class T>  bool operator!=(const Vec3<T>& lhs, const Vec3<T>& rhs) { return !operator==(lhs, rhs); }
	template<class T>  bool operator< (const Vec3<T>& lhs, const Vec3<T>& rhs) { return std::tie(lhs[0], lhs[1], lhs[2]) < std::tie(rhs[0], rhs[1], rhs[2]); }
	template<class T>  bool operator> (const Vec3<T>& lhs, const Vec3<T>& rhs) { return  operator< (rhs, lhs); }
	template<class T>  bool operator<=(const Vec3<T>& lhs, const Vec3<T>& rhs) { return !operator> (lhs, rhs); }
	template<class T>  bool operator>=(const Vec3<T>& lhs, const Vec3<T>& rhs) { return !operator< (lhs, rhs); }

	typedef Vec3<double> Vec3d;
	typedef Vec3<float> Vec3f;
	typedef Vec3<int> Vec3i;
	typedef Vec3<short> Vec3s;
	typedef Vec3<unsigned short> Vec3us;
	typedef Vec3<unsigned int> Vec3ui;
	typedef Vec3<unsigned char> Vec3uc;
	typedef Vec3<unsigned long long> Vec3ul;
	typedef Vec3<long long> Vec3l;

	/**
	 * Math operations.
	 */
	namespace math_proc {
		template<class T>
		 Vec3i round(const Vec3<T>& f) {
			return Vec3i(round(f.x()), round(f.y()), round(f.z()));
		}
		template<class T>
		 Vec3i ceil(const Vec3<T>& f) {
			return Vec3i(ceil(f.x()), ceil(f.y()), ceil(f.z()));
		}
		template<class T>
		 Vec3i floor(const Vec3<T>& f) {
			return Vec3i(floor(f.x()), floor(f.y()), floor(f.z()));
		}
		template<class T>
		 Vec3<T> abs(const Vec3<T>& p) {
			return Vec3<T>(abs(p.x()), abs(p.y()), abs(p.z()));
		}
		template<class T>
		 Vec3<T> sqrt(const Vec3<T>& p) {
			return Vec3<T>(sqrt(p.x()), sqrt(p.y()), sqrt(p.z()));
		}
		template<class T>
		 Vec3<T> max(const Vec3<T>& p, T v) {
			return Vec3<T>(max(p.x(), v),
				max(p.y(), v),
				max(p.z(), v));
		}
		template<class T>
		 Vec3<T> max(const Vec3<T>& p, const Vec3<T>& v) {
			return Vec3<T>(
				max(p.x(), v.x()),
				max(p.y(), v.y()),
				max(p.z(), v.z()));
		}
		template<class T>
		 Vec3<T> min(const Vec3<T>& p, T v) {
			return Vec3<T>(min(p.x(), v),
				min(p.y(), v),
				min(p.z(), v));
		}
		template<class T>
		 Vec3<T> min(const Vec3<T>& p, const Vec3<T>& v) {
			return Vec3<T>(
				min(p.x(), v.x()),
				min(p.y(), v.y()),
				min(p.z(), v.z()));
		}
	} // namespace math_proc

} // namespace matrix_lib
