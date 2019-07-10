#pragma once

namespace matrix_lib {

	/**
	 * s-dimensional vector.
	 */
	template <class T, int s>
	class VecX {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		 explicit VecX(T v) {
			for (int i = 0; i < s; ++i) m_array[i] = v;
		}

		// Default constructor (we only set values to zero if we are dealing
		// with fundamental types, otherwise we just use default constructor of
		// a compound type).
		 void reset(std::true_type) {
			for (int i = 0; i < s; ++i) m_array[i] = T(0);
		}

		 void reset(std::false_type) { }

		 VecX() {
			reset(std::is_fundamental<T>::type());
		}

		// Copy data from other template type.
		template <class U>
		 VecX(const VecX<U, s>& other) {
			for (int i = 0; i < s; ++i) m_array[i] = T(other[i]);
		}

		// Initialize with raw array.
		 explicit VecX(const T* values) {
			for (int i = 0; i < s; ++i) m_array[i] = values[i];
		}

		 VecX(std::initializer_list<T> values) {
			runtime_assert(values.size() == s, "The number of parameters should equal s.");
			int i = 0;
			for (const auto& val : values) {
				m_array[i] = val;
				i++;
			}
		}

		/**
		 * Basic operations.
		 */
		 VecX<T, s> operator-() const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = -m_array[i];
			return output;
		}

		 VecX<T, s> operator+(const VecX& other) const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = m_array[i] + other.m_array[i];
			return output;
		}

		 VecX<T, s> operator+(T val) const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = m_array[i] + val;
			return output;
		}

		 void operator+=(const VecX& other) {
			for (int i = 0; i < s; ++i) m_array[i] += other.m_array[i];
		}

		 void operator-=(const VecX& other) {
			for (int i = 0; i < s; ++i) m_array[i] -= other.m_array[i];
		}

		 void operator+=(T val) {
			for (int i = 0; i < s; ++i) m_array[i] += val;
		}

		 void operator-=(T val) {
			for (int i = 0; i < s; ++i) m_array[i] -= val;
		}

		 void operator*=(T val) {
			for (int i = 0; i < s; ++i) m_array[i] *= val;
		}

		 void operator/=(T val) {\
			for (int i = 0; i < s; ++i) m_array[i] /= val;
		}

		 VecX<T, s> operator*(T val) const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = m_array[i] * val;
			return output;
		}

		 VecX<T, s> operator/(T val) const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = m_array[i] / val;
			return output;
		}

		// Dot product.
		 T operator|(const VecX& other) const {
			T output;
			for (int i = 0; i < s; ++i) output += m_array[i] * other[i];
			return output;
		}

		 VecX<T, s> operator-(const VecX& other) const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = m_array[i] - other[i];
			return output;
		}

		 VecX<T, s> operator-(T val) const {
			VecX<T, s> output;
			for (int i = 0; i < s; ++i) output[i] = m_array[i] - val;
			return output;
		}

		 bool operator==(const VecX& other) const {
			bool bEqual = true;
			for (int i = 0; i < s; ++i) bEqual = bEqual && (m_array[i] == other[i]);
			return bEqual;
		}

		 bool operator!=(const VecX& other) const {
			return !(*this == other);
		}

		 T lengthSq() const {
			return (*this) | (*this);
		}

		 T length() const {
			return sqrt(lengthSq());
		}

		 static inline VecX<T, s> normalize(const VecX<T, s>& v) {
			return v.getNormalized();
		}

		 static T distSq(const VecX& v0, const VecX& v1) {
			return (v0 - v1).lengthSq();
		}

		 static T dist(const VecX& v0, const VecX& v1) {
			return (v0 - v1).length();
		}

		 void normalize() {
			T val = T(1.0) / length();
			for (int i = 0; i < s; ++i) m_array[i] *= val;
		}

		 bool isValid() const {
			bool bValid = true;
			for (int i = 0; i < s; ++i) bValid = bValid && (m_array[i] == m_array[i]);
			return bValid;
		}

		 VecX<T, s> getNormalized() const {
			VecX<T, s> output{ *this };
			output.normalize();
			return output;
		}

		/**
		 * Indexing operators.
		 */
		 const T& operator[](int i) const {
			runtime_assert(i < s, "Index out of bounds.");
			return m_array[i];
		}

		 T& operator[](int i) {
			runtime_assert(i < s, "Index out of bounds.");
			return m_array[i];
		}

		template<unsigned i>
		 T& operator[](I<i>) {
			static_assert(i < s, "Index out of bounds.");
			return m_array[i];
		}

		template<unsigned i>
		 const T& operator[](I<i>) const {
			static_assert(i < s, "Index out of bounds.");
			return m_array[i];
		}

		/**
		 * Getters/setters.
		 */
		 T* getData() {
			return &m_array[0];
		}

		 const T* getData() const {
			return &m_array[0];
		}

		 std::string toString(char separator = ' ') const {
			return toString(std::string(1, separator));
		}

		 std::string toString(const std::string& separator) const {
			std::string output;
			for (int i = 0; i < s - 1; ++i) output += std::to_string(m_array[i]) + separator;
			if (s > 0) output += std::to_string(m_array[s - 1]);
			return output;
		}

		 void print() const {
			if (s > 0) {
				std::cout << "(";
				for (int i = 0; i < s - 1; ++i) std::cout << m_array[i] << " / ";
				std::cout << m_array[s - 1] << ")" << std::endl;
			}
		}

		 static int size() {
			return s;
		}

	private:
		T m_array[s];
	};

	/**
	 * Math operations.
	 */
	template <class T, int s>
	 VecX<T, s> operator*(T val, const VecX<T, s>& v) {
		return v * val;
	}

	template <class T, int s>
	 VecX<T, s> operator/(T val, const VecX<T, s>& v) {
		VecX<T, s> output;
		for (int i = 0; i < s; ++i) output[i] = val / v[i];
		return output;
	}

	template <class T, int s>
	 VecX<T, s> operator+(T val, const VecX<T, s>& v) {
		return v + val;
	}

	template <class T, int s>
	 VecX<T, s> operator-(T val, const VecX<T, s>& v) {
		return -v + val;
	}

	// Write a VecX to a stream.
	template <class T, int s>
	 std::ostream& operator<<(std::ostream& os, const VecX<T, s>& v) {
		for (int i = 0; i < s; ++i) os << v[i];
		return os;
	}

	// Read a VecX from a stream.
	template <class T, int s>
	 std::istream& operator>>(std::istream& is, VecX<T, s>& v) {
		for (int i = 0; i < s; ++i) is >> v[i];
		return is;
	}

	/**
	 * Comparison operators.
	 */
	template<class T, int s>  bool operator==(const VecX<T, s>& lhs, const VecX<T, s>& rhs) { 
		for (int i = 0; i < s; ++i) {
			if (lhs[i] != rhs[i]) return false;
		}
		return true;
	}
	template<class T, int s>  bool operator!=(const VecX<T, s>& lhs, const VecX<T, s>& rhs) { return !operator==(lhs, rhs); }
	template<class T, int s>  bool operator< (const VecX<T, s>& lhs, const VecX<T, s>& rhs) { 
		for (int i = 0; i < s; ++i) {
			if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
		}
		return false;
	}	
	template<class T, int s>  bool operator> (const VecX<T, s>& lhs, const VecX<T, s>& rhs) { return  operator< (rhs, lhs); }
	template<class T, int s>  bool operator<=(const VecX<T, s>& lhs, const VecX<T, s>& rhs) { return !operator> (lhs, rhs); }
	template<class T, int s>  bool operator>=(const VecX<T, s>& lhs, const VecX<T, s>& rhs) { return !operator< (lhs, rhs); }

	template <typename T> using Vec1 = VecX<T, 1>;
	template <typename T> using Vec5 = VecX<T, 5>;
	template <typename T> using Vec6 = VecX<T, 6>;
	template <typename T> using Vec7 = VecX<T, 7>;
	template <typename T> using Vec8 = VecX<T, 8>;
	template <typename T> using Vec9 = VecX<T, 9>;

	typedef Vec1<float> Vec1f;
	typedef Vec1<double> Vec1d;
	typedef Vec1<unsigned> Vec1ui;
	typedef Vec1<int> Vec1i;
	typedef Vec5<float> Vec5f;
	typedef Vec5<double> Vec5d;
	typedef Vec5<unsigned> Vec5ui;
	typedef Vec5<int> Vec5i;
	typedef Vec6<float> Vec6f;
	typedef Vec6<double> Vec6d;
	typedef Vec6<unsigned> Vec6ui;
	typedef Vec6<int> Vec6i;
	typedef Vec7<float> Vec7f;
	typedef Vec7<double> Vec7d;
	typedef Vec7<unsigned> Vec7ui;
	typedef Vec7<int> Vec7i;
	typedef Vec8<float> Vec8f;
	typedef Vec8<double> Vec8d;
	typedef Vec8<unsigned> Vec8ui;
	typedef Vec8<int> Vec8i;
	typedef Vec9<float> Vec9f;
	typedef Vec9<double> Vec9d;
	typedef Vec9<unsigned> Vec9ui;
	typedef Vec9<int> Vec9i;

} // namespace matrix_lib
