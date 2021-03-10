
MCCL_BEGIN_NAMESPACE

namespace detail
{

#ifdef _MSC_VER
	inline unsigned int trailing_zeroes(uint32_t n)
	{
		unsigned long ret = 0;
		_BitScanForward(&ret, n);
		return ret;
	}
#else
	inline unsigned int trailing_zeroes(uint32_t n)
	{
		return __builtin_ctz(n);
	}
#endif

	using namespace std;

	template<typename data_t>
	void matrix_print(std::ostream& o, const matrix_base_ref_t<data_t>& m, bool transpose)
	{
		o << "[";
		if (!transpose)
		{
			for (size_t r = 0; r < m.rows; ++r)
			{
				o << (r==0 ? "[" : " [");
				for (size_t c = 0; c < m.columns; ++c)
					o << m(r,c);
				o << "]" << std::endl;
			}
		}
		else
		{
			for (size_t c = 0; c < m.columns; ++c)
			{
				o << "[";
				for (size_t r = 0; r < m.rows; ++r)
					o << m(r,c);
				o << "]" << std::endl;
			}
		}
		o << "]";
	}
	template<typename data_t>
	void vector_print(std::ostream& o, const matrix_base_ref_t<data_t>& m)
	{
		o << "[";
		for (size_t c = 0; c < m.columns; ++c)
			o << m(0,c);
		o << "]";
	}

	template<typename data_t>
	inline bool matrix_compare(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2)
	{
		if (m1.rows != m2.rows || m1.columns != m2.columns)
			return false;
		const size_t words = (m1.columns + m1.word_bits - 1) / m1.word_bits;
		if ((m1.columns % m1.word_bits) == 0)
		{
			for (size_t r = 0; r < m1.rows; ++r)
			{
				data_t* first1 = m1.data(r), first2 = m2.data(r), last1 = m1.data(r) + words;
				for (; first1 != last1; ++first1, ++first2)
					if (*first1 != *first2)
						return false;
			}
		}
		else
		{
			const data_t lastwordmask = m1.lastwordmask();
			for (size_t r = 0; r < m1.rows; ++r)
			{
				data_t* first1 = m1.data(r), first2 = m2.data(r), last1 = m1.data(r) + words - 1;
				for (; first1 != last1; ++first1, ++first2)
					if (*first1 != *first2)
						return false;
				if ((lastwordmask & *first1) != (lastwordmask & *first2))
					return false;
			}
		}
		return true;
	}
	template<typename data_t>
	inline bool vector_compare(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2)
	{
		if (m1.columns != m2.columns)
			return false;
		const size_t words = (m1.columns + m1.word_bits - 1) / m1.word_bits;
		if ((m1.columns % m1.word_bits) == 0)
		{
			data_t* first1 = m1.data(0), first2 = m2.data(0), last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1, ++first2)
				if (*first1 != *first2)
					return false;
		}
		else
		{
			const data_t lastwordmask = m1.lastwordmask();
			data_t* first1 = m1.data(0), first2 = m2.data(0), last1 = m1.data(0) + words - 1;
			for (; first1 != last1; ++first1, ++first2)
				if (*first1 != *first2)
					return false;
			if ((lastwordmask & *first1) != (lastwordmask & *first2))
				return false;
		}
		return true;
	}



	template<typename data_t>
	inline void matrix_set(const matrix_base_ref_t<data_t>& m, const data_t& value)
	{
		if (m.stride * m.word_bits == m.columns + m.scratchcolumns)
		{
			std::fill(m.data(0), m.data(m.rows), value);
		}
		else
		{
			const size_t words = (m.columns + m.word_bits - 1) / m.word_bits;
			for (size_t r = 0; r < m.rows; ++r)
				std::fill(m.data(r), m.data(r) + words, value);
		}
	}
	template<typename data_t>
	inline void matrix_set(const matrix_base_ref_t<data_t>& m, bool b)
	{
		const data_t value = b ? data_t(-1) : data_t();
		matrix_set(m, value);
	}
	template<typename data_t>
	inline void vector_set(const matrix_base_ref_t<data_t>& m, const data_t& value)
	{
		const size_t words = (m.columns + m.word_bits - 1) / m.word_bits;
		std::fill(m.data(), m.data() + words, value);
	}
	template<typename data_t>
	inline void vector_set(const matrix_base_ref_t<data_t>& m, bool b)
	{
		const data_t value = b ? data_t(-1) : data_t();
		vector_set(m, value);
	}



	template<typename data_t>
	inline void matrix_setcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns, bool b)
	{
		MCCL_MATRIX_BASE_ASSERT(column_offset + columns <= m.columns + m.scratchcolumns);
		const size_t firstword = column_offset / m.word_bits;
		const size_t lastword = (column_offset + columns) / m.word_bits;
		data_t firstwordmask = m.wordmaskhigh(column_offset);
		data_t lastwordmask = ~m.wordmaskhigh(column_offset + columns);
		if (firstword == lastword)
		{
			firstwordmask &= lastwordmask;
			if (b == true)
			{
				for (size_t r = 0; r < m.rows; ++r)
					*(m.data(r) + firstword) |= firstwordmask;
			}
			else
			{
				firstwordmask = ~firstwordmask;
				for (size_t r = 0; r < m.rows; ++r)
					*(m.data(r) + firstword) &= firstwordmask;
			}
			return;
		}
		if (b == true)
		{
			for (size_t r = 0; r < m.rows; ++r)
			{
				*(m.data(r) + firstword) |= firstwordmask;
				std::fill(m.data(r) + firstword + 1, m.data(r) + lastword, data_t(-1));
				*(m.data(r) + lastword) |= lastwordmask;
			}
		}
		else
		{
			firstwordmask = ~firstwordmask;
			lastwordmask = ~lastwordmask;
			for (size_t r = 0; r < m.rows; ++r)
			{
				*(m.data(r) + firstword) &= firstwordmask;
				std::fill(m.data(r) + firstword + 1, m.data(r) + lastword, data_t(0));
				*(m.data(r) + lastword) &= lastwordmask;
			}
		}
	}
	template<typename data_t>
	inline void vector_setcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns, bool b)
	{
		MCCL_MATRIX_BASE_ASSERT(column_offset + columns <= m.columns + m.scratchcolumns);
		const size_t firstword = column_offset / m.word_bits;
		const size_t lastword = (column_offset + columns) / m.word_bits;
		data_t firstwordmask = m.wordmaskhigh(column_offset);
		data_t lastwordmask = ~m.wordmaskhigh(column_offset + columns);
		if (firstword == lastword)
		{
			firstwordmask &= lastwordmask;
			if (b == true)
				*(m.data(0) + firstword) |= firstwordmask;
			else
				*(m.data(0) + firstword) &= ~firstwordmask;
			return;
		}
		const size_t memsetbytes = (lastword - firstword - 1) * sizeof(data_t);
		if (b == true)
		{
			*(m.data(0) + firstword) |= firstwordmask;
			std::fill(m.data() + firstword + 1, m.data() + lastword, data_t(-1));
			*(m.data(0) + lastword) |= lastwordmask;
		}
		else
		{
			*(m.data(0) + firstword) &= ~firstwordmask;
			std::fill(m.data() + firstword + 1, m.data() + lastword, data_t(0));
			*(m.data(0) + lastword) &= ~lastwordmask;
		}
	}
	template<typename data_t>
	inline void matrix_flipcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns)
	{
		MCCL_MATRIX_BASE_ASSERT(column_offset + columns <= m.columns + m.scratchcolumns);
		const size_t firstword = column_offset / m.word_bits;
		const size_t lastword = (column_offset + columns) / m.word_bits;
		data_t firstwordmask = m.wordmaskhigh(column_offset);
		data_t lastwordmask = ~m.wordmaskhigh(column_offset + columns);
		if (firstword == lastword)
		{
			firstwordmask &= lastwordmask;
			for (size_t r = 0; r < m.rows; ++r)
				*(m.data(r) + firstword) ^= firstwordmask;
			return;
		}
		for (size_t r = 0; r < m.rows; ++r)
		{
			*(m.data(r) + firstword) ^= firstwordmask;
			for (auto ptr = m.data(r) + firstword + 1; ptr != m.data(r) + lastword; ++ptr)
				*ptr = ~(*ptr);
			*(m.data(r) + lastword) ^= lastwordmask;
		}
	}
	template<typename data_t>
	inline void vector_flipcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns)
	{
		MCCL_MATRIX_BASE_ASSERT(column_offset + columns <= m.columns + m.scratchcolumns);
		const size_t firstword = column_offset / m.word_bits;
		const size_t lastword = (column_offset + columns) / m.word_bits;
		data_t firstwordmask = m.wordmaskhigh(column_offset);
		data_t lastwordmask = ~m.wordmaskhigh(column_offset + columns);
		if (firstword == lastword)
		{
			firstwordmask &= lastwordmask;
			*(m.data(0) + firstword) ^= firstwordmask;
			return;
		}
		*(m.data(0) + firstword) ^= firstwordmask;
		for (auto ptr = m.data() + firstword + 1; ptr != m.data() + lastword; ++ptr)
			*ptr = ~(*ptr);
		*(m.data(0) + lastword) ^= lastwordmask;
	}




	template<typename data_t>
	inline void matrix_not(const matrix_base_ref_t<data_t>& m)
	{
		if (m.stride * m.word_bits == m.columns + m.scratchcolumns)
		{
			data_t* endptr = m.data(m.rows);
			for (data_t* ptr = m.data(0); ptr != endptr; ++ptr)
				*ptr = ~(*ptr);
		}
		else
		{
			const size_t words = (m.columns + m.word_bits - 1) / m.word_bits;
			for (size_t r = 0; r < m.rows; ++r)
			{
				data_t* endptr = m.data(r) + words;
				for (data_t* ptr = m.data(r); ptr != endptr; ++ptr)
					*ptr = ~(*ptr);
			}
		}
	}
	template<typename data_t>
	inline void vector_not(const matrix_base_ref_t<data_t>& m)
	{
		const size_t words = (m.columns + m.word_bits - 1) / m.word_bits;
		data_t* endptr = m.data(1) + words;
		for (data_t* ptr = m.data(0); ptr != endptr; ++ptr)
			*ptr = ~(*ptr);
	}


	/* binary boolean functions

	ab ab ab ab : a=dst, b=src
	00 01 10 11
	 0  0  0  0 = set 0
	 0  0  0  1 = a and b          => and
	 0  0  1  0 = a and not b      => andin
	 0  0  1  1 = copy a
	 0  1  0  0 = not a and b      => andni
	 0  1  0  1 = copy b
	 0  1  1  0 = a xor b          => xor
	 0  1  1  1 = a or b           => or
	 1  0  0  0 = not a and not b  => nor
	 1  0  0  1 = not (a xor b)    => nxor
	 1  0  1  0 = not b
	 1  0  1  1 = a or not b       => orin
	 1  1  0  0 = not a
	 1  1  0  1 = not a or b       => orni
	 1  1  1  0 = not (a and b)    => nand
	 1  1  1  1 = set 1
	 */

	/* dst = bf(dst,src) versions */
	template<typename data_t>
	inline void copy_not(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = ~(*first);
	}
	template<typename data_t>
	inline void binary_and(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = (*dst) & (*first);
	}
	template<typename data_t>
	inline void binary_xor(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = (*dst) ^ (*first);
	}
	template<typename data_t>
	inline void binary_or(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = (*dst) | (*first);
	}
	template<typename data_t>
	inline void binary_nand(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = ~((*dst) & (*first));
	}
	template<typename data_t>
	inline void binary_nxor(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = ~((*dst) ^ (*first));
	}
	template<typename data_t>
	inline void binary_nor(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = ~((*dst) | (*first));
	}
	template<typename data_t>
	inline void binary_andin(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = (*dst) & ~(*first);
	}
	template<typename data_t>
	inline void binary_andni(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = ~(*dst) & (*first);
	}
	template<typename data_t>
	inline void binary_orin(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = (*dst) | ~(*first);
	}
	template<typename data_t>
	inline void binary_orni(const data_t* first, const data_t* last, data_t* dst)
	{
		for (; first != last; ++first, ++dst)
			*dst = ~(*dst) | (*first);
	}

	/* dst = bf(src1,src2) versions */
	template<typename data_t>
	inline void binary_and(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = (*first1) & (*first2);
	}
	template<typename data_t>
	inline void binary_xor(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = (*first1) ^ (*first2);
	}
	template<typename data_t>
	inline void binary_or(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = (*first1) | (*first2);
	}
	template<typename data_t>
	inline void binary_nand(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = ~((*first1) & (*first2));
	}
	template<typename data_t>
	inline void binary_nxor(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = ~((*first1) ^ (*first2));
	}
	template<typename data_t>
	inline void binary_nor(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = ~((*first1) | (*first2));
	}
	template<typename data_t>
	inline void binary_andin(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = (*first1) & ~(*first2);
	}
	template<typename data_t>
	inline void binary_andni(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = ~(*first1) & (*first2);
	}
	template<typename data_t>
	inline void binary_orin(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = (*first1) | ~(*first2);
	}
	template<typename data_t>
	inline void binary_orni(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst)
	{
		for (; first1 != last1; ++first1, ++first2, ++dst)
			*dst = ~(*first1) | (*first2);
	}

#define GENERATE_MATRIX_OP2(matrixfunc, binaryfunc) \
	template<typename data_t> \
	inline void matrixfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src) \
	{ \
		if (dst.columns != src.columns || dst.rows != src.rows) \
			throw std::runtime_error("matrix_func: matrix size does not match"); \
		if (dst.stride == src.stride \
			&& dst.stride * dst.word_bits == dst.columns + dst.scratchcolumns \
			&& dst.scratchcolumns == src.scratchcolumns) \
		{ \
			binaryfunc(src.data(0), src.data(src.rows), dst.data(0)); \
		} \
		else \
		{ \
			const size_t words = (dst.columns + dst.word_bits - 1) / dst.word_bits; \
			for (size_t r = 0; r < dst.rows; ++r) \
				binaryfunc(src.data(r), src.data(r)+words, dst.data(r)); \
		} \
	}
	GENERATE_MATRIX_OP2(matrix_copy, std::copy)
	GENERATE_MATRIX_OP2(matrix_copynot, copy_not)
	GENERATE_MATRIX_OP2(matrix_and, binary_and)
	GENERATE_MATRIX_OP2(matrix_xor, binary_xor)
	GENERATE_MATRIX_OP2(matrix_or, binary_or)
	GENERATE_MATRIX_OP2(matrix_nand, binary_nand)
	GENERATE_MATRIX_OP2(matrix_nxor, binary_nxor)
	GENERATE_MATRIX_OP2(matrix_nor, binary_nor)
	GENERATE_MATRIX_OP2(matrix_andin, binary_andin)
	GENERATE_MATRIX_OP2(matrix_andni, binary_andni)
	GENERATE_MATRIX_OP2(matrix_orin, binary_orin)
	GENERATE_MATRIX_OP2(matrix_orni, binary_orni)

#define GENERATE_MATRIX_OP3(matrixfunc, binaryfunc) \
	template<typename data_t> \
	inline void matrixfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src1, const matrix_base_ref_t<const data_t>& src2) \
	{ \
		if (dst.columns != src1.columns || dst.rows != src1.rows || dst.columns != src2.columns || dst.rows != src2.rows) \
			throw std::runtime_error("matrix_func: matrix size does not match"); \
		if (dst.stride == src1.stride && dst.stride == src2.stride \
			&& dst.stride * dst.word_bits == dst.columns + dst.scratchcolumns \
			&& dst.scratchcolumns == src1.scratchcolumns && dst.scratchcolumns == src2.scratchcolumns) \
		{ \
			binaryfunc(src1.data(0), src1.data(src1.rows), src2.data(0), dst.data(0)); \
		} \
		else \
		{ \
			const size_t words = (dst.columns + dst.word_bits - 1) / dst.word_bits; \
			for (size_t r = 0; r < dst.rows; ++r) \
				binaryfunc(src1.data(r), src1.data(r)+words, src2.data(r), dst.data(r)); \
		} \
	}
	GENERATE_MATRIX_OP3(matrix_copy, std::copy)
	GENERATE_MATRIX_OP3(matrix_copynot, copy_not)
	GENERATE_MATRIX_OP3(matrix_and, binary_and)
	GENERATE_MATRIX_OP3(matrix_xor, binary_xor)
	GENERATE_MATRIX_OP3(matrix_or, binary_or)
	GENERATE_MATRIX_OP3(matrix_nand, binary_nand)
	GENERATE_MATRIX_OP3(matrix_nxor, binary_nxor)
	GENERATE_MATRIX_OP3(matrix_nor, binary_nor)
	GENERATE_MATRIX_OP3(matrix_andin, binary_andin)
	GENERATE_MATRIX_OP3(matrix_andni, binary_andni)
	GENERATE_MATRIX_OP3(matrix_orin, binary_orin)
	GENERATE_MATRIX_OP3(matrix_orni, binary_orni)



#define GENERATE_VECTOR_OP2(vectorfunc, binaryfunc) \
	template<typename data_t> \
	inline void vectorfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src) \
	{ \
		if (dst.columns != src.columns) \
			throw std::runtime_error("vector_func: vector size does not match"); \
		const size_t words = (dst.columns + dst.word_bits - 1) / dst.word_bits; \
		binaryfunc(src.data(0), src.data(0)+words, dst.data(0)); \
	}
	GENERATE_VECTOR_OP2(vector_copy, std::copy)
	GENERATE_VECTOR_OP2(vector_copynot, copy_not)
	GENERATE_VECTOR_OP2(vector_and, binary_and)
	GENERATE_VECTOR_OP2(vector_xor, binary_xor)
	GENERATE_VECTOR_OP2(vector_or, binary_or)
	GENERATE_VECTOR_OP2(vector_nand, binary_nand)
	GENERATE_VECTOR_OP2(vector_nxor, binary_nxor)
	GENERATE_VECTOR_OP2(vector_nor, binary_nor)
	GENERATE_VECTOR_OP2(vector_andin, binary_andin)
	GENERATE_VECTOR_OP2(vector_andni, binary_andni)
	GENERATE_VECTOR_OP2(vector_orin, binary_orin)
	GENERATE_VECTOR_OP2(vector_orni, binary_orni)

#define GENERATE_VECTOR_OP3(vectorfunc, binaryfunc) \
	template<typename data_t> \
	inline void vectorfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src1, const matrix_base_ref_t<const data_t>& src2) \
	{ \
		if (dst.columns != src1.columns || dst.columns != src2.columns) \
			throw std::runtime_error("vector_func: vector size does not match"); \
		const size_t words = (dst.columns + dst.word_bits - 1) / dst.word_bits; \
		binaryfunc(src1.data(0), src1.data(0)+words, src2.data(0), dst.data(0)); \
	}
	GENERATE_VECTOR_OP3(vector_copy, std::copy)
	GENERATE_VECTOR_OP3(vector_copynot, copy_not)
	GENERATE_VECTOR_OP3(vector_and, binary_and)
	GENERATE_VECTOR_OP3(vector_xor, binary_xor)
	GENERATE_VECTOR_OP3(vector_or, binary_or)
	GENERATE_VECTOR_OP3(vector_nand, binary_nand)
	GENERATE_VECTOR_OP3(vector_nxor, binary_nxor)
	GENERATE_VECTOR_OP3(vector_nor, binary_nor)
	GENERATE_VECTOR_OP3(vector_andin, binary_andin)
	GENERATE_VECTOR_OP3(vector_andni, binary_andni)
	GENERATE_VECTOR_OP3(vector_orin, binary_orin)
	GENERATE_VECTOR_OP3(vector_orni, binary_orni)



	template<typename data_t, size_t bits>
	inline void block_transpose(data_t* dst, size_t dststride, const data_t* src, size_t srcstride)
	{
		static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
		static_assert(sizeof(data_t)*8 >= bits, "bits must not exceed data_t bitsize");

		// mask of lower half bits
		data_t m = (data_t(1) << (bits/2))-1;
		unsigned int j = (bits/2);
		data_t tmp[bits];

		// first loop iteration, load src store in tmp
//#pragma unroll
		const data_t* src2 = src + ((bits/2)*srcstride);
		for (unsigned int k=0;  k<bits/2;  ++k, src+=srcstride, src2+=srcstride)
		{
			// j = (bits/2)
			data_t a = *src, b = *src2;
			data_t t = ((a>>(bits/2)) ^ b) & m;
			tmp[k] = a ^ (t << (bits/2));
			tmp[k+(bits/2)] = b ^ t;
		}
		j>>=1; m^=m<<j;
		// main loop
		for (;  1 != j;  j>>=1,m^=m<<j)
		{
//#pragma unroll
			for (unsigned int l=0,k=0;  l<bits/2;  ++l)
			{
				data_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
				tmp[k] ^= t<<j;
				tmp[k+j] ^= t;
				k=(k+j+1)&~j;
			}
		}
		// last loop iteration (j==1), load tmp store in dst
//#pragma unroll
		for (unsigned int k=0;  k<bits;  k += 2)
		{
			data_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
			*dst = tmp[k] ^ (t<<1); dst+=dststride;
			*dst = tmp[k+1] ^ t; dst+=dststride;
		}
	}

	template<typename data_t, size_t bits>
	inline void block_transpose2(data_t* dst, size_t dststride, const data_t* src, size_t srcstride)
	{
		static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
		static_assert(sizeof(data_t)*8 >= bits, "bits must not exceed data_t bitsize");

		// mask of lower half bits
		data_t m = (data_t(1) << (bits/2))-1;
		unsigned int j = (bits/2);
		data_t tmp[2*bits];

		// first loop iteration, load src store in tmp
//#pragma unroll
		const data_t* src2 = src + ((bits/2)*srcstride);
		for (unsigned int k=0;  k<bits/2;  ++k, src+=srcstride, src2+=srcstride)
		{
			// j = (bits/2)
			data_t a1 = *src, b1 = *src2;
			data_t t1 = ((a1>>(bits/2)) ^ b1) & m;
			tmp[k] = a1 ^ (t1 << (bits/2));
			tmp[k+(bits/2)] = b1 ^ t1;
			data_t a2 = *(src+1), b2 = *(src2+1);
			data_t t2 = ((a2>>(bits/2)) ^ b2) & m;
			tmp[k+bits] = a2 ^ (t2 << (bits/2));
			tmp[k+(bits/2)+bits] = b2 ^ t2;

		}
		j>>=1; m^=m<<j;
		// main loop
		for (;  1 != j;  j>>=1,m^=m<<j)
		{
//#pragma unroll
			for (unsigned int l=0,k=0;  l<bits/2;  ++l)
			{
				data_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
				tmp[k] ^= t<<j;
				tmp[k+j] ^= t;
				data_t t2 = ((tmp[k+bits]>>j) ^ tmp[k+j+bits]) & m;
				tmp[k+bits] ^= t2<<j;
				tmp[k+j+bits] ^= t2;

				k=(k+j+1)&~j;
			}
		}
		// last loop iteration (j==1), load tmp store in dst
//#pragma unroll
		for (unsigned int k=0;  k<2*bits;  k += 2)
		{
			data_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
			*dst = tmp[k] ^ (t<<1); dst+=dststride;
			*dst = tmp[k+1] ^ t; dst+=dststride;
		}
	}



	template<typename data_t, size_t bits>
	inline void block_transpose(data_t* dst, size_t dststride, size_t dstrows, const data_t* src, size_t srcstride, size_t srcrows)
	{
		static_assert(bits >= 4, "bits >= 4");
		static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
		static_assert(sizeof(data_t)*8 >= bits, "bits must not exceed data_t bitsize");
		assert(dstrows <= bits);
		assert(srcrows <= bits);

		// mask of lower half bits
		data_t m = (data_t(1) << (bits/2))-1;
		unsigned int j = (bits/2);
		data_t tmp[bits+2]; // <= add 2 to avoid incorrect out-of-bounds warning

		// first loop iteration, load src store in tmp
		const data_t* src2 = src + ((bits/2)*srcstride);
		for (unsigned int k=0;  k<bits/2;  ++k)
		{
			if (k < srcrows)
			{
				data_t a = *src, b = 0;
				src += srcstride;
				if ((k+(bits/2)) < srcrows)
				{
					b = *src2;
					src2 += srcstride;
				}
				data_t t = (b ^ (a >> (bits/2))) & m;
				tmp[k] = a ^ (t << (bits/2));
				tmp[k+(bits/2)] = b ^ t;
			}
			else
			{
				tmp[k] = 0;
				tmp[k+(bits/2)] = 0;
			}
		}
		j>>=1; m^=m<<j;
		// main loop
		for (;  1 != j;  j>>=1,m^=m<<j)
		{
			for (unsigned int l=0,k=0;  l<bits/2;  ++l)
			{
				data_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
				tmp[k] ^= t<<j;
				tmp[k+j] ^= t;
				k=(k+j+1)&~j;
			}
		}
		// last loop iteration (j==1), load tmp store in dst
		unsigned int k=0;
		for (;  k+1 < dstrows;  k += 2)
		{
			data_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
			*dst = tmp[k] ^ (t<<1); dst+=dststride;
			*dst = tmp[k+1] ^ t; dst+=dststride;
		}
		// note both k and bits are even and k < dstrows <= bits
		// so k+1 < bits as well, nevertheless compilers may warn
		if (k < dstrows)
		{
			data_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
			*dst = tmp[k] ^ (t<<1);
		}
	}



	template<typename data_t>
	inline void block_transpose(data_t* dst, size_t dststride, size_t dstrows, const data_t* src, size_t srcstride, size_t srcrows, size_t bits)
	{
		assert(0 == (bits&(bits-1))); // bits must be power of 2
		assert(sizeof(data_t)*8 >= bits); // bits must not exceed data_t bitsize
		assert(dstrows <= bits);
		assert(srcrows <= bits);
		if (bits < 4)
			bits = 4;
		if (bits > 8*sizeof(data_t)) throw;

		// mask of lower half bits
		data_t m = (data_t(1) << (bits/2))-1;
		unsigned int j = (bits/2);
		data_t tmp[8*sizeof(data_t)];

		// first loop iteration, load src store in tmp
		const data_t* src2 = src + ((bits/2)*srcstride);
		for (unsigned int k=0;  k<bits/2;  ++k)
		{
			if (k < srcrows)
			{
				data_t a = *src, b = 0;
				src += srcstride;
				if ((k+(bits/2)) < srcrows)
				{
					b = *src2;
					src2 += srcstride;
				}
				data_t t = (b ^ (a >> (bits/2))) & m;
				tmp[k] = a ^ (t << (bits/2));
				tmp[k+(bits/2)] = b ^ t;
			}
			else
			{
				tmp[k] = 0;
				tmp[k+(bits/2)] = 0;
			}
		}
		j>>=1; m^=m<<j;
		// main loop
		for (;  1 != j;  j>>=1,m^=m<<j)
		{
			for (unsigned l=0,k=0;  l<bits/2;  ++l)
			{
				data_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
				tmp[k] ^= t<<j;
				tmp[k+j] ^= t;
				k=(k+j+1)&~j;
			}
		}
		// last loop iteration (j==1), load tmp store in dst
		unsigned int k=0;
		for (;  k+1 < dstrows;  k += 2)
		{
			data_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
			*dst = tmp[k] ^ (t<<1); dst+=dststride;
			*dst = tmp[k+1] ^ t; dst+=dststride;
		}
		if (k < dstrows)
		{
			data_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
			*dst = tmp[k] ^ (t<<1);
		}
	}



	template<typename data_t>
	inline void matrix_transpose(matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src)
	{
		static const size_t bits = sizeof(data_t)*8;
		if (dst.columns != src.rows || dst.rows != src.columns)
			throw std::runtime_error("matrix_transpose: matrix sizes do not match");
		// process batch of bits rows
		size_t r = 0;
		for (; r+bits <= src.rows; r += bits)
		{
			// process block of bits columns
			size_t c = 0;
			for (; c+2*bits <= src.columns; c += 2*bits)
			{
				block_transpose2(dst.data(c,r), dst.stride, src.data(r,c), src.stride);
			}
			if (c+bits <= src.columns)
			{
				block_transpose(dst.data(c,r), dst.stride, src.data(r,c), src.stride);
				c += bits;
			}
			// process block of partial C columns
			if (c < src.columns)
			{
				block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, bits);
			}
		}
		// process last rows
		if (r < src.rows)
		{
			size_t c = 0;
			for (; c+bits <= src.columns; c += bits)
			{
				block_transpose(dst.data(c,r), dst.stride, bits, src.data(r,c), src.stride, (src.rows % bits));
			}
			// process final bits x C submatrix
			if (c < src.columns)
			{
				size_t partialbits = next_pow2<uint32_t>(std::max(src.columns % bits, src.rows % bits));
				if (partialbits == bits)
					block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, (src.rows % bits));
				else
					block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, (src.rows % bits), partialbits);
			}
		}
	}




	template<typename data_t>
	inline size_t hammingweight(const data_t* first, const data_t* last)
	{
		size_t w = 0;
		for (; first != last; ++first)
			w += hammingweight(*first);
		return w;
	}
	template<typename data_t> 
	inline size_t hammingweight(const data_t* first, const data_t* last, data_t mask)
	{
		size_t w = 0;
		for (; first != last; ++first)
			w += hammingweight(mask & *first);
		return w;
	}
	template<typename data_t> inline size_t hammingweight_and(const data_t* first1, const data_t* last1, const data_t* first2)
	{
		size_t w = 0;
		for (; first1 != last1; ++first1,++first2)
			w += hammingweight(*first1 & *first2);
		return w;
	}
	template<typename data_t> inline size_t hammingweight_xor(const data_t* first1, const data_t* last1, const data_t* first2)
	{
		size_t w = 0;
		for (; first1 != last1; ++first1,++first2)
			w += hammingweight(*first1 ^ *first2);
		return w;
	}
	template<typename data_t> inline size_t hammingweight_or (const data_t* first1, const data_t* last1, const data_t* first2)
	{
		size_t w = 0;
		for (; first1 != last1; ++first1,++first2)
			w += hammingweight(*first1 | *first2);
		return w;
	}

	template<typename data_t>
	inline size_t matrix_hammingweight(const matrix_base_ref_t<const data_t>& m)
	{
		size_t words = (m.columns + m.word_bits - 1) / m.word_bits;
		if ((m.columns % m.word_bits) == 0)
		{
			size_t w = 0;
			for (size_t r = 0; r < m.rows; ++r)
			{
				const data_t* first = m.data(r);
				const data_t* last = m.data(r) + words;
				for (; first != last; ++first)
					w += hammingweight(*first);
			}
			return w;
		}
		else
		{
			--words;
			const data_t lastwordmask = m.lastwordmask();
			size_t w = 0;
			for (size_t r = 0; r < m.rows; ++r)
			{
				const data_t* first = m.data(r);
				const data_t* last = m.data(r) + words;
				for (; first != last; ++first)
					w += hammingweight(*first);
				w += hammingweight(*first & lastwordmask);
			}
			return w;
		}
	}
	template<typename data_t>
	inline size_t vector_hammingweight(const matrix_base_ref_t<const data_t>& m)
	{
		size_t words = (m.columns + m.word_bits - 1) / m.word_bits;
		if ((m.columns % m.word_bits) == 0)
		{
			size_t w = 0;
			const data_t* first = m.data(0);
			const data_t* last = m.data(0) + words;
			for (; first != last; ++first)
				w += hammingweight(*first);
			return w;
		}
		else
		{
			--words;
			const data_t lastwordmask = m.lastwordmask();
			size_t w = 0;
			const data_t* first = m.data(0);
			const data_t* last = m.data(0) + words;
			for (; first != last; ++first)
				w += hammingweight(*first);
			w += hammingweight(*first & lastwordmask);
			return w;
		}
	}
	template<typename data_t>
	inline size_t vector_hammingweight_and(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2)
	{
		if (m1.columns != m2.columns) \
			throw std::runtime_error("vector_weight_and: vector sizes don't match"); \
		size_t words = (m1.columns + m1.word_bits - 1) / m1.word_bits;
		if ((m1.columns % m1.word_bits) == 0)
		{
			size_t w = 0;
			const data_t* first1 = m1.data(0);
			const data_t* first2 = m2.data(0);
			const data_t* last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1,++first2)
				w += hammingweight(*first1 & *first2);
			return w;
		}
		else
		{
			--words;
			const data_t lastwordmask = m1.lastwordmask();
			size_t w = 0;
			const data_t* first1 = m1.data(0);
			const data_t* first2 = m2.data(0);
			const data_t* last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1,++first2)
				w += hammingweight(*first1 & *first2);
			w += hammingweight((*first1 & *first2) & lastwordmask);
			return w;
		}
	}
	template<typename data_t>
	inline size_t vector_hammingweight_xor(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2)
	{
		if (m1.columns != m2.columns) \
			throw std::runtime_error("vector_weight_xor: vector sizes don't match"); \
		size_t words = (m1.columns + m1.word_bits - 1) / m1.word_bits;
		if ((m1.columns % m1.word_bits) == 0)
		{
			size_t w = 0;
			const data_t* first1 = m1.data(0);
			const data_t* first2 = m2.data(0);
			const data_t* last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1,++first2)
				w += hammingweight(*first1 ^ *first2);
			return w;
		}
		else
		{
			--words;
			const data_t lastwordmask = m1.lastwordmask();
			size_t w = 0;
			const data_t* first1 = m1.data(0);
			const data_t* first2 = m2.data(0);
			const data_t* last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1,++first2)
				w += hammingweight(*first1 & *first2);
			w += hammingweight((*first1 ^ *first2) & lastwordmask);
			return w;
		}
	}
	template<typename data_t>
	inline size_t vector_hammingweight_or (const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2)
	{
		if (m1.columns != m2.columns) \
			throw std::runtime_error("vector_weight_or: vector sizes don't match"); \
		size_t words = (m1.columns + m1.word_bits - 1) / m1.word_bits;
		if ((m1.columns % m1.word_bits) == 0)
		{
			size_t w = 0;
			const data_t* first1 = m1.data(0);
			const data_t* first2 = m2.data(0);
			const data_t* last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1,++first2)
				w += hammingweight(*first1 | *first2);
			return w;
		}
		else
		{
			--words;
			const data_t lastwordmask = m1.lastwordmask();
			size_t w = 0;
			const data_t* first1 = m1.data(0);
			const data_t* first2 = m2.data(0);
			const data_t* last1 = m1.data(0) + words;
			for (; first1 != last1; ++first1,++first2)
				w += hammingweight(*first1 & *first2);
			w += hammingweight((*first1 | *first2) & lastwordmask);
			return w;
		}
	}

	template<typename data_t>
	inline void vector_swap(data_t* first1, data_t* last1, data_t* first2)
	{
		for (; first1 != last1; ++first1,++first2)
		{
			data_t tmp = *first1;
			*first1 = *first2;
			*first2 = tmp;
		}
	}

} // namespace detail

MCCL_END_NAMESPACE
