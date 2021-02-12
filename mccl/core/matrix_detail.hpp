#ifndef MCCL_CORE_MATRIX_DETAIL_HPP
#define MCCL_CORE_MATRIX_DETAIL_HPP

#include <mccl/config/config.hpp>

#include <nmmintrin.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <string>

//#define MCCL_MATRIX_BASE_ASSERT(s) assert(s);
#define MCCL_MATRIX_BASE_ASSERT(s) if (!(s)) throw std::runtime_error("matrix_base_ref_t: assert throw");
//#define MCCL_MATRIX_BASE_ASSERT(s)


MCCL_BEGIN_NAMESPACE

namespace detail
{

	// return the lowest bit position belonging to a `1'-bit == the number of trailing zeroes
	unsigned int trailing_zeroes(uint32_t n);
	// return the smallest 2^t >= n
	template<typename Int>
	inline Int next_pow2(Int n)
	{
		static const unsigned bits = sizeof(Int)*8;
		--n;
		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		if (8 < bits) n |= n >> 8;
		if (16 < bits) n |= n >> 16;
		if (32 < bits) n |= n >> 32;
		return ++n;
	}
	inline size_t hammingweight(uint64_t n)
	{
		return _mm_popcnt_u64(n);
	}
	inline size_t hammingweight(uint32_t n)
	{
		return _mm_popcnt_u32(n);
	}


	// allow more efficient overloads for simd vectors
	template<typename data_t>
	inline bool get_bit(const data_t& v, unsigned int pos)
	{
		return (v >> pos) & 1;
	}
	template<typename data_t>
	inline void set_bit(data_t& v, unsigned int pos)
	{
		v |= data_t(1) << pos;
	}
	template<typename data_t>
	inline void reset_bit(data_t& v, unsigned int pos)
	{
		v &= ~(data_t(1) << pos);
	}
	template<typename data_t>
	inline void flip_bit(data_t& v, unsigned int pos)
	{
		v ^= data_t(1) << pos;
	}

	/* matrix_base_ref_t: a struct containing a (sub)matrix description 
	* data_t: the underlying base word type. 
	*         Columns must start at base word alignment
	*         Total number of columns (incl scratchcolumns) must be a whole number of words.
	*         the expression `data_t(-1)` should efficiently convert to all `1`-bits 
	*/
	template<typename data_t>
	struct matrix_base_ref_t
	{
		static const size_t word_bytes = sizeof(data_t);
		static const size_t word_bits = sizeof(data_t) * 8;
		static const size_t bit_alignment = word_bits;
		static const size_t byte_alignment = bit_alignment / 8;
		static const data_t _datazero = 0; // all zeroes
		static const data_t _dataones = -1; // all ones
		static const data_t _dataone = 1; // only a one at bit position 0

		data_t* ptr;
		size_t rows, columns, scratchcolumns;
		size_t stride; // with unit data_t

		matrix_base_ref_t(): ptr(nullptr), rows(0), columns(0), scratchcolumns(0), stride(0) {}
		matrix_base_ref_t(const matrix_base_ref_t&  m) = default;
		matrix_base_ref_t(      matrix_base_ref_t&& m) = default;
		matrix_base_ref_t(data_t* _ptr, size_t _rows, size_t _columns, size_t _scratchcolumns, size_t _stride)
			: ptr(_ptr), rows(_rows), columns(_columns), scratchcolumns(_scratchcolumns), stride(_stride)
		{
			assert_bit_alignment();
		}

		matrix_base_ref_t& operator=(const matrix_base_ref_t&  m) = default;
		matrix_base_ref_t& operator=(      matrix_base_ref_t&& m) = default;

		bool operator==(const matrix_base_ref_t& m) const
		{
			return ptr == m.ptr && rows == m.rows && columns == m.columns && scratchcolumns == m.scratchcolumns && stride == m.stride;
		}
		bool operator!=(const matrix_base_ref_t& m) const
		{
			return !(*this == m);
		}

		// obtain pointer to the first word of the first row
		data_t* data()                   const { return ptr; }
		// obtain pointer to the first word of row r
		data_t* data(size_t r)           const { return ptr + r * stride; }
		// obtain pointer to the word containing value of row r column c
		data_t* data(size_t r, size_t c) const { return ptr + r * stride + (c / word_bits); }
		// obtain word mask for column c
		static data_t wordmaskbit(size_t c) { return _dataone << (c % word_bits); }
		// obtain word mask for column c and higher
		static data_t wordmaskhigh(size_t c) { return _dataones << (c % word_bits); }
		// obtain word mask for column c and lower
		static data_t wordmasklow(size_t c) { return _dataones >> ((word_bits - 1 - c) % word_bits); }		
		// obtain word mask for column c
		data_t lastwordmask() const { return columns % word_bits == 0 ? _dataones : ~wordmaskhigh(columns); }

		bool operator()(size_t r, size_t c) const { return get_bit<data_t>(*data(r,c), c % word_bits);  }
		
		void bitset(size_t r, size_t c) { set_bit<data_t>(*data(r,c), c % word_bits); }
		void bitreset(size_t r, size_t c) { reset_bit<data_t>(*data(r,c), c % word_bits); }
		void bitflip(size_t r, size_t c) { flip_bit<data_t>(*data(r,c), c % word_bits); }
		void bitset(size_t r, size_t c, bool b)
		{
			if (b)
				set_bit<data_t>(*data(r,c), c % word_bits);
			else
				reset_bit<data_t>(*data(r,c), c % word_bits);
		}

		/* change the dividing line between columns and scratchcolumns, but do not change the sum */
		void reset_columns(size_t _columns, size_t _scratchcolumns)
		{
			if (_columns + _scratchcolumns != columns + scratchcolumns)
				std::runtime_error("matrix_base_ref_t: sum of columns and scratchcolumns must be equal");
			scratchcolumns = _scratchcolumns;
			columns = _columns;
		}
		void reset(data_t* _ptr, size_t _rows, size_t _columns, size_t _scratchcolumns, size_t _stride)
		{
			ptr = _ptr;
			rows = _rows;
			columns = _columns;
			scratchcolumns = _scratchcolumns;
			stride = _stride;
			assert_bit_alignment();
		}
		void reset_submatrix(size_t _row_offset, size_t _rows, size_t _column_offset, size_t _columns, size_t _scratchcolumns = 0)
		{
			MCCL_MATRIX_BASE_ASSERT(_row_offset + _rows <= rows);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns <= columns);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns + _scratchcolumns <= columns + scratchcolumns);
			ptr += _row_offset * stride + (_column_offset / word_bits);
			rows = _rows;
			columns = _columns;
			scratchcolumns = _scratchcolumns;
			assert_bit_alignment();
		}
		void reset_subvector(size_t _row_offset, size_t _column_offset, size_t _columns, size_t _scratchcolumns = 0)
		{
			const size_t _rows = 1;
			MCCL_MATRIX_BASE_ASSERT(_row_offset + _rows <= rows);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns <= columns);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns + _scratchcolumns <= columns + scratchcolumns);
			ptr += _row_offset * stride + (_column_offset / word_bits);
			rows = _rows;
			columns = _columns;
			scratchcolumns = _scratchcolumns;
			assert_bit_alignment();
		}
		matrix_base_ref_t submatrix(size_t _row_offset, size_t _rows, size_t _column_offset, size_t _columns, size_t _scratchcolumns = 0) const
		{
			MCCL_MATRIX_BASE_ASSERT(_row_offset + _rows <= rows);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns <= columns);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns + _scratchcolumns <= columns + scratchcolumns);
			return matrix_base_ref_t(ptr + _row_offset * stride + (_column_offset / word_bits), _rows, _columns, _scratchcolumns, stride);
		}
		matrix_base_ref_t subvector(size_t _row_offset, size_t _column_offset, size_t _columns, size_t _scratchcolumns = 0) const
		{
			const size_t _rows = 1;
			MCCL_MATRIX_BASE_ASSERT(_row_offset + _rows <= rows);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns <= columns);
			MCCL_MATRIX_BASE_ASSERT(_column_offset + _columns + _scratchcolumns <= columns + scratchcolumns);
			return matrix_base_ref_t(ptr + _row_offset * stride + (_column_offset / word_bits), _rows, _columns, _scratchcolumns, stride);
		}

		/* automatic conversion to const data_t versions */
		typedef const data_t cdata_t;
		typedef matrix_base_ref_t<cdata_t> cmatrix_base_ref_t;
		cmatrix_base_ref_t& as_const() { return *reinterpret_cast<cmatrix_base_ref_t*>(this); }
		const cmatrix_base_ref_t& as_const() const { return *reinterpret_cast<const cmatrix_base_ref_t*>(this); }
		operator cmatrix_base_ref_t& () { return as_const(); }
		operator const cmatrix_base_ref_t& () const { return as_const(); }

		/* alignment checks */
		bool has_bit_alignment(size_t alignment = bit_alignment) const
		{
			bool ok = ((uintptr_t(ptr) * 8) % alignment == 0);
			ok &= ((columns + scratchcolumns) % alignment == 0);
			ok &= ((stride * word_bits) % alignment == 0);
			return ok;
		}
		inline void assert_bit_alignment() const
		{
			MCCL_MATRIX_BASE_ASSERT(has_bit_alignment());
		}
		size_t max_bit_alignment() const
		{
			size_t maxalign = 512;
			maxalign = 1ULL << trailing_zeroes((uintptr_t(ptr) * 8) | (stride * word_bits) | maxalign);
			while (true)
			{
				size_t residue_bits = columns % maxalign;
				if (residue_bits == 0 || residue_bits + scratchcolumns >= maxalign)
					return maxalign;
				maxalign >>= 1;
			}
		}

		/* copy as a larger data_t type, not allowed to remove const */
		template<typename T>
		matrix_base_ref_t<T> to_data() const
		{
			static_assert(std::is_const<T>::value == std::is_const<data_t>::value, "not allowed to remove const");
			MCCL_MATRIX_BASE_ASSERT((stride * sizeof(data_t)) % sizeof(T) == 0);
			return matrix_base_ref_t<T>(reinterpret_cast<T*>(ptr), rows, columns, scratchcolumns, (stride * sizeof(data_t)) / sizeof(T));
		}
	};

	template<typename data_t>
	void matrix_print(std::ostream& o, const matrix_base_ref_t<data_t>& m, bool transpose = false);
	template<typename data_t>
	void vector_print(std::ostream& o, const matrix_base_ref_t<data_t>& m);

	// return true if the matrices m1 and m2 are equal (same dimensions & same content, scratchcolumns are ignored)
	template<typename data_t>
	inline bool matrix_compare(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2);
	// same as matrix_compare, but assume rows == 1
	template<typename data_t>
	inline bool vector_compare(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2);

	// fill matrix with given boolean value (for every bit) or data_t value
	template<typename data_t>
	inline void matrix_set(const matrix_base_ref_t<data_t>& m, const data_t& value);
	template<typename data_t>
	inline void matrix_set(const matrix_base_ref_t<data_t>& m, bool b);
	// same as matrix_set, but assume rows == 1
	template<typename data_t>
	inline void vector_set(const matrix_base_ref_t<data_t>& m, const data_t& value);
	template<typename data_t>
	inline void vector_set(const matrix_base_ref_t<data_t>& m, bool b);

	template<typename data_t>
	inline void matrix_setcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns, bool b);
	template<typename data_t>
	inline void vector_setcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns, bool b);
	template<typename data_t>
	inline void matrix_setscratch(matrix_base_ref_t<data_t>& m, bool b = false)
	{
		matrix_setcolumns(m, m.columns, m.scratchcolumns, b);
	}
	template<typename data_t>
	inline void vector_setscratch(matrix_base_ref_t<data_t>& m, bool b = false)
	{
		vector_setcolumns(m, m.columns, m.scratchcolumns, b);
	}
	template<typename data_t>
	inline void matrix_flipcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns);
	template<typename data_t>
	inline void vector_flipcolumns(matrix_base_ref_t<data_t>& m, size_t column_offset, size_t columns);

	// flip every bit (scratchcolumns may be partially affected)
	template<typename data_t>
	inline void matrix_not(const matrix_base_ref_t<data_t>& m);
	template<typename data_t>
	inline void vector_not(const matrix_base_ref_t<data_t>& m);

	// block transpose of bits x bits bit matrix (i.e., dst and src are data_t[bits] arrays)
	// bits must be a power of 2
	template<typename data_t, size_t bits = sizeof(data_t)*8>
	inline void block_transpose(data_t* dst, size_t dststride, const data_t* src, size_t srcstride);
	// variant that does two blocks simultaneously (src = (A B) => dst = (A B)^T)
	template<typename data_t, size_t bits = sizeof(data_t)*8>
	inline void block_transpose2(data_t* dst, size_t dststride, const data_t* src, size_t srcstride);
	// specialization for partial matrix
	template<typename data_t, size_t bits = sizeof(data_t)*8>
	inline void block_transpose(data_t* dst, size_t dststride, size_t dstrows, const data_t* src, size_t srcstride, size_t srcrows);
	template<typename data_t>
	inline void block_transpose(data_t* dst, size_t dststride, size_t dstrows, const data_t* src, size_t srcstride, size_t srcrows, size_t bits);

	template<typename data_t>
	inline void matrix_transpose(matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src);



	// hamming weight
	template<typename data_t>
	inline size_t matrix_hammingweight(const matrix_base_ref_t<const data_t>& m);
	template<typename data_t>
	inline size_t vector_hammingweight(const matrix_base_ref_t<const data_t>& m);
	template<typename data_t>
	inline size_t vector_hammingweight_and(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2);
	template<typename data_t>
	inline size_t vector_hammingweight_xor(const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2);
	template<typename data_t>
	inline size_t vector_hammingweight_or (const matrix_base_ref_t<const data_t>& m1, const matrix_base_ref_t<const data_t>& m2);



	/* binary boolean functions

	ab ab ab ab : a=dst, b=src
	00 01 10 11
	 0  0  0  0 = set 0            => set(0)
	 0  0  0  1 = a and b          => and
	 0  0  1  0 = a and not b      => andin
	 0  0  1  1 = null op
	 0  1  0  0 = not a and b      => andni
	 0  1  0  1 = copy b           => copy
	 0  1  1  0 = a xor b          => xor
	 0  1  1  1 = a or b           => or
	 1  0  0  0 = not a and not b  => nor
	 1  0  0  1 = not (a xor b)    => nxor
	 1  0  1  0 = not b            => copy_not
	 1  0  1  1 = a or not b       => orin
	 1  1  0  0 = not a            => matrix_not
	 1  1  0  1 = not a or b       => orni
	 1  1  1  0 = not (a and b)    => nand
	 1  1  1  1 = set 1            => set(1)
	 */

	template<typename data_t> inline void copy_not    (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_and  (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_xor  (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_or   (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_nand (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_nxor (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_nor  (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_andin(const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_andni(const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_orin (const data_t* first, const data_t* last, data_t* dst);
	template<typename data_t> inline void binary_orni (const data_t* first, const data_t* last, data_t* dst);

	template<typename data_t> inline void binary_and  (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_xor  (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_or   (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_nand (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_nxor (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_nor  (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_andin(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_andni(const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_orin (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);
	template<typename data_t> inline void binary_orni (const data_t* first1, const data_t* last1, const data_t* first2, data_t* dst);


	// apply boolean function to (dst,src) and store result in dst
#define DEFINE_MATRIX_OP2(matrixfunc) \
	template<typename data_t> \
	inline void matrixfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src);

	DEFINE_MATRIX_OP2(matrix_copy)    // dst = src
	DEFINE_MATRIX_OP2(matrix_copynot) // dst = ~src
	DEFINE_MATRIX_OP2(matrix_and)
	DEFINE_MATRIX_OP2(matrix_xor)
	DEFINE_MATRIX_OP2(matrix_or)
	DEFINE_MATRIX_OP2(matrix_nand)
	DEFINE_MATRIX_OP2(matrix_nxor)
	DEFINE_MATRIX_OP2(matrix_nor)
	DEFINE_MATRIX_OP2(matrix_andin)
	DEFINE_MATRIX_OP2(matrix_andni)
	DEFINE_MATRIX_OP2(matrix_orin)
	DEFINE_MATRIX_OP2(matrix_orni)

#define DEFINE_MATRIX_OP3(matrixfunc) \
	template<typename data_t> \
	inline void matrixfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src1, const matrix_base_ref_t<const data_t>& src2);

	DEFINE_MATRIX_OP3(matrix_and)   // dst = src1 & src2
	DEFINE_MATRIX_OP3(matrix_xor)
	DEFINE_MATRIX_OP3(matrix_or)
	DEFINE_MATRIX_OP3(matrix_nand)
	DEFINE_MATRIX_OP3(matrix_nxor)
	DEFINE_MATRIX_OP3(matrix_nor)
	DEFINE_MATRIX_OP3(matrix_andin)
	DEFINE_MATRIX_OP3(matrix_andni)
	DEFINE_MATRIX_OP3(matrix_orin)
	DEFINE_MATRIX_OP3(matrix_orni)


		// same as above, but assume rows == 1
#define DEFINE_VECTOR_OP2(vectorfunc) \
	template<typename data_t> \
	inline void vectorfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src);

	DEFINE_VECTOR_OP2(vector_copy)
	DEFINE_VECTOR_OP2(vector_copynot)
	DEFINE_VECTOR_OP2(vector_and)
	DEFINE_VECTOR_OP2(vector_xor)
	DEFINE_VECTOR_OP2(vector_or)
	DEFINE_VECTOR_OP2(vector_nand)
	DEFINE_VECTOR_OP2(vector_nxor)
	DEFINE_VECTOR_OP2(vector_nor)
	DEFINE_VECTOR_OP2(vector_andin)
	DEFINE_VECTOR_OP2(vector_andni)
	DEFINE_VECTOR_OP2(vector_orin)
	DEFINE_VECTOR_OP2(vector_orni)

#define DEFINE_VECTOR_OP3(vectorfunc) \
	template<typename data_t> \
	inline void vectorfunc (matrix_base_ref_t<data_t>& dst, const matrix_base_ref_t<const data_t>& src1, const matrix_base_ref_t<const data_t>& src2);

	DEFINE_VECTOR_OP3(vector_copy)
	DEFINE_VECTOR_OP3(vector_copynot)
	DEFINE_VECTOR_OP3(vector_and)
	DEFINE_VECTOR_OP3(vector_xor)
	DEFINE_VECTOR_OP3(vector_or)
	DEFINE_VECTOR_OP3(vector_nand)
	DEFINE_VECTOR_OP3(vector_nxor)
	DEFINE_VECTOR_OP3(vector_nor)
	DEFINE_VECTOR_OP3(vector_andin)
	DEFINE_VECTOR_OP3(vector_andni)
	DEFINE_VECTOR_OP3(vector_orin)
	DEFINE_VECTOR_OP3(vector_orni)

} // namespace detail

MCCL_END_NAMESPACE

#include "matrix_detail.inl"

#endif
