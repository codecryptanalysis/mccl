#ifndef MCCL_CORE_MATRIX_BASE_HPP
#define MCCL_CORE_MATRIX_BASE_HPP

#include <mccl/config/config.hpp>

#include <nmmintrin.h>
#include <stdexcept>
#include <cstdint>
#include <array>


MCCL_BEGIN_NAMESPACE

using std::uint64_t;
using std::size_t;
using std::ptrdiff_t;

inline size_t hammingweight(uint64_t n)
{
	return __builtin_popcountll(n);
}

inline size_t hammingweight(uint32_t n)
{
	return __builtin_popcountl(n);
}


// bit vectors, vector iterators and matrices with 64 packed bits in a 64-bit machine word uint64_t

// pointer to (const) vector: (word)ptr, (bit)columns
struct cv_ptr;
struct v_ptr;

// pointer to (const) vector iterator: (word)ptr, (bit)columns, (word)stride
struct cvi_ptr;
struct vi_ptr;

// pointer to (const) matrix: (word)ptr, (bit)columns, (word)stride, rows
struct cm_ptr;
struct m_ptr;

/* automatic type conversion diagram

                non-const         const
	
matrix          m_ptr       -->   cm_ptr


                  
vector iter     vi_ptr      -->   cvi_ptr

                  |                 |
                  v                 v

vector          v_ptr       -->   cv_ptr

*/

struct cv_ptr
{
	typedef uint64_t value_type;
	typedef const value_type* pointer_type;
	
	pointer_type ptr;
	size_t columns;

	cv_ptr()
		: ptr(nullptr), columns(0)
	{}
	cv_ptr(pointer_type _ptr, size_t _columns)
		: ptr(_ptr), columns(_columns)
	{}
	cv_ptr(const cv_ptr&  ) = default;
	cv_ptr(      cv_ptr&& ) = default;
	cv_ptr& operator=(const cv_ptr&  ) = default;
	cv_ptr& operator=(      cv_ptr&& ) = default;

	pointer_type data()         const { return ptr; }
	pointer_type data(size_t c) const { return ptr+(c/64); }
	
	void reset(pointer_type _ptr, size_t _columns)
	{
		ptr = _ptr;
		columns = _columns;
	}
	cv_ptr subvector(size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("cv_ptr::subvector(): columns out of range");
#endif
		return cv_ptr(ptr + (coloffset/64), cols);
	}
};

struct v_ptr
{
	typedef uint64_t value_type;
	typedef value_type* pointer_type;
	
	pointer_type ptr;
	size_t columns;
	
	v_ptr()
		: ptr(nullptr), columns(0)
	{}
	v_ptr(pointer_type _ptr, size_t _columns)
		: ptr(_ptr), columns(_columns)
	{}
	v_ptr(const v_ptr&  ) = default;
	v_ptr(      v_ptr&& ) = default;
	v_ptr& operator=(const v_ptr&  ) = default;
	v_ptr& operator=(      v_ptr&& ) = default;
	
	operator       cv_ptr&()       { return *reinterpret_cast<      cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }

	pointer_type data()         const { return ptr; }
	pointer_type data(size_t c) const { return ptr+(c/64); }

	void reset(pointer_type _ptr, size_t _columns)
	{
		ptr = _ptr;
		columns = _columns;
	}
	v_ptr subvector(size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("v_ptr::subvector(): columns out of range");
#endif
		return v_ptr(ptr + (coloffset/64), cols);
	}
};


struct cvi_ptr
{
	typedef uint64_t value_type;
	typedef const value_type* pointer_type;
	
	pointer_type ptr;
	size_t columns, stride;

	cvi_ptr()
		: ptr(nullptr), columns(0), stride(0)
	{}
	cvi_ptr(pointer_type _ptr, size_t _columns, size_t _stride)
		: ptr(_ptr), columns(_columns), stride(_stride)
	{}
	cvi_ptr(const cvi_ptr&  ) = default;
	cvi_ptr(      cvi_ptr&& ) = default;
	cvi_ptr& operator=(const cvi_ptr&  ) = default;
	cvi_ptr& operator=(      cvi_ptr&& ) = default;

	operator       cv_ptr&()       { return *reinterpret_cast<      cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }

	pointer_type data()         const { return ptr; }
	pointer_type data(size_t c) const { return ptr+(c/64); }

	void reset(pointer_type _ptr, size_t _columns, size_t _stride)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
	}
	cv_ptr subvector(size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("cvi_ptr::subvector(): columns out of range");
#endif
		return cv_ptr(ptr + (coloffset/64), cols);
	}
	cvi_ptr subvectorit(size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("cvi_ptr::subvectorit(): columns out of range");
#endif
		return cvi_ptr(ptr + (coloffset/64), cols, stride);
	}

	cvi_ptr& operator++()         { ptr += stride; return *this; }
	cvi_ptr& operator--()         { ptr -= stride; return *this; }
	cvi_ptr& operator+=(size_t n) { ptr += n*stride; return *this; }
	cvi_ptr& operator-=(size_t n) { ptr -= n*stride; return *this; }
	cvi_ptr  operator++(int)      { ptr += stride; return cvi_ptr(ptr-stride,columns,stride); }
	cvi_ptr  operator--(int)      { ptr -= stride; return cvi_ptr(ptr+stride,columns,stride); }

	cvi_ptr   operator+(size_t n) const { return cvi_ptr(ptr + n*stride,columns,stride); }
	cvi_ptr   operator-(size_t n) const { return cvi_ptr(ptr - n*stride,columns,stride); }
	ptrdiff_t operator-(const cvi_ptr& r) const { return (ptr-r.ptr)/ptrdiff_t(stride); }
};

struct vi_ptr
{
	typedef uint64_t value_type;
	typedef value_type* pointer_type;
	
	pointer_type ptr;
	size_t columns, stride;
	
	vi_ptr()
		: ptr(nullptr), columns(0)
	{}
	vi_ptr(pointer_type _ptr, size_t _columns, size_t _stride)
		: ptr(_ptr), columns(_columns), stride(_stride)
	{}
	vi_ptr(const vi_ptr&  ) = default;
	vi_ptr(      vi_ptr&& ) = default;
	vi_ptr& operator=(const vi_ptr&  ) = default;
	vi_ptr& operator=(      vi_ptr&& ) = default;
	
	operator       cvi_ptr&()       { return *reinterpret_cast<      cvi_ptr*>(this); }
	operator const cvi_ptr&() const { return *reinterpret_cast<const cvi_ptr*>(this); }
	operator       cv_ptr&()        { return *reinterpret_cast<      cv_ptr*>(this); }
	operator const cv_ptr&()  const { return *reinterpret_cast<const cv_ptr*>(this); }
	operator       v_ptr&()         { return *reinterpret_cast<      v_ptr*>(this); }
	operator const v_ptr&()   const { return *reinterpret_cast<const v_ptr*>(this); }

	pointer_type data()         const { return ptr; }
	pointer_type data(size_t c) const { return ptr+(c/64); }

	void reset(pointer_type _ptr, size_t _columns, size_t _stride)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
	}
	v_ptr subvector(size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("vi_ptr::subvector(): columns out of range");
#endif
		return v_ptr(ptr + (coloffset/64), cols);
	}
	vi_ptr subvectorit(size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("vi_ptr::subvectorit(): columns out of range");
#endif
		return vi_ptr(ptr + (coloffset/64), cols, stride);
	}

	vi_ptr& operator++()         { ptr += stride; return *this; }
	vi_ptr& operator--()         { ptr -= stride; return *this; }
	vi_ptr& operator+=(size_t n) { ptr += n*stride; return *this; }
	vi_ptr& operator-=(size_t n) { ptr -= n*stride; return *this; }
	vi_ptr  operator++(int)      { ptr += stride; return vi_ptr(ptr-stride,columns,stride); }
	vi_ptr  operator--(int)      { ptr -= stride; return vi_ptr(ptr+stride,columns,stride); }
	
	vi_ptr operator+(size_t n) const { return vi_ptr(ptr + n*stride,columns,stride); }
	vi_ptr operator-(size_t n) const { return vi_ptr(ptr - n*stride,columns,stride); }
	ptrdiff_t operator-(const cvi_ptr& r) const { return (ptr-r.ptr)/ptrdiff_t(stride); }
};


struct cm_ptr
{
	typedef uint64_t value_type;
	typedef const value_type* pointer_type;
	
	pointer_type ptr;
	size_t columns, stride, rows;
	
	cm_ptr()
		: ptr(nullptr), columns(0), stride(0), rows(0)
	{}
	cm_ptr(pointer_type _ptr, size_t _columns, size_t _stride, size_t _rows)
		: ptr(_ptr), columns(_columns), stride(_stride), rows(_rows)
	{}
	cm_ptr(const cm_ptr&  ) = default;
	cm_ptr(      cm_ptr&& ) = default;
	cm_ptr& operator=(const cm_ptr&  ) = default;
	cm_ptr& operator=(      cm_ptr&& ) = default;
	
	pointer_type data()                   const { return ptr; }
	pointer_type data(size_t r)           const { return ptr+r*stride; }
	pointer_type data(size_t r, size_t c) const { return ptr + r*stride + (c/64); }

	void reset(pointer_type _ptr, size_t _columns, size_t _stride, size_t _rows)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
		rows = _rows;
	}
	cv_ptr subvector(size_t rowoffset = 0) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if (rowoffset >= rows) throw std::out_of_range("cm_ptr::subvector(): columns out of range");
#endif
		return cv_ptr(ptr + (rowoffset*stride), columns);
	}
	cv_ptr subvector(size_t rowoffset, size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("cm_ptr::subvector(): columns out of range");
#endif
		return cv_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols);
	}
	cvi_ptr subvectorit(size_t rowoffset = 0) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if (rowoffset >= rows) throw std::out_of_range("cm_ptr::subvectorit(): columns out of range");
#endif
		return cvi_ptr(ptr + (rowoffset*stride), columns, stride);
	}
	cvi_ptr subvectorit(size_t rowoffset, size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("cm_ptr::subvectorit(): columns out of range");
#endif
		return cvi_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride);
	}
	cm_ptr submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset+_rows>rows)) throw std::out_of_range("cm_ptr::submatrix(): columns and/or rows out of range");
#endif
		return cm_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride, _rows);
	}
};

struct m_ptr
{
	typedef uint64_t value_type;
	typedef value_type* pointer_type;
	
	pointer_type ptr;
	size_t columns, stride, rows;
	
	m_ptr()
		: ptr(nullptr), columns(0), stride(0), rows(0)
	{}
	m_ptr(pointer_type _ptr, size_t _columns, size_t _stride, size_t _rows)
		: ptr(_ptr), columns(_columns), stride(_stride), rows(_rows)
	{}
	m_ptr(const m_ptr&  ) = default;
	m_ptr(      m_ptr&& ) = default;
	m_ptr& operator=(const m_ptr&  ) = default;
	m_ptr& operator=(      m_ptr&& ) = default;
	
	operator       cm_ptr&()       { return *reinterpret_cast<      cm_ptr*>(this); }
	operator const cm_ptr&() const { return *reinterpret_cast<const cm_ptr*>(this); }

	pointer_type data()                   const { return ptr; }
	pointer_type data(size_t r)           const { return ptr+r*stride; }
	pointer_type data(size_t r, size_t c) const { return ptr + r*stride + (c/64); }

	void reset(pointer_type _ptr, size_t _columns, size_t _stride, size_t _rows)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
		rows = _rows;
	}
	v_ptr subvector(size_t rowoffset = 0) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if (rowoffset >= rows) throw std::out_of_range("m_ptr::subvector(): columns out of range");
#endif
		return v_ptr(ptr + (rowoffset*stride), columns);
	}
	v_ptr subvector(size_t rowoffset, size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("m_ptr::subvector(): columns out of range");
#endif
		return v_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols);
	}
	vi_ptr subvectorit(size_t rowoffset = 0) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if (rowoffset >= rows) throw std::out_of_range("m_ptr::subvectorit(): columns out of range");
#endif
		return vi_ptr(ptr + (rowoffset*stride), columns, stride);
	}
	vi_ptr subvectorit(size_t rowoffset, size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("m_ptr::subvectorit(): columns out of range");
#endif
		return vi_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride);
	}
	m_ptr submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const
	{
#ifndef MCCL_VECTOR_NO_SANITY_CHECKS
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset+_rows>rows)) throw std::out_of_range("m_ptr::submatrix(): columns and/or rows out of range");
#endif
		return m_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride, _rows);
	}
};

inline bool operator==(const cv_ptr& v1, const cv_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
inline bool operator!=(const cv_ptr& v1, const cv_ptr& v2) { return !(v1 == v2); }
inline bool operator==(const cv_ptr& v1, const  v_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
inline bool operator!=(const cv_ptr& v1, const  v_ptr& v2) { return !(v1 == v2); }
inline bool operator==(const  v_ptr& v1, const cv_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
inline bool operator!=(const  v_ptr& v1, const cv_ptr& v2) { return !(v1 == v2); }
inline bool operator==(const  v_ptr& v1, const  v_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
inline bool operator!=(const  v_ptr& v1, const  v_ptr& v2) { return !(v1 == v2); }

inline bool operator==(const cvi_ptr& v1, const cvi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
inline bool operator!=(const cvi_ptr& v1, const cvi_ptr& v2) { return !(v1 == v2); }
inline bool operator==(const cvi_ptr& v1, const  vi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
inline bool operator!=(const cvi_ptr& v1, const  vi_ptr& v2) { return !(v1 == v2); }
inline bool operator==(const  vi_ptr& v1, const cvi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
inline bool operator!=(const  vi_ptr& v1, const cvi_ptr& v2) { return !(v1 == v2); }
inline bool operator==(const  vi_ptr& v1, const  vi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
inline bool operator!=(const  vi_ptr& v1, const  vi_ptr& v2) { return !(v1 == v2); }

inline bool operator==(const cm_ptr& m1, const cm_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
inline bool operator!=(const cm_ptr& m1, const cm_ptr& m2) { return !(m1 == m2); }
inline bool operator==(const cm_ptr& m1, const  m_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
inline bool operator!=(const cm_ptr& m1, const  m_ptr& m2) { return !(m1 == m2); }
inline bool operator==(const  m_ptr& m1, const cm_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
inline bool operator!=(const  m_ptr& m1, const cm_ptr& m2) { return !(m1 == m2); }
inline bool operator==(const  m_ptr& m1, const  m_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
inline bool operator!=(const  m_ptr& m1, const  m_ptr& m2) { return !(m1 == m2); }




template<size_t Bits>
struct alignas(Bits/8) uint64_block_t
{
    static const size_t bits = Bits;
    static const size_t size = bits/64;
    
    std::array<uint64_t,size> v;
    
    uint64_block_t& operator&=(const uint64_block_t& v2) { for (size_t i = 0; i < size; ++i) v[i] &= v2.v[i]; return *this; }
    uint64_block_t& operator^=(const uint64_block_t& v2) { for (size_t i = 0; i < size; ++i) v[i] ^= v2.v[i]; return *this; }
    uint64_block_t& operator|=(const uint64_block_t& v2) { for (size_t i = 0; i < size; ++i) v[i] |= v2.v[i]; return *this; }
    uint64_block_t operator&(const uint64_block_t& v2) const { uint64_block_t tmp(*this); return tmp &= v2; }
    uint64_block_t operator^(const uint64_block_t& v2) const { uint64_block_t tmp(*this); return tmp ^= v2; }
    uint64_block_t operator|(const uint64_block_t& v2) const { uint64_block_t tmp(*this); return tmp |= v2; }
    uint64_block_t operator~() const { uint64_block_t tmp; for (size_t i = 0; i < size; ++i) tmp.v[i] = ~v[i]; return tmp; }
    bool operator==(const uint64_block_t& v2) const { for (size_t i = 0; i < size; ++i) if (v[i] != v2.v[i]) return false; return true; }
    bool operator!=(const uint64_block_t& v2) const { for (size_t i = 0; i < size; ++i) if (v[i] != v2.v[i]) return true; return false; }
    
    bool get_bit(size_t c) const   { return (v[c/64]>>(c%64))&1; }
    void set_bit(size_t c)         { v[c/64] |= uint64_t(1)<<(c%64); }
    void flip_bit(size_t c)        { v[c/64] ^= uint64_t(1)<<(c%64); }
    void clear_bit(size_t c)       { v[c/64] &= ~(uint64_t(1)<<(c%64)); }
    void set_bit(size_t c, bool b) { v[c/64] &= ~(uint64_t(1)<<(c%64)); v[c/64] |= uint64_t(b ? 1 : 0)<<(c%64); }
    size_t hammingweight() const   { size_t w = 0; for (size_t i = 0; i < size; ++i) w += mccl::hammingweight(v[i]); return w; }
    bool parity() const            { uint64_t x = 0; for (size_t i = 0; i < size; ++i) x ^= v[i]; return mccl::hammingweight(x)%2; }
};
template<size_t bits> size_t hammingweight(const uint64_block_t<bits>& x) { return x.hammingweight(); }

typedef uint64_block_t<64> block64_t; // default case
typedef uint64_block_t<128> block128_t; // for sse
typedef uint64_block_t<256> block256_t; // for avx
typedef uint64_block_t<512> block512_t; // cacheline, 2x avx2



template<size_t _bits, bool _maskedlastblock>
struct block_tag {
	typedef block_tag<_bits, _maskedlastblock> type;
	static const size_t bits = _bits;
	static const bool maskedlastblock = _maskedlastblock;
};

template<typename desttag, typename srctag1, typename srctag2 = block_tag<512, false> >
struct mixed_process_block_tag
{
	// take the minimum bits of the three tags
	static const size_t bits = srctag1::bits < srctag2::bits 
		? (desttag::bits < srctag1::bits ? desttag::bits : srctag1::bits) 
		: (desttag::bits < srctag2::bits ? desttag::bits : srctag2::bits); 
	// use maskedlastblock if at least one uses maskedlastblock
	static const bool maskedlastblock = desttag::maskedlastblock | srctag1::maskedlastblock | srctag2::maskedlastblock; 

	typedef block_tag<bits, maskedlastblock> type;
};

template<typename desttag, typename srctag1, typename srctag2 = block_tag<512, false> >
struct dest_process_block_tag
{
	// take the minimum bits of the three tags
	static const size_t bits = srctag1::bits < srctag2::bits 
		? (desttag::bits < srctag1::bits ? desttag::bits : srctag1::bits) 
		: (desttag::bits < srctag2::bits ? desttag::bits : srctag2::bits); 
	// use maskedlastblock if destination block tag requires maskedlastblock
	static const bool maskedlastblock = desttag::maskedlastblock; 

	typedef block_tag<bits, maskedlastblock> type;
};

typedef block_tag<0, true> void_block_tag; // should never be used

typedef block_tag<64 , true> default_block_tag;

typedef block_tag<64 , true> block64_lwm_tag;  // with last word mask
typedef block_tag<128, true> block128_lwm_tag; // with last word mask
typedef block_tag<256, true> block256_lwm_tag; // with last word mask
typedef block_tag<512, true> block512_lwm_tag; // with last word mask

typedef block_tag<64 , false> block64_nom_tag;  // no last word mask
typedef block_tag<128, false> block128_nom_tag; // no last word mask
typedef block_tag<256, false> block256_nom_tag; // no last word mask
typedef block_tag<512, false> block512_nom_tag; // no last word mask

// convert ptr to block ptr
template<size_t bits, bool masked>
uint64_block_t<bits>* make_block_ptr(uint64_t* ptr, block_tag<bits,masked>)
{
    static_assert(bits % 64 == 0, "make_block_ptr(): bits must be multiple of 64");
    return reinterpret_cast<uint64_block_t<bits>*>(ptr);
}
template<size_t bits, bool masked>
const uint64_block_t<bits>* make_block_ptr(const uint64_t* ptr, block_tag<bits,masked>)
{
    static_assert(bits % 64 == 0, "make_block_ptr(): bits must be multiple of 64");
    return reinterpret_cast<const uint64_block_t<bits>*>(ptr);
}
// compute stride in blocks
template<size_t bits, bool masked>
size_t row_block_stride(size_t stride, block_tag<bits,masked>)
{
    static_assert(bits % 64 == 0, "row_block_stride(): bits must be multiple of 64");
    return stride/(bits/64);
}
// compute row blocks (default rounded down)
template<size_t bits, bool masked>
size_t row_blocks(size_t columns, block_tag<bits,masked>)
{
    static_assert(bits % 64 == 0, "row_blocks(): bits must be multiple of 64");
    return columns/bits;
}
// compute row blocks (rounded up)
template<size_t bits, bool masked>
size_t row_blocks_ceil(size_t columns, block_tag<bits,masked>)
{
    static_assert(bits % 64 == 0, "row_blocks_ceil(): must be multiple of 64");
    return (columns+bits-1)/bits;
}

namespace detail
{

// matrix_ops.cpp
//template<size_t bits>
//uint64_block_t<bits> lastwordmask(size_t columns, block_tag<bits,true>);

}

MCCL_END_NAMESPACE

#endif
