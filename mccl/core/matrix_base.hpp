#ifndef MCCL_CORE_MATRIX_BASE_HPP
#define MCCL_CORE_MATRIX_BASE_HPP

#include <mccl/config/config.hpp>
#include <stdexcept>
#include <cstdint>

MCCL_BEGIN_NAMESPACE

using std::uint64_t;
using std::size_t;
using std::ptrdiff_t;

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
	
struct cv_ptr {
	const uint64_t* ptr;
	size_t columns;

	cv_ptr(): ptr(nullptr), columns(0) {}
	cv_ptr(const uint64_t* _ptr, size_t _columns)
		: ptr(_ptr), columns(_columns)
	{}
	cv_ptr(const cv_ptr& v) = default;
	cv_ptr(cv_ptr&& v) = default;
	cv_ptr& operator=(const cv_ptr& v) = default;
	cv_ptr& operator=(cv_ptr&&) = default;

	const uint64_t* data() const { return ptr; }
	const uint64_t* data(size_t c) const { return ptr+(c/64); }
	
	void reset(const uint64_t* _ptr, size_t _columns)
	{
		ptr = _ptr;
		columns = _columns;
	}
	cv_ptr subvector(size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("subvector: columns out of range");
		return cv_ptr(ptr + (coloffset/64), cols);
	}
};
struct v_ptr {
	uint64_t* ptr;
	size_t columns;
	
	v_ptr(): ptr(nullptr), columns(0) {}
	v_ptr(uint64_t* _ptr, size_t _columns)
		: ptr(_ptr), columns(_columns)
	{}
	v_ptr(const v_ptr& v) = default;
	v_ptr(v_ptr&& v) = default;
	v_ptr& operator=(const v_ptr& v) = default;
	v_ptr& operator=(v_ptr&&) = default;
	
	operator cv_ptr&() { return *reinterpret_cast<cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }

	uint64_t* data() const { return ptr; }
	uint64_t* data(size_t c) const { return ptr+(c/64); }

	void reset(uint64_t* _ptr, size_t _columns)
	{
		ptr = _ptr;
		columns = _columns;
	}
	v_ptr subvector(size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("subvector: columns out of range");
		return v_ptr(ptr + (coloffset/64), cols);
	}
};


struct cvi_ptr {
	const uint64_t* ptr;
	size_t columns, stride;

	cvi_ptr(): ptr(nullptr), columns(0), stride(0) {}
	cvi_ptr(const uint64_t* _ptr, size_t _columns, size_t _stride)
		: ptr(_ptr), columns(_columns), stride(_stride)
	{}
	cvi_ptr(const cvi_ptr& v) = default;
	cvi_ptr(cvi_ptr&& v) = default;
	cvi_ptr& operator=(const cvi_ptr& v) = default;
	cvi_ptr& operator=(cvi_ptr&&) = default;

	operator cv_ptr&() { return *reinterpret_cast<cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }

	const uint64_t* data() const { return ptr; }
	const uint64_t* data(size_t c) const { return ptr+(c/64); }

	void reset(const uint64_t* _ptr, size_t _columns, size_t _stride)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
	}
	cv_ptr subvector(size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("subvector: columns out of range");
		return cv_ptr(ptr + (coloffset/64), cols);
	}
	cvi_ptr subvectorit(size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("subvectorit: columns out of range");
		return cvi_ptr(ptr + (coloffset/64), cols, stride);
	}

	cvi_ptr& operator++() { ptr += stride; return *this; }
	cvi_ptr& operator--() { ptr -= stride; return *this; }
	cvi_ptr& operator+=(size_t n) { ptr += n*stride; return *this; }
	cvi_ptr& operator-=(size_t n) { ptr -= n*stride; return *this; }
	cvi_ptr operator++(int) { ptr += stride; return cvi_ptr(ptr-stride,columns,stride); }
	cvi_ptr operator--(int) { ptr -= stride; return cvi_ptr(ptr+stride,columns,stride); }

	cvi_ptr operator+(size_t n) const { return cvi_ptr(ptr + n*stride,columns,stride); }
	cvi_ptr operator-(size_t n) const { return cvi_ptr(ptr - n*stride,columns,stride); }
	ptrdiff_t operator-(const cvi_ptr& r) const { return (ptr-r.ptr)/ptrdiff_t(stride); }
};

struct vi_ptr {
	uint64_t* ptr;
	size_t columns, stride;
	
	vi_ptr(): ptr(nullptr), columns(0) {}
	vi_ptr(uint64_t* _ptr, size_t _columns, size_t _stride)
		: ptr(_ptr), columns(_columns), stride(_stride)
	{}
	vi_ptr(const vi_ptr& v) = default;
	vi_ptr(vi_ptr&& v) = default;
	vi_ptr& operator=(const vi_ptr& v) = default;
	vi_ptr& operator=(vi_ptr&&) = default;
	
	operator cvi_ptr&() { return *reinterpret_cast<cvi_ptr*>(this); }
	operator const cvi_ptr&() const { return *reinterpret_cast<const cvi_ptr*>(this); }
	operator cv_ptr&() { return *reinterpret_cast<cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }
	operator v_ptr&() { return *reinterpret_cast<v_ptr*>(this); }
	operator const v_ptr&() const { return *reinterpret_cast<const v_ptr*>(this); }

	uint64_t* data() const { return ptr; }
	uint64_t* data(size_t c) const { return ptr+(c/64); }

	void reset(uint64_t* _ptr, size_t _columns, size_t _stride)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
	}
	v_ptr subvector(size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("subvector: columns out of range");
		return v_ptr(ptr + (coloffset/64), cols);
	}
	vi_ptr subvectorit(size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns)) throw std::out_of_range("subvectorit: columns out of range");
		return vi_ptr(ptr + (coloffset/64), cols, stride);
	}

	vi_ptr& operator++() { ptr += stride; return *this; }
	vi_ptr& operator--() { ptr -= stride; return *this; }
	vi_ptr& operator+=(size_t n) { ptr += n*stride; return *this; }
	vi_ptr& operator-=(size_t n) { ptr -= n*stride; return *this; }
	vi_ptr operator++(int) { ptr += stride; return vi_ptr(ptr-stride,columns,stride); }
	vi_ptr operator--(int) { ptr -= stride; return vi_ptr(ptr+stride,columns,stride); }
	
	vi_ptr operator+(size_t n) const { return vi_ptr(ptr + n*stride,columns,stride); }
	vi_ptr operator-(size_t n) const { return vi_ptr(ptr - n*stride,columns,stride); }
	ptrdiff_t operator-(const cvi_ptr& r) const { return (ptr-r.ptr)/ptrdiff_t(stride); }
};


struct cm_ptr {
	const uint64_t* ptr;
	size_t columns, stride, rows;
	
	cm_ptr(): ptr(nullptr), columns(0), stride(0), rows(0) {}
	cm_ptr(const uint64_t* _ptr, size_t _rows, size_t _columns, size_t _stride)
		: ptr(_ptr), columns(_columns), stride(_stride), rows(_rows)
	{}
	cm_ptr(const cm_ptr& m) = default;
	cm_ptr(cm_ptr&& m) = default;
	cm_ptr& operator=(const cm_ptr& m) = default;
	cm_ptr& operator=(cm_ptr&& m) = default;
	
	operator cv_ptr&() { return *reinterpret_cast<cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }
	operator cvi_ptr&() { return *reinterpret_cast<cvi_ptr*>(this); }
	operator const cvi_ptr&() const { return *reinterpret_cast<const cvi_ptr*>(this); }

	const uint64_t* data() const { return ptr; }
	const uint64_t* data(size_t r) const { return ptr+r*stride; }
	const uint64_t* data(size_t r, size_t c) const { return ptr + r*stride + (c/64); }

	void reset(const uint64_t* _ptr, size_t _columns, size_t _stride, size_t _rows)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
		rows = _rows;
	}
	cv_ptr subvector(size_t rowoffset = 0) const
	{
		if (rowoffset >= rows) throw std::out_of_range("subvector: columns out of range");
		return cv_ptr(ptr + (rowoffset*stride), columns);
	}
	cv_ptr subvector(size_t rowoffset, size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("subvector: columns out of range");
		return cv_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols);
	}
	cvi_ptr subvectorit(size_t rowoffset = 0) const
	{
		if (rowoffset >= rows) throw std::out_of_range("subvectorit: columns out of range");
		return cvi_ptr(ptr + (rowoffset*stride), columns, stride);
	}
	cvi_ptr subvectorit(size_t rowoffset, size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("subvectorit: columns out of range");
		return cvi_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride);
	}
	cm_ptr submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset+_rows>rows)) throw std::out_of_range("submatrix: columns out of range");
		return cm_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride, _rows);
	}
};
struct m_ptr {
	uint64_t* ptr;
	size_t columns, stride, rows;
	
	m_ptr(): ptr(nullptr), columns(0), stride(0), rows(0) {}
	m_ptr(uint64_t* _ptr, size_t _rows, size_t _columns, size_t _stride)
		: ptr(_ptr), columns(_columns), stride(_stride), rows(_rows)
	{}
	m_ptr(const m_ptr& m) = default;
	m_ptr(m_ptr&& m) = default;
	m_ptr& operator=(const m_ptr& m) = default;
	m_ptr& operator=(m_ptr&& m) = default;
	
	operator       v_ptr&()        { return *reinterpret_cast<      v_ptr*>(this); }
	operator const v_ptr&() const  { return *reinterpret_cast<const v_ptr*>(this); }
	operator       cv_ptr&()       { return *reinterpret_cast<      cv_ptr*>(this); }
	operator const cv_ptr&() const { return *reinterpret_cast<const cv_ptr*>(this); }
	operator       vi_ptr&()        { return *reinterpret_cast<      vi_ptr*>(this); }
	operator const vi_ptr&() const  { return *reinterpret_cast<const vi_ptr*>(this); }
	operator       cvi_ptr&()       { return *reinterpret_cast<      cvi_ptr*>(this); }
	operator const cvi_ptr&() const { return *reinterpret_cast<const cvi_ptr*>(this); }
	operator       cm_ptr&()       { return *reinterpret_cast<      cm_ptr*>(this); }
	operator const cm_ptr&() const { return *reinterpret_cast<const cm_ptr*>(this); }

	uint64_t* data() const { return ptr; }
	uint64_t* data(size_t r) const { return ptr+r*stride; }
	uint64_t* data(size_t r, size_t c) const { return ptr + r*stride + (c/64); }

	void reset(uint64_t* _ptr, size_t _columns, size_t _stride, size_t _rows)
	{
		ptr = _ptr;
		columns = _columns;
		stride = _stride;
		rows = _rows;
	}
	v_ptr subvector(size_t rowoffset = 0) const
	{
		if (rowoffset >= rows) throw std::out_of_range("subvector: columns out of range");
		return v_ptr(ptr + (rowoffset*stride), columns);
	}
	v_ptr subvector(size_t rowoffset, size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("subvector: columns out of range");
		return v_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols);
	}
	vi_ptr subvectorit(size_t rowoffset = 0) const
	{
		if (rowoffset >= rows) throw std::out_of_range("subvectorit: columns out of range");
		return vi_ptr(ptr + (rowoffset*stride), columns, stride);
	}
	vi_ptr subvectorit(size_t rowoffset, size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset >= rows)) throw std::out_of_range("subvectorit: columns out of range");
		return vi_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride);
	}
	m_ptr submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const
	{
		if ((coloffset % 64 != 0) || (coloffset+cols>columns) || (rowoffset+_rows>rows)) throw std::out_of_range("submatrix: columns out of range");
		return m_ptr(ptr + (coloffset/64) + (rowoffset*stride), cols, stride, _rows);
	}
};

bool operator==(const cv_ptr& v1, const cv_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
bool operator!=(const cv_ptr& v1, const cv_ptr& v2) { return !(v1 == v2); }
bool operator==(const cv_ptr& v1, const v_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
bool operator!=(const cv_ptr& v1, const v_ptr& v2) { return !(v1 == v2); }
bool operator==(const v_ptr& v1, const cv_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
bool operator!=(const v_ptr& v1, const cv_ptr& v2) { return !(v1 == v2); }
bool operator==(const v_ptr& v1, const v_ptr& v2) { return (v1.ptr == v2.ptr) && (v1.columns == v2.columns); }
bool operator!=(const v_ptr& v1, const v_ptr& v2) { return !(v1 == v2); }

bool operator==(const cvi_ptr& v1, const cvi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
bool operator!=(const cvi_ptr& v1, const cvi_ptr& v2) { return !(v1 == v2); }
bool operator==(const cvi_ptr& v1, const vi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
bool operator!=(const cvi_ptr& v1, const vi_ptr& v2) { return !(v1 == v2); }
bool operator==(const vi_ptr& v1, const cvi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
bool operator!=(const vi_ptr& v1, const cvi_ptr& v2) { return !(v1 == v2); }
bool operator==(const vi_ptr& v1, const vi_ptr& v2) { return (v1.ptr == v2.ptr) && ((v1.columns == v2.columns) & (v1.stride == v2.stride)); }
bool operator!=(const vi_ptr& v1, const vi_ptr& v2) { return !(v1 == v2); }

bool operator==(const cm_ptr& m1, const cm_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
bool operator!=(const cm_ptr& m1, const cm_ptr& m2) { return !(m1 == m2); }
bool operator==(const cm_ptr& m1, const m_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
bool operator!=(const cm_ptr& m1, const m_ptr& m2) { return !(m1 == m2); }
bool operator==(const m_ptr& m1, const cm_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
bool operator!=(const m_ptr& m1, const cm_ptr& m2) { return !(m1 == m2); }
bool operator==(const m_ptr& m1, const m_ptr& m2) { return (m1.ptr == m2.ptr) && ((m1.columns == m2.columns) & (m1.stride == m2.stride) & (m1.rows == m2.rows)); }
bool operator!=(const m_ptr& m1, const m_ptr& m2) { return !(m1 == m2); }


MCCL_END_NAMESPACE

#endif
