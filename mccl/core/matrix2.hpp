#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/core/matrix_base.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <iostream>
#include <functional>
#include <random>

MCCL_BEGIN_NAMESPACE

class vec;
class vec_view;
class cvec_view;
class vec_view_it;
class cvec_view_it;
class mat;
class mat_view;
class cmat_view;

class cvec_view
{
public:
    cv_ptr ptr;
    
    cvec_view(): ptr() {}
    cvec_view(const cv_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    cvec_view(const cvec_view& v) = default;
    cvec_view(cvec_view&& v) = default;
    cvec_view& operator=(const cvec_view& v) = default;
    cvec_view& operator=(cvec_view&& v) = default;

    // observers    
    const uint64_t* data() const { return ptr.ptr; }
    size_t columns() const { return ptr.columns; }
    size_t rowwords() const { return (ptr.columns+63)/64; }
    size_t hw() const { return v_hw(ptr); }

    bool operator[](size_t c) const { return v_getbit(ptr,c); }
    bool operator()(size_t c) const { return v_getbit(ptr,c); }
    
    bool isequal(const cvec_view& v2) const { return v_isequal(ptr,v2.ptr); }

    // view management
    void reset(const cv_ptr& p) { ptr = p; }
    cvec_view subvector(size_t coloffset, size_t cols) const { return cvec_view(ptr.subvector(coloffset, cols)); }
};

class vec_view
{
public:
    v_ptr ptr;

    vec_view(): ptr() {}
    vec_view(const v_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    vec_view(const vec_view& v) = default;
    vec_view(vec_view&& v) = default;
    vec_view& operator=(const vec_view& v) = default;
    vec_view& operator=(vec_view&& v) = default;

    // observers    
    uint64_t* data() const { return ptr.ptr; }
    size_t columns() const { return ptr.columns; }
    size_t rowwords() const { return (ptr.columns+63)/64; }
    size_t hw() const { return v_hw(ptr); }

    bool operator[](size_t c) const { return v_getbit(ptr,c); }
    bool operator()(size_t c) const { return v_getbit(ptr,c); }

    bool isequal(const cvec_view& v2) const { return v_isequal(ptr,v2.ptr); }

    // view management
    void reset(const v_ptr& p) { ptr = p; }
    vec_view subvector(size_t coloffset, size_t cols) const { return vec_view(ptr.subvector(coloffset, cols)); }
    
    // automatic conversion
    operator       cvec_view&()       { return *reinterpret_cast<cvec_view*>(this); }
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }

    // modifiers
    vec_view& clear()                    { v_set(ptr, 0); return *this; }
    vec_view& set(bool b = true)         { v_set(ptr, b); return *this; }
    vec_view& copy(const cvec_view& src) { v_copy(ptr, src.ptr); return *this; }

    vec_view& vnot()                      { v_not(ptr); return *this; }
    vec_view& vnot(const cvec_view& src)  { v_copynot(ptr, src.ptr); return *this; }
    vec_view& vxor(const cvec_view& v2)   { v_xor(ptr, v2.ptr); return *this; }
    vec_view& vand(const cvec_view& v2)   { v_and(ptr, v2.ptr); return *this; }
    vec_view& vor (const cvec_view& v2)   { v_or (ptr, v2.ptr); return *this; }
    vec_view& vxor(const cvec_view& v1, const cvec_view& v2)   { v_xor(ptr, v1.ptr, v2.ptr); return *this; }
    vec_view& vand(const cvec_view& v1, const cvec_view& v2)   { v_and(ptr, v1.ptr, v2.ptr); return *this; }
    vec_view& vor (const cvec_view& v1, const cvec_view& v2)   { v_or (ptr, v1.ptr, v2.ptr); return *this; }
    vec_view& operator &=(const cvec_view& v2) { return vand(v2); }
    vec_view& operator |=(const cvec_view& v2) { return vor(v2); }
    vec_view& operator ^=(const cvec_view& v2) { return vxor(v2); }
    
    vec_view& clearbit(size_t c)       { v_clearbit(ptr, c); return *this; }
    vec_view& flipbit(size_t c)        { v_flipbit(ptr, c); return *this; }
    vec_view& setbit(size_t c)         { v_setbit(ptr, c); return *this; }
    vec_view& setbit(size_t c, bool b) { v_setbit(ptr, c, b); return *this; }

    vec_view& setcolumns(size_t c_off, size_t c_cnt, bool b) { v_setcolumns(ptr, c_off, c_cnt, b); return *this; }
    vec_view& flipcolumns(size_t c_off, size_t c_cnt)        { v_flipcolumns(ptr, c_off, c_cnt); return *this; }
};

// global vector operators: comparison and output stream
bool operator==(const cvec_view& v1, const cvec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view& v1, const cvec_view& v2) { return v1.ptr != v2.ptr; }
bool operator==(const cvec_view& v1, const vec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view& v1, const vec_view& v2) { return v1.ptr != v2.ptr; }
bool operator==(const vec_view& v1, const cvec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const vec_view& v1, const cvec_view& v2) { return v1.ptr != v2.ptr; }
bool operator==(const vec_view& v1, const vec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const vec_view& v1, const vec_view& v2) { return v1.ptr != v2.ptr; }
std::ostream& operator<<(std::ostream& o, const cvec_view& v) { v_print(o, v.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const vec_view& v) { v_print(o, v.ptr); return o; }

class cvec_view_it
{
public:
    cvi_ptr ptr;
    
    cvec_view_it(): ptr() {}
    cvec_view_it(const cvi_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    cvec_view_it(const cvec_view_it& v) = default;
    cvec_view_it(cvec_view_it&& v) = default;
    cvec_view_it& operator=(const cvec_view_it& v) = default;
    cvec_view_it& operator=(cvec_view_it&& v) = default;

    // observers    
    const uint64_t* data() const { return ptr.ptr; }
    size_t columns() const { return ptr.columns; }
    size_t rowwords() const { return (ptr.columns+63)/64; }
    size_t stride() const { return ptr.stride; }
    size_t hw() const { return v_hw(ptr); }

    bool operator[](size_t c) const { return v_getbit(ptr,c); }
    bool operator()(size_t c) const { return v_getbit(ptr,c); }

    bool isequal(const cvec_view& v2) const { return v_isequal(ptr,v2.ptr); }

    // view management
    void reset(const cvi_ptr& p) { ptr = p; }
    cvec_view_it subvector(size_t coloffset, size_t cols) const { return cvec_view_it(ptr.subvectorit(coloffset, cols)); }
    cvec_view_it& operator++() { ++ptr; return *this; }
    cvec_view_it& operator--() { --ptr; return *this; }
    cvec_view_it& operator+=(size_t n) { ptr+=n; return *this; }
    cvec_view_it& operator-=(size_t n) { ptr-=n; return *this; }

    cvec_view_it operator++(int) { return cvec_view_it(ptr++); }
    cvec_view_it operator--(int) { return cvec_view_it(ptr--); }
    cvec_view_it operator+(size_t n) { return cvec_view_it(ptr+n); }
    cvec_view_it operator-(size_t n) { return cvec_view_it(ptr-n); }
    ptrdiff_t operator-(cvec_view_it& v2) { return ptr - v2.ptr; }

    // automatic conversion
    operator       cvec_view&()       { return *reinterpret_cast<cvec_view*>(this); }
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }
};

class vec_view_it
{
public:
    vi_ptr ptr;

    vec_view_it(): ptr() {}
    vec_view_it(const vi_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    vec_view_it(const vec_view_it& v) = default;
    vec_view_it(vec_view_it&& v) = default;
    vec_view_it& operator=(const vec_view_it& v) = default;
    vec_view_it& operator=(vec_view_it&& v) = default;

    // observers    
    uint64_t* data() const { return ptr.ptr; }
    size_t columns() const { return ptr.columns; }
    size_t rowwords() const { return (ptr.columns+63)/64; }
    size_t stride() const { return ptr.stride; }
    size_t hw() const { return v_hw(ptr); }

    bool operator[](size_t c) const { return v_getbit(ptr,c); }
    bool operator()(size_t c) const { return v_getbit(ptr,c); }

    bool isequal(const cvec_view& v2) const { return v_isequal(ptr,v2.ptr); }

    // view management
    void reset(const vi_ptr& p) { ptr = p; }
    vec_view_it subvector(size_t coloffset, size_t cols) const { return vec_view_it(ptr.subvectorit(coloffset, cols)); }
    vec_view_it& operator++() { ++ptr; return *this; }
    vec_view_it& operator--() { --ptr; return *this; }
    vec_view_it& operator+=(size_t n) { ptr+=n; return *this; }
    vec_view_it& operator-=(size_t n) { ptr-=n; return *this; }

    vec_view_it operator++(int) { return vec_view_it(ptr++); }
    vec_view_it operator--(int) { return vec_view_it(ptr--); }
    vec_view_it operator+(size_t n) { return vec_view_it(ptr+n); }
    vec_view_it operator-(size_t n) { return vec_view_it(ptr-n); }
    ptrdiff_t operator-(cvec_view_it& v2) { return ptr - v2.ptr; }
    
    // automatic conversion
    operator       cvec_view&()       { return *reinterpret_cast<cvec_view*>(this); }
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }
    operator       vec_view&()       { return *reinterpret_cast<vec_view*>(this); }
    operator const vec_view&() const { return *reinterpret_cast<const vec_view*>(this); }
    operator       cvec_view_it&()       { return *reinterpret_cast<cvec_view_it*>(this); }
    operator const cvec_view_it&() const { return *reinterpret_cast<const cvec_view_it*>(this); }

    // content modifiers
    vec_view_it& clear()                    { v_set(ptr, 0); return *this; }
    vec_view_it& set(bool b = true)         { v_set(ptr, b); return *this; }
    vec_view_it& copy(const cvec_view& src) { v_copy(ptr, src.ptr); return *this; }

    vec_view_it& vnot()                      { v_not(ptr); return *this; }
    vec_view_it& vnot(const cvec_view& src)  { v_copynot(ptr, src.ptr); return *this; }
    vec_view_it& vxor(const cvec_view& v2)   { v_xor(ptr, v2.ptr); return *this; }
    vec_view_it& vand(const cvec_view& v2)   { v_and(ptr, v2.ptr); return *this; }
    vec_view_it& vor (const cvec_view& v2)   { v_or (ptr, v2.ptr); return *this; }
    vec_view_it& vxor(const cvec_view& v1, const cvec_view& v2)   { v_xor(ptr, v1.ptr, v2.ptr); return *this; }
    vec_view_it& vand(const cvec_view& v1, const cvec_view& v2)   { v_and(ptr, v1.ptr, v2.ptr); return *this; }
    vec_view_it& vor (const cvec_view& v1, const cvec_view& v2)   { v_or (ptr, v1.ptr, v2.ptr); return *this; }
    vec_view_it& operator &=(const cvec_view& v2) { return vand(v2); }
    vec_view_it& operator |=(const cvec_view& v2) { return vor(v2); }
    vec_view_it& operator ^=(const cvec_view& v2) { return vxor(v2); }
    
    vec_view_it& clearbit(size_t c)       { v_clearbit(ptr, c); return *this; }
    vec_view_it& flipbit(size_t c)        { v_flipbit(ptr, c); return *this; }
    vec_view_it& setbit(size_t c)         { v_setbit(ptr, c); return *this; }
    vec_view_it& setbit(size_t c, bool b) { v_setbit(ptr, c, b); return *this; }

    vec_view_it& setcolumns(size_t c_off, size_t c_cnt, bool b) { v_setcolumns(ptr, c_off, c_cnt, b); return *this; }
    vec_view_it& flipcolumns(size_t c_off, size_t c_cnt)        { v_flipcolumns(ptr, c_off, c_cnt); return *this; }
};

// global vector operators: comparison and output stream
bool operator==(const cvec_view_it& v1, const cvec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view_it& v1, const cvec_view_it& v2) { return v1.ptr != v2.ptr; }
bool operator==(const cvec_view_it& v1, const vec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view_it& v1, const vec_view_it& v2) { return v1.ptr != v2.ptr; }
bool operator==(const vec_view_it& v1, const cvec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const vec_view_it& v1, const cvec_view_it& v2) { return v1.ptr != v2.ptr; }
bool operator==(const vec_view_it& v1, const vec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const vec_view_it& v1, const vec_view_it& v2) { return v1.ptr != v2.ptr; }
std::ostream& operator<<(std::ostream& o, const cvec_view_it& v) { v_print(o, v.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const vec_view_it& v) { v_print(o, v.ptr); return o; }


class cmat_view
{
public:
    cm_ptr ptr;
    
    cmat_view(): ptr() {}
    cmat_view(const cm_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    cmat_view(const cmat_view& m) = default;
    cmat_view(cmat_view&& m) = default;
    cmat_view& operator=(const cmat_view& m) = default;
    cmat_view& operator=(cmat_view&& m) = default;

    // observers    
    const uint64_t* data() const { return ptr.ptr; }
    size_t columns() const { return ptr.columns; }
    size_t rowwords() const { return (ptr.columns+63)/64; }
    size_t rows() const { return ptr.rows; }
    size_t stride() const { return ptr.stride; }
    size_t hw() const { return m_hw(ptr); }

    cvec_view_it operator[](size_t r) const { return cvec_view_it(ptr.subvectorit(r)); }
    cvec_view_it operator()(size_t r) const { return cvec_view_it(ptr.subvectorit(r)); }
    bool operator()(size_t r, size_t c) const { return m_getbit(ptr,r,c); }

    // view management
    void reset(const cm_ptr& p) { ptr = p; }
    cvec_view_it subvector(size_t row, size_t coloffset, size_t cols) const { return cvec_view_it(ptr.subvectorit(row, coloffset, cols)); }
    cmat_view submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const { return cmat_view(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
};

class mat_view
{
public:
    m_ptr ptr;
    
    mat_view(): ptr() {}
    mat_view(const m_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    mat_view(const mat_view& m) = default;
    mat_view(mat_view&& m) = default;
    mat_view& operator=(const mat_view& m) = default;
    mat_view& operator=(mat_view&& m) = default;

    // observers    
    uint64_t* data() const { return ptr.ptr; }
    size_t columns() const { return ptr.columns; }
    size_t rowwords() const { return (ptr.columns+63)/64; }
    size_t rows() const { return ptr.rows; }
    size_t stride() const { return ptr.stride; }
    size_t hw() const { return m_hw(ptr); }

    vec_view_it operator[](size_t r) const { return vec_view_it(ptr.subvectorit(r)); }
    vec_view_it operator()(size_t r) const { return vec_view_it(ptr.subvectorit(r)); }
    bool operator()(size_t r, size_t c) const { return m_getbit(ptr,r,c); }

    bool isequal(const cmat_view& m2) const { return m_isequal(ptr,m2.ptr); }

    // view management
    void reset(const m_ptr& p) { ptr = p; }
    vec_view_it subvector(size_t row, size_t coloffset, size_t cols) const { return vec_view_it(ptr.subvectorit(row, coloffset, cols)); }
    mat_view submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const { return mat_view(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    // automatic conversion
    operator       cmat_view&()       { return *reinterpret_cast<cmat_view*>(this); }
    operator const cmat_view&() const { return *reinterpret_cast<const cmat_view*>(this); }

    // content modifiers
    mat_view& clear()                    { m_set(ptr, 0); return *this; }
    mat_view& set(bool b = true)         { m_set(ptr, b); return *this; }
    mat_view& copy(const cmat_view& src) { m_copy(ptr, src.ptr); return *this; }
    mat_view& transpose(const cmat_view& src) { m_transpose(ptr, src.ptr); return *this; }

    mat_view& mnot()                      { m_not(ptr); return *this; }
    mat_view& mnot(const cmat_view& src)  { m_copynot(ptr, src.ptr); return *this; }
    mat_view& mxor(const cmat_view& m2)   { m_xor(ptr, m2.ptr); return *this; }
    mat_view& mand(const cmat_view& m2)   { m_and(ptr, m2.ptr); return *this; }
    mat_view& mor (const cmat_view& m2)   { m_or (ptr, m2.ptr); return *this; }
    mat_view& mxor(const cmat_view& m1, const cmat_view& m2)   { m_xor(ptr, m1.ptr, m2.ptr); return *this; }
    mat_view& mand(const cmat_view& m1, const cmat_view& m2)   { m_and(ptr, m1.ptr, m2.ptr); return *this; }
    mat_view& mor (const cmat_view& m1, const cmat_view& m2)   { m_or (ptr, m1.ptr, m2.ptr); return *this; }
    mat_view& operator &=(const cmat_view& m2) { return mand(m2); }
    mat_view& operator |=(const cmat_view& m2) { return mor(m2); }
    mat_view& operator ^=(const cmat_view& m2) { return mxor(m2); }
    
    mat_view& clearbit(size_t r, size_t c)       { m_clearbit(ptr, r, c); return *this; }
    mat_view& flipbit(size_t r, size_t c)        { m_flipbit(ptr, r, c); return *this; }
    mat_view& setbit(size_t r, size_t c)         { m_setbit(ptr, r, c); return *this; }
    mat_view& setbit(size_t r, size_t c, bool b) { m_setbit(ptr, r, c, b); return *this; }

    mat_view& setcolumns(size_t c_off, size_t c_cnt, bool b) { m_setcolumns(ptr, c_off, c_cnt, b); return *this; }
    mat_view& flipcolumns(size_t c_off, size_t c_cnt)        { m_flipcolumns(ptr, c_off, c_cnt); return *this; }

};

// global matrix operators: comparison and output stream
bool operator==(const cmat_view& m1, const cmat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const cmat_view& m1, const cmat_view& m2) { return m1.ptr != m2.ptr; }
bool operator==(const cmat_view& m1, const mat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const cmat_view& m1, const mat_view& m2) { return m1.ptr != m2.ptr; }
bool operator==(const mat_view& m1, const cmat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const mat_view& m1, const cmat_view& m2) { return m1.ptr != m2.ptr; }
bool operator==(const mat_view& m1, const mat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const mat_view& m1, const mat_view& m2) { return m1.ptr != m2.ptr; }
std::ostream& operator<<(std::ostream& o, const cmat_view& m) { v_print(o, m.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const mat_view& m) { v_print(o, m.ptr); return o; }

MCCL_END_NAMESPACE

#endif
