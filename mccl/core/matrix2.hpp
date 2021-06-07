#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/core/matrix_base.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <iostream>
#include <functional>
#include <random>

MCCL_BEGIN_NAMESPACE

// = vector view classes =
// These only point to a preallocated vector in memory
// Copy/move constructor/assignment only alter the view, not the vector contents
//   use vec_view.copy(src) etc. to modify view's contents
// cvec_view points to const vector contents, vec_view points to non-const vector contents
// Note that 'const vec_view' like a 'const pointer_t' still allows to modify the vector contents
// Allows assigning a vector_result:
//   however it must match the result vector dimension
class vec_view;
class cvec_view;

// Extension of vec_view / cvec_view with rowstride
// and pointer operators ++,--,+=,-=,+,- that change the view by # rows forward/backward
class vec_view_it;
class cvec_view_it;

// = matrix view classes =
// These only point to a preallocated matrix in memory
// Copy/move constructor/assignment only alter the view, not the matrix contents
//   use mat_view.copy(src) etc. to modify view's contents
// cmat_view points to const matrix contents, mat_view points to non-const matrix contents
// Note that 'const mat_view' like a 'const pointer_t' still allows to modify the matrix contents
// Allows assigning a matrix_result:
//   however it must match the result matrix dimensions
class mat_view;
class cmat_view;

// = matrix & vector classes =
// maintains memory for a matrix/vector of specified dimensions
// behaves as vec_view/mat_view when non-const
// behaves as cvec_view/cmat_view when const
// except:
// 1) copy/move constructor/assignment from cvec_view/cmat_view
//    automatically resizes and copies content
// 2) when assigning a vector_result/matrix_result:
//    then it will automatically resize to the result vector/matrix
class vec;
class mat;



// common class members for: cvec_view, vec_view, vec_view_it, cvec_view_it, vec
#define CONST_VECTOR_CLASS_MEMBERS \
    const uint64_t* data() const { return ptr.ptr; } \
    size_t columns() const { return ptr.columns; } \
    size_t rowwords() const { return (ptr.columns+63)/64; } \
    size_t hw() const { return v_hw(ptr); } \
    bool operator[](size_t c) const { return v_getbit(ptr,c); } \
    bool operator()(size_t c) const { return v_getbit(ptr,c); } \
    bool isequal(const cvec_view& v2) const { return v_isequal(ptr,v2.ptr); }

// common class members for: vec_view, vec_view_it, vec
// cnst = '' / 'const' to allow for const and non-const versions
// vec_view & vec_view have both const and non-const versions
// vec only has non-const versions
#define VECTOR_CLASS_MEMBERS(vectype,cnst) \
    cnst vectype& clearbit(size_t c)         cnst { v_clearbit(ptr, c); return *this; } \
    cnst vectype& flipbit(size_t c)          cnst { v_flipbit(ptr, c); return *this; } \
    cnst vectype& setbit(size_t c)           cnst { v_setbit(ptr, c); return *this; } \
    cnst vectype& setbit(size_t c, bool b)   cnst { v_setbit(ptr, c, b); return *this; } \
    cnst vectype& clear()                    cnst { v_set(ptr, 0); return *this; } \
    cnst vectype& set(bool b = true)         cnst { v_set(ptr, b); return *this; } \
    cnst vectype& copy(const cvec_view& src) cnst { v_copy(ptr, src.ptr); return *this; } \
    cnst vectype& vnot()                     cnst { v_not(ptr); return *this; } \
    cnst vectype& vnot(const cvec_view& src) cnst { v_copynot(ptr, src.ptr); return *this; } \
    cnst vectype& vxor(const cvec_view& v2)  cnst { v_xor(ptr, v2.ptr); return *this; } \
    cnst vectype& vand(const cvec_view& v2)  cnst { v_and(ptr, v2.ptr); return *this; } \
    cnst vectype& vor (const cvec_view& v2)  cnst { v_or (ptr, v2.ptr); return *this; } \
    cnst vectype& operator&=(const cvec_view& v2) cnst { v_and(ptr, v2.ptr); return *this; } \
    cnst vectype& operator^=(const cvec_view& v2) cnst { v_xor(ptr, v2.ptr); return *this; } \
    cnst vectype& operator|=(const cvec_view& v2) cnst { v_or (ptr, v2.ptr); return *this; } \
    cnst vectype& vxor(const cvec_view& v1, const cvec_view& v2)  cnst { v_xor(ptr, v1.ptr, v2.ptr); return *this; } \
    cnst vectype& vand(const cvec_view& v1, const cvec_view& v2)  cnst { v_and(ptr, v1.ptr, v2.ptr); return *this; } \
    cnst vectype& vor (const cvec_view& v1, const cvec_view& v2)  cnst { v_or (ptr, v1.ptr, v2.ptr); return *this; } \
    cnst vectype& setcolumns(size_t c_off, size_t c_cnt, bool b)  cnst { v_setcolumns(ptr, c_off, c_cnt, b); return *this; } \
    cnst vectype& setcolumns(size_t c_off, size_t c_cnt)          cnst { v_setcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst vectype& clearcolumns(size_t c_off, size_t c_cnt)        cnst { v_clearcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst vectype& flipcolumns(size_t c_off, size_t c_cnt)         cnst { v_flipcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst vectype& swap(const  vec_view& v2)  cnst { v_swap(ptr, v2.ptr); return *this; }

// common class members for: cvec_view_it, vec_view_it
#define CONST_VECTOR_ITERATOR_CLASS_MEMBERS(vectype) \
    vectype& operator++() { ++ptr; return *this; } \
    vectype& operator--() { --ptr; return *this; } \
    vectype& operator+=(size_t n) { ptr+=n; return *this; } \
    vectype& operator-=(size_t n) { ptr-=n; return *this; } \
    vectype operator++(int) { return vectype(ptr++); } \
    vectype operator--(int) { return vectype(ptr--); } \
    vectype operator+(size_t n) const { return vectype(ptr+n); } \
    vectype operator-(size_t n) const { return vectype(ptr-n); } \
    ptrdiff_t operator-(cvec_view_it& v2) const { return ptr - v2.ptr; }

// common class members for: cmat_view, mat_view, mat
#define CONST_MATRIX_CLASS_MEMBERS \
    const uint64_t* data() const { return ptr.ptr; } \
    size_t columns() const { return ptr.columns; } \
    size_t rowwords() const { return (ptr.columns+63)/64; } \
    size_t rows() const { return ptr.rows; } \
    size_t stride() const { return ptr.stride; } \
    size_t hw() const { return m_hw(ptr); } \
    bool operator()(size_t r, size_t c) const { return m_getbit(ptr,r,c); } \
    bool isequal(const cmat_view& m2) const { return m_isequal(ptr,m2.ptr); }

// common class members for: mat_view, mat
// cnst = '' / 'const' to allow for const and non-const versions
// mat_view has both const and non-const versions
// mat only has non-const versions
#define MATRIX_CLASS_MEMBERS(mattype,cnst) \
    cnst mattype& clearbit(size_t r, size_t c)       cnst { m_clearbit(ptr, r, c); return *this; } \
    cnst mattype& flipbit(size_t r, size_t c)        cnst { m_flipbit(ptr, r, c); return *this; } \
    cnst mattype& setbit(size_t r, size_t c)         cnst { m_setbit(ptr, r, c); return *this; } \
    cnst mattype& setbit(size_t r, size_t c, bool b) cnst { m_setbit(ptr, r, c, b); return *this; } \
    cnst mattype& clear()                            cnst { m_set(ptr, 0); return *this; } \
    cnst mattype& set(bool b = true)                 cnst { m_set(ptr, b); return *this; } \
    cnst mattype& copy(const cmat_view& src)         cnst { m_copy(ptr, src.ptr); return *this; } \
    cnst mattype& transpose(const cmat_view& src)    cnst { m_transpose(ptr, src.ptr); return *this; } \
    cnst mattype& mnot()                             cnst { m_not(ptr); return *this; } \
    cnst mattype& mnot(const cmat_view& src)         cnst { m_copynot(ptr, src.ptr); return *this; } \
    cnst mattype& mxor(const cmat_view& m2)          cnst { m_xor(ptr, m2.ptr); return *this; } \
    cnst mattype& mand(const cmat_view& m2)          cnst { m_and(ptr, m2.ptr); return *this; } \
    cnst mattype& mor (const cmat_view& m2)          cnst { m_or (ptr, m2.ptr); return *this; } \
    cnst mattype& operator&=(const cmat_view& m2)    cnst { m_and(ptr, m2.ptr); return *this; } \
    cnst mattype& operator^=(const cmat_view& m2)    cnst { m_xor(ptr, m2.ptr); return *this; } \
    cnst mattype& operator|=(const cmat_view& m2)    cnst { m_or (ptr, m2.ptr); return *this; } \
    cnst mattype& mxor(const cmat_view& m1, const cmat_view& m2) cnst { m_xor(ptr, m1.ptr, m2.ptr); return *this; } \
    cnst mattype& mand(const cmat_view& m1, const cmat_view& m2) cnst { m_and(ptr, m1.ptr, m2.ptr); return *this; } \
    cnst mattype& mor (const cmat_view& m1, const cmat_view& m2) cnst { m_or (ptr, m1.ptr, m2.ptr); return *this; } \
    cnst mattype& setcolumns(size_t c_off, size_t c_cnt, bool b) cnst { m_setcolumns(ptr, c_off, c_cnt, b); return *this; } \
    cnst mattype& setcolumns(size_t c_off, size_t c_cnt)         cnst { m_setcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst mattype& clearcolumns(size_t c_off, size_t c_cnt)       cnst { m_clearcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst mattype& flipcolumns(size_t c_off, size_t c_cnt)        cnst { m_flipcolumns(ptr, c_off, c_cnt); return *this; }

// meta-programming construct to convert 'v.vand(v1,v2)' to 'v = v_and(v1,v2)';
// v_and(v1,v2) returns a vector_result<R> such that 'r' (of type R) contains the pointers to v1 & v2 
// and the expression 'r(v)' calls the respective function 'v.vand(v1,v2)'
// note: to allow vec & mat to automatically resize to the correct result dimensions
// r should have a member 'resize_me(cols)' / 'resize_me(rows,cols)'
template<typename v_ptr_op_result>
struct vector_result
{
    v_ptr_op_result r;
    vector_result(): r() {}
    vector_result(const vector_result&) = default;
    vector_result(vector_result&&) = default;
    vector_result& operator=(const vector_result&) = default;
    vector_result& operator=(vector_result&&) = default;
    template<typename... Args>
    vector_result(Args... args...): r(std::forward<Args>(args)...) {}
};
template<typename m_ptr_op_result>
struct matrix_result
{
    m_ptr_op_result r;
    matrix_result(): r() {}
    matrix_result(const matrix_result&) = default;
    matrix_result(matrix_result&&) = default;
    matrix_result& operator=(const matrix_result&) = default;
    matrix_result& operator=(matrix_result&&) = default;
    template<typename... Args>
    matrix_result(Args... args...): r(std::forward<Args>(args)...) {}
};


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
    
    // view management
    void reset(const cv_ptr& p) { ptr = p; }
    cvec_view subvector(size_t coloffset, size_t cols) const { return cvec_view(ptr.subvector(coloffset, cols)); }

    // automatic conversion

    // common vector API class members
    CONST_VECTOR_CLASS_MEMBERS
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

    // view management
    void reset(const v_ptr& p) { ptr = p; }
    vec_view subvector(size_t coloffset, size_t cols) const { return vec_view(ptr.subvector(coloffset, cols)); }

    // automatic conversion
    operator       cvec_view&()       { return *reinterpret_cast<cvec_view*>(this); }
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }

    // common vector API class members
    CONST_VECTOR_CLASS_MEMBERS
    VECTOR_CLASS_MEMBERS(vec_view,const)
    VECTOR_CLASS_MEMBERS(vec_view,)

    // vector result
    template<typename F> const vec_view& operator=(vector_result<F>&& vr) const { vr.r(ptr); return *this; }
    template<typename F>       vec_view& operator=(vector_result<F>&& vr)       { vr.r(ptr); return *this; }
};

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

    // view management
    void reset(const cvi_ptr& p) { ptr = p; }
    cvec_view_it subvector(size_t coloffset, size_t cols) const { return cvec_view_it(ptr.subvectorit(coloffset, cols)); }

    // automatic conversion
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }
          cvec_view_it& operator*()       { return *this; }
    const cvec_view_it& operator*() const { return *this; }
          cvec_view_it* operator->()       { return this; }
    const cvec_view_it* operator->() const { return this; }

    // common vector API class members
    CONST_VECTOR_ITERATOR_CLASS_MEMBERS(cvec_view_it)
    CONST_VECTOR_CLASS_MEMBERS
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

    // view management
    void reset(const vi_ptr& p) { ptr = p; }
    vec_view_it subvector(size_t coloffset, size_t cols) const { return vec_view_it(ptr.subvectorit(coloffset, cols)); }

    // automatic conversion
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }
    operator const vec_view&() const { return *reinterpret_cast<const vec_view*>(this); }
    operator       cvec_view_it&()       { return *reinterpret_cast<cvec_view_it*>(this); }
    operator const cvec_view_it&() const { return *reinterpret_cast<const cvec_view_it*>(this); }
          vec_view_it& operator*()       { return *this; }
    const vec_view_it& operator*() const { return *this; }
          vec_view_it* operator->()       { return this; }
    const vec_view_it* operator->() const { return this; }

    // common vector API class members
    CONST_VECTOR_ITERATOR_CLASS_MEMBERS(vec_view_it)
    CONST_VECTOR_CLASS_MEMBERS
    VECTOR_CLASS_MEMBERS(vec_view_it,const)
    VECTOR_CLASS_MEMBERS(vec_view_it,)

    // vector result
    template<typename F> const vec_view_it& operator=(vector_result<F>&& vr) const { vr.r(ptr); return *this; }
    template<typename F>       vec_view_it& operator=(vector_result<F>&& vr)       { vr.r(ptr); return *this; }
};



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

    // view management
    void reset(const cm_ptr& p) { ptr = p; }
    cvec_view_it subvector(size_t row, size_t coloffset, size_t cols) const { return cvec_view_it(ptr.subvectorit(row, coloffset, cols)); }
    cmat_view submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const { return cmat_view(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    cvec_view_it operator[](size_t r) const { return cvec_view_it(ptr.subvectorit(r)); }
    cvec_view_it operator()(size_t r) const { return cvec_view_it(ptr.subvectorit(r)); }
    cvec_view_it begin() const { return cvec_view_it(ptr.subvectorit(0)); }
    cvec_view_it end()   const { return cvec_view_it(ptr.subvectorit(rows())); }

    // automatic conversion

    // common matrix API class members
    CONST_MATRIX_CLASS_MEMBERS
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

    // view management
    void reset(const m_ptr& p) { ptr = p; }
    vec_view_it subvector(size_t row, size_t coloffset, size_t cols) const { return vec_view_it(ptr.subvectorit(row, coloffset, cols)); }
    mat_view submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const { return mat_view(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    vec_view_it operator[](size_t r) const { return vec_view_it(ptr.subvectorit(r)); }
    vec_view_it operator()(size_t r) const { return vec_view_it(ptr.subvectorit(r)); }
    vec_view_it begin() const { return vec_view_it(ptr.subvectorit(0)); }
    vec_view_it end()   const { return vec_view_it(ptr.subvectorit(rows())); }

    // automatic conversion
    operator       cmat_view&()       { return *reinterpret_cast<cmat_view*>(this); }
    operator const cmat_view&() const { return *reinterpret_cast<const cmat_view*>(this); }

    // common matrix API class members
    CONST_MATRIX_CLASS_MEMBERS
    MATRIX_CLASS_MEMBERS(mat_view,const)
    MATRIX_CLASS_MEMBERS(mat_view,)

    // matrix result
    template<typename F> const mat_view& operator=(matrix_result<F>&& mr) const { mr.r(ptr); return *this; }
    template<typename F>       mat_view& operator=(matrix_result<F>&& mr)       { mr.r(ptr); return *this; }
};


class vec
{
private:
    v_ptr ptr;
    std::vector<uint64_t> mem;
public:
    void assign(const cvec_view& v)
    {
        resize(v.columns());
        copy(v);
    }
    
    void resize(size_t _columns, bool value = false)
    {
        if (_columns == columns())
            return;
        if (mem.empty())
        {
            size_t rowwords = (_columns+63)/64;
            mem.resize(rowwords, value ? ~uint64_t(0) : uint64_t(0));
            ptr.reset(&mem[0], _columns);
            return;
        }
        vec tmp(_columns, value);
        size_t mincols = std::min(_columns, columns());
        tmp.subvector(0, mincols).copy(subvector(0, mincols));
        tmp.swap(*this);
    }
    
    void swap(vec& v)
    {
        std::swap(mem, v.mem);
        std::swap(ptr, v.ptr);
    }

    vec(): ptr(), mem() {}
    vec(size_t _columns, bool value = false): ptr(), mem() { resize(_columns,value); }
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    vec(const vec& v): ptr(), mem() { assign(v); }
    vec(const cvec_view& v): ptr(), mem() { assign(v); }
    vec(vec&& v): ptr(), mem() { swap(v); }
    vec& operator=(const vec& v) { assign(v); return *this; }
    vec& operator=(const cvec_view& v) { assign(v); return *this; }
    vec& operator=(vec&& v) { swap(v); return *this; }

    // view management
    cvec_view subvector(size_t coloffset, size_t cols) const { return cvec_view(ptr.subvector(coloffset, cols)); }
     vec_view subvector(size_t coloffset, size_t cols)       { return vec_view(ptr.subvector(coloffset, cols)); }
    
    // automatic conversion
    operator const cvec_view&() const { return *reinterpret_cast<const cvec_view*>(this); }
    operator const  vec_view&()       { return *reinterpret_cast<const vec_view*>(this); }

    // common matrix API class members
    CONST_VECTOR_CLASS_MEMBERS
    VECTOR_CLASS_MEMBERS(vec,)

    // vector result
    template<typename F>       vec& operator=(vector_result<F>&& vr)       { vr.r.resize_me(*this); vr.r(ptr); return *this; }
};

class mat
{
private:
    m_ptr ptr;
    std::vector<uint64_t> mem;
public:
    void assign(const cmat_view& m)
    {
        resize(m.rows(), m.columns());
        this->copy(m);
    }
    
    void resize(size_t _rows, size_t _columns, bool value = false)
    {
        if (_rows == rows() && _columns == columns())
            return;
        if (mem.empty())
        {
            size_t rowwords = (_columns+63)/64;
            mem.resize(_rows * rowwords, value ? ~uint64_t(0) : uint64_t(0));
            ptr.reset(&mem[0], _columns, rowwords, _rows);
            return;
        }
        mat tmp(_rows, _columns, value);
        size_t minrows = std::min(_rows, rows()), mincols = std::min(_columns, columns());
        tmp.submatrix(0, minrows, 0, mincols).copy(this->submatrix(0, minrows, 0, mincols));
        tmp.swap(*this);
    }
    
    void swap(mat& m)
    {
        std::swap(mem, m.mem);
        std::swap(ptr, m.ptr);
    }

    mat(): ptr(), mem() {}
    mat(const size_t rows, const size_t columns, bool value = false): ptr(), mem() { resize(rows, columns, value); }
    
    // copy/move constructors & assignment copy/move the contents
    mat(const mat& m): ptr(), mem() { assign(m); }
    mat(const cmat_view& m): ptr(), mem() { assign(m); }
    mat(mat&& m): ptr(), mem() { swap(m); }
    mat& operator=(const mat& m) { assign(m); return *this; }
    mat& operator=(const cmat_view& m) { assign(m); return *this; }
    mat& operator=(mat&& m) { assign(m); return *this; }
    template<typename F>
    mat(matrix_result<F>&& mr) { mr.r.resize_me(*this); mr.r(ptr); }

    // view management
    cvec_view_it subvector(size_t row, size_t coloffset, size_t cols) const { return cvec_view_it(ptr.subvectorit(row, coloffset, cols)); }
     vec_view_it subvector(size_t row, size_t coloffset, size_t cols)       { return  vec_view_it(ptr.subvectorit(row, coloffset, cols)); }
    cmat_view submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols) const { return cmat_view(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
     mat_view submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols)       { return  mat_view(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    cvec_view_it operator[](size_t r) const { return cvec_view_it(ptr.subvectorit(r)); }
    cvec_view_it operator()(size_t r) const { return cvec_view_it(ptr.subvectorit(r)); }
     vec_view_it operator[](size_t r)       { return  vec_view_it(ptr.subvectorit(r)); }
     vec_view_it operator()(size_t r)       { return  vec_view_it(ptr.subvectorit(r)); }
    cvec_view_it begin() const { return cvec_view_it(ptr.subvectorit(0)); }
    cvec_view_it end()   const { return cvec_view_it(ptr.subvectorit(rows())); }
     vec_view_it begin()       { return vec_view_it(ptr.subvectorit(0)); }
     vec_view_it end()         { return vec_view_it(ptr.subvectorit(rows())); }

    // automatic conversion
    operator const cmat_view&() const { return *reinterpret_cast<const cmat_view*>(this); }
    operator const  mat_view&()       { return *reinterpret_cast<const  mat_view*>(this); }

    // common matrix API class members
    CONST_MATRIX_CLASS_MEMBERS
    MATRIX_CLASS_MEMBERS(mat,)

    // vector result
    template<typename F>       mat& operator=(matrix_result<F>&& mr)       { mr.r.resize_me(*this); mr.r(ptr); return *this; }
};

bool operator==(const cvec_view& v1, const cvec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view& v1, const cvec_view& v2) { return v1.ptr != v2.ptr; }
bool operator==(const cvec_view& v1, const  vec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view& v1, const  vec_view& v2) { return v1.ptr != v2.ptr; }
bool operator==(const  vec_view& v1, const cvec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const  vec_view& v1, const cvec_view& v2) { return v1.ptr != v2.ptr; }
bool operator==(const  vec_view& v1, const  vec_view& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const  vec_view& v1, const  vec_view& v2) { return v1.ptr != v2.ptr; }

bool operator==(const cvec_view_it& v1, const cvec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view_it& v1, const cvec_view_it& v2) { return v1.ptr != v2.ptr; }
bool operator==(const cvec_view_it& v1, const  vec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const cvec_view_it& v1, const  vec_view_it& v2) { return v1.ptr != v2.ptr; }
bool operator==(const  vec_view_it& v1, const cvec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const  vec_view_it& v1, const cvec_view_it& v2) { return v1.ptr != v2.ptr; }
bool operator==(const  vec_view_it& v1, const  vec_view_it& v2) { return v1.ptr == v2.ptr; }
bool operator!=(const  vec_view_it& v1, const  vec_view_it& v2) { return v1.ptr != v2.ptr; }

bool operator==(const cmat_view& m1, const cmat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const cmat_view& m1, const cmat_view& m2) { return m1.ptr != m2.ptr; }
bool operator==(const cmat_view& m1, const  mat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const cmat_view& m1, const  mat_view& m2) { return m1.ptr != m2.ptr; }
bool operator==(const  mat_view& m1, const cmat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const  mat_view& m1, const cmat_view& m2) { return m1.ptr != m2.ptr; }
bool operator==(const  mat_view& m1, const  mat_view& m2) { return m1.ptr == m2.ptr; }
bool operator!=(const  mat_view& m1, const  mat_view& m2) { return m1.ptr != m2.ptr; }

bool operator==(const mat& m1, const mat& m2)       { return m1.isequal(m2); }
bool operator!=(const mat& m1, const mat& m2)       { return !m1.isequal(m2); }
bool operator==(const cmat_view& m1, const mat& m2) { return m1.isequal(m2); }
bool operator!=(const cmat_view& m1, const mat& m2) { return !m1.isequal(m2); }
bool operator==(const mat& m1, const cmat_view& m2) { return m1.isequal(m2); }
bool operator!=(const mat& m1, const cmat_view& m2) { return !m1.isequal(m2); }

std::ostream& operator<<(std::ostream& o, const cvec_view& v) { v_print(o, v.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const  vec_view& v) { v_print(o, v.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const cvec_view_it& v) { v_print(o, v.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const  vec_view_it& v) { v_print(o, v.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const cmat_view& m) { m_print(o, m.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const  mat_view& m) { m_print(o, m.ptr); return o; }
std::ostream& operator<<(std::ostream& o, const  mat& m) { return o << static_cast<const cmat_view&>(m); }


template<void f(const v_ptr&, const cv_ptr&)>
struct v_ptr_op2_result
{
    const cv_ptr* v2;
    v_ptr_op2_result(): v2(nullptr) {}
    v_ptr_op2_result(const cv_ptr* _v2) { v2 = _v2; }
    v_ptr_op2_result(const v_ptr_op2_result&) = default;
    v_ptr_op2_result(v_ptr_op2_result&&) = default;
    v_ptr_op2_result& operator=(const v_ptr_op2_result&) = default;
    v_ptr_op2_result& operator=(v_ptr_op2_result&&) = default;
    
    void operator()(const v_ptr& v1) {  f(v1,*v2); }
    void resize_me(vec& v) { v.resize(v2->columns); }
};
template<void f(const v_ptr&, const cv_ptr&, const cv_ptr&)>
struct v_ptr_op3_result
{
    const cv_ptr* v2;
    const cv_ptr* v3;
    v_ptr_op3_result(): v2(nullptr), v3(nullptr) {}
    v_ptr_op3_result(const cv_ptr* _v2, const cv_ptr* _v3) { v2 = _v2; v3 = _v3; }
    v_ptr_op3_result(const v_ptr_op3_result&) = default;
    v_ptr_op3_result(v_ptr_op3_result&&) = default;
    v_ptr_op3_result& operator=(const v_ptr_op3_result&) = default;
    v_ptr_op3_result& operator=(v_ptr_op3_result&&) = default;
    
    void operator()(const v_ptr& v1) {  f(v1,*v2,*v3); }
    void resize_me(vec& v) { v.resize(v2->columns); }
};

template<void f(const m_ptr&, const cm_ptr&)>
struct m_ptr_op2_result
{
    const cm_ptr* m2;
    m_ptr_op2_result(): m2(nullptr) {}
    m_ptr_op2_result(const cm_ptr* _m2) { m2 = _m2; }
    m_ptr_op2_result(const m_ptr_op2_result&) = default;
    m_ptr_op2_result(m_ptr_op2_result&&) = default;
    m_ptr_op2_result& operator=(const m_ptr_op2_result&) = default;
    m_ptr_op2_result& operator=(m_ptr_op2_result&&) = default;
    
    void operator()(const m_ptr& m1) {  f(m1,*m2); }
    void resize_me(mat& m) { m.resize(m2->rows, m2->columns); }
};
template<void f(const m_ptr&, const cm_ptr&, const cm_ptr&)>
struct m_ptr_op3_result
{
    const cm_ptr* m2;
    const cm_ptr* m3;
    m_ptr_op3_result(): m2(nullptr), m3(nullptr) {}
    m_ptr_op3_result(const cm_ptr* _m2, const cm_ptr* _m3) { m2 = _m2; m3 = _m3; }
    m_ptr_op3_result(const m_ptr_op3_result&) = default;
    m_ptr_op3_result(m_ptr_op3_result&&) = default;
    m_ptr_op3_result& operator=(const m_ptr_op3_result&) = default;
    m_ptr_op3_result& operator=(m_ptr_op3_result&&) = default;
    
    void operator()(const m_ptr& m1) {  f(m1,*m2,*m3); }
    void resize_me(mat& m) { m.resize(m2->rows, m2->columns); }
};
struct m_ptr_transpose_result
{
    const cm_ptr* m2;
    m_ptr_transpose_result(): m2(nullptr) {}
    m_ptr_transpose_result(const cm_ptr* _m2) { m2 = _m2; }
    m_ptr_transpose_result(const m_ptr_transpose_result&) = default;
    m_ptr_transpose_result(m_ptr_transpose_result&&) = default;
    m_ptr_transpose_result& operator=(const m_ptr_transpose_result&) = default;
    m_ptr_transpose_result& operator=(m_ptr_transpose_result&&) = default;
    
    void operator()(const m_ptr& m1) {  m_transpose(m1,*m2); }
    void resize_me(mat& m) { m.resize(m2->columns, m2->rows); }
};

static inline vector_result<v_ptr_op2_result<detail::v_copy   >> v_copy   (const cvec_view& v2) { return vector_result<v_ptr_op2_result<detail::v_copy>>(&v2.ptr); }
static inline vector_result<v_ptr_op2_result<detail::v_copynot>> v_copynot(const cvec_view& v2) { return vector_result<v_ptr_op2_result<detail::v_copynot>>(&v2.ptr); }

static inline vector_result<v_ptr_op3_result<detail::v_and>> v_and(const cvec_view& v2, const cvec_view& v3) { return vector_result<v_ptr_op3_result<detail::v_and>>(&v2.ptr, &v3.ptr); }
static inline vector_result<v_ptr_op3_result<detail::v_or >> v_or (const cvec_view& v2, const cvec_view& v3) { return vector_result<v_ptr_op3_result<detail::v_or >>(&v2.ptr, &v3.ptr); }
static inline vector_result<v_ptr_op3_result<detail::v_xor>> v_xor(const cvec_view& v2, const cvec_view& v3) { return vector_result<v_ptr_op3_result<detail::v_xor>>(&v2.ptr, &v3.ptr); }
static inline vector_result<v_ptr_op3_result<detail::v_and>> operator&(const cvec_view& v2, const cvec_view& v3) { return vector_result<v_ptr_op3_result<detail::v_and>>(&v2.ptr, &v3.ptr); }
static inline vector_result<v_ptr_op3_result<detail::v_or >> operator|(const cvec_view& v2, const cvec_view& v3) { return vector_result<v_ptr_op3_result<detail::v_or >>(&v2.ptr, &v3.ptr); }
static inline vector_result<v_ptr_op3_result<detail::v_xor>> operator^(const cvec_view& v2, const cvec_view& v3) { return vector_result<v_ptr_op3_result<detail::v_xor>>(&v2.ptr, &v3.ptr); }


static inline matrix_result<m_ptr_op2_result<detail::m_copy   >> m_copy     (const cmat_view& m2) { return matrix_result<m_ptr_op2_result<detail::m_copy>>(&m2.ptr); }
static inline matrix_result<m_ptr_op2_result<detail::m_copynot>> m_copynot  (const cmat_view& m2) { return matrix_result<m_ptr_op2_result<detail::m_copynot>>(&m2.ptr); }
static inline matrix_result<m_ptr_transpose_result>              m_transpose(const cmat_view& m2) { return matrix_result<m_ptr_transpose_result>(&m2.ptr); }

static inline matrix_result<m_ptr_op3_result<detail::m_and>> m_and(const cmat_view& m2, const cmat_view& m3) { return matrix_result<m_ptr_op3_result<detail::m_and>>(&m2.ptr, &m3.ptr); }
static inline matrix_result<m_ptr_op3_result<detail::m_or >> m_or (const cmat_view& m2, const cmat_view& m3) { return matrix_result<m_ptr_op3_result<detail::m_or >>(&m2.ptr, &m3.ptr); }
static inline matrix_result<m_ptr_op3_result<detail::m_xor>> m_xor(const cmat_view& m2, const cmat_view& m3) { return matrix_result<m_ptr_op3_result<detail::m_xor>>(&m2.ptr, &m3.ptr); }
static inline matrix_result<m_ptr_op3_result<detail::m_and>> operator&(const cmat_view& m2, const cmat_view& m3) { return matrix_result<m_ptr_op3_result<detail::m_and>>(&m2.ptr, &m3.ptr); }
static inline matrix_result<m_ptr_op3_result<detail::m_or >> operator|(const cmat_view& m2, const cmat_view& m3) { return matrix_result<m_ptr_op3_result<detail::m_or >>(&m2.ptr, &m3.ptr); }
static inline matrix_result<m_ptr_op3_result<detail::m_xor>> operator^(const cmat_view& m2, const cmat_view& m3) { return matrix_result<m_ptr_op3_result<detail::m_xor>>(&m2.ptr, &m3.ptr); }


static inline size_t hammingweight(const cmat_view& m) { return m.hw(); }
static inline size_t hammingweight(const cvec_view& v) { return v.hw(); }
static inline size_t hammingweight_and(const cvec_view& v1, const cvec_view& v2) { return v_hw_and(v1.ptr, v2.ptr); }
static inline size_t hammingweight_xor(const cvec_view& v1, const cvec_view& v2) { return v_hw_xor(v1.ptr, v2.ptr); }
static inline size_t hammingweight_or (const cvec_view& v1, const cvec_view& v2) { return v_hw_or(v1.ptr, v2.ptr); }

template<typename Func = std::function<bool(size_t,size_t)>>
void fill(const mat_view& m, Func& f)
{
    for (size_t r = 0; r < m.rows(); ++r)
        for (size_t c = 0; c < m.columns(); ++c)
            m.setbit(r,c, f(r,c));
}

template<typename Func = std::function<bool(size_t)>>
void fill(const vec_view& m, Func& f)
{
    for (size_t c = 0; c < m.columns(); ++c)
        m.setbit(c, f(c));
}

template<typename Func = std::function<uint64_t(size_t,size_t)>>
void fillword(const mat_view& m, Func& f)
{
    const size_t words = m.rowwords();
    for (size_t r = 0; r < m.rows(); ++r)
    {
        uint64_t* first = m.ptr.data(r);
        uint64_t* last = first + words;
        for (size_t w = 0; first != last; ++first,++w)
            *first = f(r,w);
    }
}

template<typename Func = std::function<uint64_t(size_t)>>
void fillword(const vec_view& m, Func& f)
{
    const size_t words = m.rowwords();
    uint64_t* first = m.ptr.data(0);
    uint64_t* last = first + words;
    for (size_t w = 0; first != last; ++first,++w)
        *first = f(w);
}


template<typename Generator>
void fillgenerator(const mat_view& m, Generator& g)
{
    const size_t words = m.rowwords();
    for (size_t r = 0; r < m.rows(); ++r)
    {
        uint64_t* first = m.ptr.data(r);
        uint64_t* last = first + words;
        for (; first != last; ++first)
            g(*first);
    }
}

template<typename Generator>
void fillgenerator(const vec_view& m, Generator& g)
{
    const size_t words = m.rowwords();
    uint64_t* first = m.ptr.data(0);
    uint64_t* last = first + words;
    for (; first != last; ++first)
        g(*first);
}

struct mccl_base_random_generator
{
    mccl_base_random_generator()
        : rnd(std::random_device()())
    {
    }
    void operator()(uint64_t& word)
    {
        word = rnd();
    }
    std::mt19937_64 rnd;
};

void fillrandom(const mat_view& m)
{
    mccl_base_random_generator gen;
    fillgenerator(m, gen);
}

void fillrandom(const vec_view& m)
{
    mccl_base_random_generator gen;
    fillgenerator(m, gen);
}

// full row reduction of matrix m over columns [column_start,column_end)
// pivots may be selected from rows [pivot_start,rows())
// returns pivotend = pivot_start + nrnewrowpivots
size_t echelonize(const mat_view& m, size_t column_start = 0, size_t column_end = ~size_t(0), size_t pivot_start = 0)
{
    if (column_end > m.columns())
        column_end = m.columns();
    for (size_t c = column_start; c < column_end; ++c)
    {
        // find pivot for column c
        size_t p = pivot_start;
        for (; p < m.rows() && m(p,c) == false; ++p)
            ;
        // if no pivot found the continue with next column
        if (p >= m.rows())
            continue;
        // swap row if necessary
        if (p != pivot_start)
            m[p].swap(m[pivot_start]);
        // reduce column c
        auto pivotrow = m[pivot_start];
        auto mrowit = m[0];
        //std::cerr << (mrowit) << std::endl;
        for (size_t r = 0; r < pivot_start; ++r,++mrowit)
            if (m(r,c))
                mrowit.vxor(pivotrow);
        // skip pivotrow itself
        ++mrowit;
        for (size_t r = pivot_start+1; r < m.rows(); ++r,++mrowit)
            if (m(r,c))
                mrowit.vxor(pivotrow);
        // increase pivot_start for next column
        ++pivot_start;
    }
    return pivot_start;
}


MCCL_END_NAMESPACE

#endif
