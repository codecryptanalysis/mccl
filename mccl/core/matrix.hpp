#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix_base.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <array>
#include <vector>
#include <iostream>
#include <functional>

MCCL_BEGIN_NAMESPACE

// == base vector & matrix traits ==
template<typename _block_tag>
struct vector_trait
{
    typedef _block_tag this_block_tag;
    static const bool is_vector = true;
};
template<typename _block_tag>
struct matrix_trait
{
    typedef _block_tag this_block_tag;
    static const bool is_matrix = true;
};
// use as last template parameters to enable function resolution only if type t is a MCCL matrix/vector type
#define MCCL_ENABLE_IF_VECTOR(t) typename std::enable_if< t ::is_vector,bool>::type = true
#define MCCL_ENABLE_IF_MATRIX(t) typename std::enable_if< t ::is_matrix,bool>::type = true

// = vector view classes =
// These only point to a preallocated vector in memory
// Copy/move constructor/assignment only alter the view, not the vector contents
//   use vec_view.copy(src) etc. to modify view's contents
// cvec_view points to const vector contents, vec_view points to non-const vector contents
// Note that 'const vec_view' like a 'const pointer_t' still allows to modify the vector contents
// Allows assigning a vector_result:
//   however it must match the result vector dimension
template<typename _block_tag = default_block_tag> class vec_view_t;
template<typename _block_tag = default_block_tag> class cvec_view_t;

typedef  vec_view_t<default_block_tag>  vec_view;
typedef cvec_view_t<default_block_tag> cvec_view;

//template<size_t bits, bool masked> using cvec_view_t = cvec_view_t<block_tag<bits,masked>>;
//template<size_t bits, bool masked> using  vec_view_t =  vec_view_t<block_tag<bits,masked>>;

// Extension of vec_view / cvec_view with rowstride
// and pointer operators ++,--,+=,-=,+,- that change the view by # rows forward/backward
template<typename _block_tag = default_block_tag> class vec_view_it_t;
template<typename _block_tag = default_block_tag> class cvec_view_it_t;

typedef  vec_view_it_t<default_block_tag>  vec_view_it;
typedef cvec_view_it_t<default_block_tag> cvec_view_it;

//template<size_t bits, bool masked> using cvec_view_it_t = cvec_view_it_t<block_tag<bits,masked>>;
//template<size_t bits, bool masked> using  vec_view_it_t =  vec_view_it_t<block_tag<bits,masked>>;

// = matrix view classes =
// These only point to a preallocated matrix in memory
// Copy/move constructor/assignment only alter the view, not the matrix contents
//   use mat_view.copy(src) etc. to modify view's contents
// cmat_view points to const matrix contents, mat_view points to non-const matrix contents
// Note that 'const mat_view' like a 'const pointer_t' still allows to modify the matrix contents
// Allows assigning a matrix_result:
//   however it must match the result matrix dimensions
template<typename _block_tag = default_block_tag> class mat_view_t;
template<typename _block_tag = default_block_tag> class cmat_view_t;

typedef  mat_view_t<default_block_tag>  mat_view;
typedef cmat_view_t<default_block_tag> cmat_view;

//template<size_t bits, bool masked> using cmat_view_t = cmat_view_t<block_tag<bits,masked>>;
//template<size_t bits, bool masked> using  mat_view_t =  mat_view_t<block_tag<bits,masked>>;

// = matrix & vector classes =
// maintains memory for a matrix/vector of specified dimensions
// behaves as vec_view/mat_view when non-const
// behaves as cvec_view/cmat_view when const
// except:
// 1) copy/move constructor/assignment from cvec_view/cmat_view
//    automatically resizes and copies content
// 2) when assigning a vector_result/matrix_result:
//    then it will automatically resize to the result vector/matrix
template<typename _block_tag = default_block_tag> class vec_t;
template<typename _block_tag = default_block_tag> class mat_t;

typedef vec_t<block_tag<256,false>> vec;
typedef mat_t<block_tag<256,false>> mat;

//template<size_t bits, bool masked> using  vec_t = vec_t<block_tag<bits,masked>>;
//template<size_t bits, bool masked> using  mat_t = mat_t<block_tag<bits,masked>>;

// some functions have specializations that can be used by adding a tag
// e.g. v.hw(fullword_tag()), v.copy(v2, aligned256_tag()), v.vnot(v2, aligned512_tag())
// fullword_tag: 
//   only for vectors/matrices with # columns a multiple of 64
//   benefit: simpler loop and no special last word overhead
// aligned256_tag/aligned512_tag: 
//   only for vectors/matrices with # columns a multiple of 256/512
//   also requires column offset with respect to original vector/matrix to be a multiple of 256/512
//   benefit: automatic SIMD processing for 256=1 x avx2 register, 512=2 x avx2 register
//   benefit2: aligned512_tag corresponds to cachelines!

#define VECTOR_BLOCK_TAG_CLASS_MEMBERS(classname) \
    static inline this_block_tag tag() { return this_block_tag(); } \
    static const size_t block_bits = this_block_tag::bits; \
    static const bool maskedlastblock = this_block_tag::maskedlastblock; \
    template<typename tag> classname(const classname<tag>& v) \
       : ptr(v.ptr) \
    { \
       static_assert(tag::bits >= this_block_tag::bits && (this_block_tag::maskedlastblock == true || tag::maskedlastblock == false), "block_tag incompatibility"); \
    } \
    template<typename tag> classname(classname<tag>&& v) \
       : ptr(std::move(v.ptr)) \
    { \
       static_assert(tag::bits >= this_block_tag::bits && (this_block_tag::maskedlastblock == true || tag::maskedlastblock == false), "block_tag incompatibility"); \
    } \
    template<typename tag> void reset(const classname<tag>& v) \
    { \
       static_assert(tag::bits >= this_block_tag::bits && (this_block_tag::maskedlastblock == true || tag::maskedlastblock == false), "block_tag incompatibility"); \
       ptr = v.ptr; \
    }

    
// common class members for: cvec_view, vec_view, vec_view_it, cvec_view_it, vec
#define CONST_VECTOR_CLASS_MEMBERS \
    auto wordptr() const -> decltype(ptr.ptr) { return ptr.ptr; } \
    auto blockptr() const \
      -> decltype(make_block_ptr(ptr.ptr, tag())) \
      { return    make_block_ptr(ptr.ptr, tag()); } \
    size_t columns() const { return ptr.columns; } \
    size_t rowwords() const { return (ptr.columns+63)/64; } \
    size_t rowblocks() const { return (ptr.columns+block_bits-1)/block_bits; } \
    size_t hw() const { return detail::v_hw(ptr); } \
    bool operator[](size_t c) const { return detail::v_getbit(ptr,c); } \
    bool operator()(size_t c) const { return detail::v_getbit(ptr,c); } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> bool isequal(const vector_t& v2) const { return detail::v_isequal(ptr,v2.ptr,tag(),v2.tag()); } \
    template<size_t bits, bool maskedlastword> \
    auto blockptr(block_tag<bits,maskedlastword>) const \
      -> decltype(make_block_ptr(ptr.ptr, block_tag<bits,maskedlastword>())) \
      { return    make_block_ptr(ptr.ptr, block_tag<bits,maskedlastword>()); } \
    template<size_t bits, bool maskedlastword> \
    size_t rowblocks(block_tag<bits,maskedlastword>) const { return (ptr.columns+bits-1)/bits; }

// common class members for: vec_view, vec_view_it, vec
// cnst = '' / 'const' to allow for const and non-const versions
// vec_view & vec_view_it have both const and non-const versions
// vec only has non-const versions
#define VECTOR_CLASS_MEMBERS(vectype,cnst) \
    cnst vectype& clear ()                        cnst { detail::v_clear(ptr, tag()); return *this; } \
    cnst vectype& vnot  ()                        cnst { detail::v_not  (ptr, tag()); return *this; } \
    cnst vectype& set   ()                        cnst { detail::v_set  (ptr, tag()); return *this; } \
    cnst vectype& set   (bool b)                  cnst { detail::v_set  (ptr, b, tag()); return *this; } \
    cnst vectype& clearbit(size_t c)              cnst { detail::v_clearbit(ptr, c); return *this; } \
    cnst vectype& flipbit(size_t c)               cnst { detail::v_flipbit(ptr, c); return *this; } \
    cnst vectype& setbit(size_t c)                cnst { detail::v_setbit(ptr, c); return *this; } \
    cnst vectype& setbit(size_t c, bool b)        cnst { detail::v_setbit(ptr, c, b); return *this; } \
    cnst vectype& setcolumns(size_t c_off, size_t c_cnt, bool b)    cnst { detail::v_setcolumns(ptr, c_off, c_cnt, b); return *this; } \
    cnst vectype& setcolumns(size_t c_off, size_t c_cnt)            cnst { detail::v_setcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst vectype& clearcolumns(size_t c_off, size_t c_cnt)          cnst { detail::v_clearcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst vectype& flipcolumns(size_t c_off, size_t c_cnt)           cnst { detail::v_flipcolumns(ptr, c_off, c_cnt); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& swap  (const vector_t& v2)     cnst { detail::v_swap(ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& copy  (const vector_t& v2)     cnst { detail::v_copy (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vnot  (const vector_t& v2)     cnst { detail::v_copynot(ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vxor  (const vector_t& v2)     cnst { detail::v_xor  (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vand  (const vector_t& v2)     cnst { detail::v_and  (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vor   (const vector_t& v2)     cnst { detail::v_or   (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vnxor (const vector_t& v2)     cnst { detail::v_nxor (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vnand (const vector_t& v2)     cnst { detail::v_nand (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vnor  (const vector_t& v2)     cnst { detail::v_nor  (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vandin(const vector_t& v2)     cnst { detail::v_andin(ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vandni(const vector_t& v2)     cnst { detail::v_andni(ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vorin (const vector_t& v2)     cnst { detail::v_orin (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& vorni (const vector_t& v2)     cnst { detail::v_orni (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& operator&=(const vector_t& v2) cnst { detail::v_and  (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& operator^=(const vector_t& v2) cnst { detail::v_xor  (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> cnst vectype& operator|=(const vector_t& v2) cnst { detail::v_or   (ptr, v2.ptr, tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vxor  (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_xor  (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vand  (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_and  (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vor   (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_or   (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vnxor (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_nxor (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vnand (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_nand (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vnor  (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_nor  (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vandin(const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_andin(ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vandni(const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_andni(ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vorin (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_orin (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; } \
    template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)> cnst vectype& vorni (const vector1_t& v1, const vector2_t& v2)  cnst { detail::v_orni (ptr, v1.ptr, v2.ptr, tag(), v1.tag(), v2.tag()); return *this; }

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
    auto wordptr(size_t r = 0) const -> decltype(ptr.ptr) { return ptr.ptr + r*ptr.stride; } \
    auto blockptr(size_t r = 0) const \
      -> decltype(make_block_ptr(ptr.ptr + r*ptr.stride, tag())) \
      { return    make_block_ptr(ptr.ptr + r*ptr.stride, tag()); } \
    size_t columns() const { return ptr.columns; } \
    size_t rowwords() const { return (ptr.columns+63)/64; } \
    size_t rowblocks() const { return (ptr.columns+block_bits-1)/block_bits; } \
    size_t rows() const { return ptr.rows; } \
    size_t stride() const { return ptr.stride; } \
    size_t blockstride() const { return ptr.stride / (block_bits/64); } \
    size_t hw() const { return detail::m_hw(ptr); } \
    bool operator()(size_t r, size_t c) const { return detail::m_getbit(ptr,r,c); } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> bool isequal(const matrix_t& m2) const { return detail::m_isequal(ptr, m2.ptr, tag(), m2.tag()); } \
    template<size_t bits, bool masked> auto blockptr(block_tag<bits,masked>) const \
      -> decltype(make_block_ptr(ptr.ptr, block_tag<bits,masked>())) \
      { return    make_block_ptr(ptr.ptr, block_tag<bits,masked>()); } \
    template<size_t bits, bool masked> auto blockptr(size_t r, block_tag<bits,masked>) const \
      -> decltype(make_block_ptr(ptr.ptr + r*ptr.stride, block_tag<bits,masked>())) \
      { return    make_block_ptr(ptr.ptr + r*ptr.stride, block_tag<bits,masked>()); } \
    template<size_t bits, bool masked> size_t rowblocks(block_tag<bits,masked>) const { return (ptr.columns+bits-1)/bits; } \
    template<size_t bits, bool masked> size_t blockstride(block_tag<bits,masked>) const { return ptr.stride / (bits/64); }

// common class members for: mat_view, mat
// cnst = '' / 'const' to allow for const and non-const versions
// mat_view has both const and non-const versions
// mat only has non-const versions
#define MATRIX_CLASS_MEMBERS(mattype,cnst) \
    cnst mattype& clearbit(size_t r, size_t c)       cnst { detail::m_clearbit(ptr, r, c); return *this; } \
    cnst mattype& flipbit(size_t r, size_t c)        cnst { detail::m_flipbit(ptr, r, c); return *this; } \
    cnst mattype& setbit(size_t r, size_t c)         cnst { detail::m_setbit(ptr, r, c); return *this; } \
    cnst mattype& setbit(size_t r, size_t c, bool b) cnst { detail::m_setbit(ptr, r, c, b); return *this; } \
    cnst mattype& transpose(const cmat_view& src)    cnst { detail::m_transpose(ptr, src.ptr); return *this; } \
    cnst mattype& clear()                            cnst { detail::m_clear(ptr, this_block_tag()); return *this; } \
    cnst mattype& set(bool b = true)                 cnst { detail::m_set(ptr, b, this_block_tag()); return *this; } \
    cnst mattype& mnot()                             cnst { detail::m_not(ptr, this_block_tag()); return *this; } \
    cnst mattype& setidentity()                      cnst { detail::m_clear(ptr, this_block_tag()); for (size_t i = 0; i < rows() && i < columns(); ++i) setbit(i,i); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& copy(const matrix_t& m2)          cnst { detail::m_copy(ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& mnot(const matrix_t& m2)          cnst { detail::m_copynot(ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& mxor(const matrix_t& m2)          cnst { detail::m_xor(ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& mand(const matrix_t& m2)          cnst { detail::m_and(ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& mor (const matrix_t& m2)          cnst { detail::m_or (ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& operator&=(const matrix_t& m2)    cnst { detail::m_and(ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& operator^=(const matrix_t& m2)    cnst { detail::m_xor(ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> cnst mattype& operator|=(const matrix_t& m2)    cnst { detail::m_or (ptr, m2.ptr, tag(), m2.tag()); return *this; } \
    template<typename matrix1_t, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix1_t), MCCL_ENABLE_IF_MATRIX(matrix2_t)> cnst mattype& mxor(const matrix1_t& m1, const matrix2_t& m2) cnst { detail::m_xor(ptr, m1.ptr, m2.ptr, tag(), m1.tag(), m2.tag()); return *this; } \
    template<typename matrix1_t, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix1_t), MCCL_ENABLE_IF_MATRIX(matrix2_t)> cnst mattype& mand(const matrix1_t& m1, const matrix2_t& m2) cnst { detail::m_and(ptr, m1.ptr, m2.ptr, tag(), m1.tag(), m2.tag()); return *this; } \
    template<typename matrix1_t, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix1_t), MCCL_ENABLE_IF_MATRIX(matrix2_t)> cnst mattype& mor (const matrix1_t& m1, const matrix2_t& m2) cnst { detail::m_or (ptr, m1.ptr, m2.ptr, tag(), m1.tag(), m2.tag()); return *this; } \
    cnst mattype& swapcolumns(size_t c1, size_t c2)              cnst { detail::m_swapcolumns(ptr, c1, c2); return *this; } \
    cnst mattype& setcolumns(size_t c_off, size_t c_cnt, bool b) cnst { detail::m_setcolumns(ptr, c_off, c_cnt, b); return *this; } \
    cnst mattype& setcolumns(size_t c_off, size_t c_cnt)         cnst { detail::m_setcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst mattype& clearcolumns(size_t c_off, size_t c_cnt)       cnst { detail::m_clearcolumns(ptr, c_off, c_cnt); return *this; } \
    cnst mattype& flipcolumns(size_t c_off, size_t c_cnt)        cnst { detail::m_flipcolumns(ptr, c_off, c_cnt); return *this; }

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
    vector_result(Args&&... args...): r(std::forward<Args>(args)...) {}
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
    matrix_result(Args&&... args...): r(std::forward<Args>(args)...) {}
};


template<typename _block_tag>
class cvec_view_t
    : public vector_trait<_block_tag>
{
public:
    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;
    
    VECTOR_BLOCK_TAG_CLASS_MEMBERS(cvec_view_t)
    
    cv_ptr ptr;
    
    cvec_view_t(): ptr() {}
    cvec_view_t(const cv_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    cvec_view_t(const cvec_view_t&  ) = default;
    cvec_view_t(      cvec_view_t&& ) = default;
    // deleted for safety, to avoid confusion
    cvec_view_t& operator=(const cvec_view_t&  ) = delete;
    cvec_view_t& operator=(      cvec_view_t&& ) = delete;
    
    // view management
    void reset(const cv_ptr& p) { ptr = p; }
    void reset(const cvec_view_t& v) { ptr = v.ptr; }

    // by default a subvector uses default_block_tag
    template<size_t bits, bool masked>
    cvec_view_t<block_tag<bits,masked>> subvector(size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cvec_view_t<block_tag<bits,masked>>(ptr.subvector(coloffset, cols)); }
    cvec_view_t<default_block_tag>      subvector(size_t coloffset, size_t cols)                         const { return cvec_view_t<default_block_tag>     (ptr.subvector(coloffset, cols)); }

    // automatic conversion
    operator const cvec_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const cvec_view_t<cond_default_block_tag>* > (this); }
    
    template<size_t bits = 64, bool masked = true>
          cvec_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<      cvec_view_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const cvec_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const cvec_view_t<block_tag<bits,masked>>*>(this); }

    // common vector API class members
    CONST_VECTOR_CLASS_MEMBERS
};

template<typename _block_tag>
class vec_view_t
    : public vector_trait<_block_tag>
{
public:
    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    VECTOR_BLOCK_TAG_CLASS_MEMBERS(vec_view_t)
    
    v_ptr ptr;

    vec_view_t(): ptr() {}
    vec_view_t(const v_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    vec_view_t(const vec_view_t& ) = default;
    vec_view_t(      vec_view_t&&) = default;
    // deleted for safety, to avoid confusion
    vec_view_t& operator=(const vec_view_t& ) = delete;
    vec_view_t& operator=(      vec_view_t&&) = delete;

    // view management
    void reset(const v_ptr& p) { ptr = p; }
    void reset(const vec_view_t& v) { ptr = v.ptr; }

    template<size_t bits, bool masked>
    vec_view_t<block_tag<bits,masked>> subvector(size_t coloffset, size_t cols, block_tag<bits,masked>) const { return vec_view_t<block_tag<bits,masked>>(ptr.subvector(coloffset, cols)); }
    vec_view_t<default_block_tag>      subvector(size_t coloffset, size_t cols)                         const { return vec_view_t<default_block_tag>     (ptr.subvector(coloffset, cols)); }

    // automatic conversion
    operator       cvec_view_t<this_block_tag>&()       { return *reinterpret_cast<      cvec_view_t<this_block_tag>*>(this); }
    operator const cvec_view_t<this_block_tag>&() const { return *reinterpret_cast<const cvec_view_t<this_block_tag>*>(this); }

    operator const cvec_view_t<cond_default_block_tag>&() const { return *reinterpret_cast< const cvec_view_t<cond_default_block_tag>* > (this); }
    operator const  vec_view_t<cond_default_block_tag>&() const { return *reinterpret_cast< const  vec_view_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
          vec_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<      vec_view_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const vec_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const vec_view_t<block_tag<bits,masked>>*>(this); }

    // common vector API class members
    CONST_VECTOR_CLASS_MEMBERS
    VECTOR_CLASS_MEMBERS(vec_view_t,const)
    VECTOR_CLASS_MEMBERS(vec_view_t,)

    // vector result
    template<typename F> const vec_view_t& operator=(vector_result<F>&& vr) const { vr.r(ptr, this_block_tag()); return *this; }
    template<typename F>       vec_view_t& operator=(vector_result<F>&& vr)       { vr.r(ptr, this_block_tag()); return *this; }
};

template<typename _block_tag>
class cvec_view_it_t
    : public vector_trait<_block_tag>
{
public:
    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    VECTOR_BLOCK_TAG_CLASS_MEMBERS(cvec_view_it_t)

    cvi_ptr ptr;
    
    cvec_view_it_t(): ptr() {}
    cvec_view_it_t(const cvi_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    cvec_view_it_t(const cvec_view_it_t&) = default;
    cvec_view_it_t(      cvec_view_it_t&&) = default;
    // deleted for safety, to avoid confusion
    cvec_view_it_t& operator=(const cvec_view_it_t& ) = delete;
    cvec_view_it_t& operator=(      cvec_view_it_t&&) = delete;

    // view management
    void reset(const cvi_ptr& p) { ptr = p; }
    void reset(const cvec_view_it_t& v) { ptr = v.ptr; }
    
    template<size_t bits, bool masked>
    cvec_view_it_t<block_tag<bits,masked>> subvector(size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cvec_view_it_t<block_tag<bits,masked>>(ptr.subvectorit(coloffset, cols)); }
    cvec_view_it_t<default_block_tag>      subvector(size_t coloffset, size_t cols)                         const { return cvec_view_it_t<default_block_tag>     (ptr.subvectorit(coloffset, cols)); }

    // automatic conversion
    operator const cvec_view_t<this_block_tag>&() const { return *reinterpret_cast<const cvec_view_t<this_block_tag>*>(this); }

    operator const cvec_view_t   <cond_default_block_tag>&() const { return *reinterpret_cast< const cvec_view_t   <cond_default_block_tag>* > (this); }
    operator const cvec_view_it_t<cond_default_block_tag>&() const { return *reinterpret_cast< const cvec_view_it_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
          cvec_view_it_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<      cvec_view_it_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const cvec_view_it_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const cvec_view_it_t<block_tag<bits,masked>>*>(this); }

          cvec_view_it_t& operator*()       { return *this; }
    const cvec_view_it_t& operator*() const { return *this; }
          cvec_view_it_t* operator->()       { return this; }
    const cvec_view_it_t* operator->() const { return this; }

    // common vector API class members
    CONST_VECTOR_ITERATOR_CLASS_MEMBERS(cvec_view_it_t)
    CONST_VECTOR_CLASS_MEMBERS
};

template<typename _block_tag>
class vec_view_it_t
    : public vector_trait<_block_tag>
{
public:
    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    VECTOR_BLOCK_TAG_CLASS_MEMBERS(vec_view_it_t)

    vi_ptr ptr;

    vec_view_it_t(): ptr() {}
    vec_view_it_t(const vi_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    vec_view_it_t(const vec_view_it_t& ) = default;
    vec_view_it_t(      vec_view_it_t&&) = default;
    // deleted for safety, to avoid confusion
    vec_view_it_t& operator=(const vec_view_it_t& ) = delete;
    vec_view_it_t& operator=(      vec_view_it_t&&) = delete;

    // view management
    void reset(const vi_ptr& p) { ptr = p; }
    void reset(const vec_view_it_t& v) { ptr = v.ptr; }
    
    template<size_t bits, bool masked>
    vec_view_it_t<block_tag<bits,masked>> subvector(size_t coloffset, size_t cols, block_tag<bits,masked>) const { return vec_view_it_t<block_tag<bits,masked>>(ptr.subvectorit(coloffset, cols)); }
    vec_view_it_t<default_block_tag>      subvector(size_t coloffset, size_t cols)                         const { return vec_view_it_t<default_block_tag>     (ptr.subvectorit(coloffset, cols)); }

    // automatic conversion
    operator const cvec_view_t   <this_block_tag>&() const { return *reinterpret_cast<const cvec_view_t   <this_block_tag>*>(this); }
    operator const  vec_view_t   <this_block_tag>&() const { return *reinterpret_cast<const  vec_view_t   <this_block_tag>*>(this); }
    operator       cvec_view_it_t<this_block_tag>&()       { return *reinterpret_cast<      cvec_view_it_t<this_block_tag>*>(this); }
    operator const cvec_view_it_t<this_block_tag>&() const { return *reinterpret_cast<const cvec_view_it_t<this_block_tag>*>(this); }

    operator const cvec_view_t   <cond_default_block_tag>&() const { return *reinterpret_cast< const cvec_view_t   <cond_default_block_tag>* > (this); }
    operator const cvec_view_it_t<cond_default_block_tag>&() const { return *reinterpret_cast< const cvec_view_it_t<cond_default_block_tag>* > (this); }
    operator const  vec_view_t   <cond_default_block_tag>&() const { return *reinterpret_cast< const  vec_view_t   <cond_default_block_tag>* > (this); }
    operator const  vec_view_it_t<cond_default_block_tag>&() const { return *reinterpret_cast< const  vec_view_it_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
          vec_view_it_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<      vec_view_it_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const vec_view_it_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const vec_view_it_t<block_tag<bits,masked>>*>(this); }

          vec_view_it_t& operator*()       { return *this; }
    const vec_view_it_t& operator*() const { return *this; }
          vec_view_it_t* operator->()       { return this; }
    const vec_view_it_t* operator->() const { return this; }

    // common vector API class members
    CONST_VECTOR_ITERATOR_CLASS_MEMBERS(vec_view_it_t)
    CONST_VECTOR_CLASS_MEMBERS
    VECTOR_CLASS_MEMBERS(vec_view_it_t,const)
    VECTOR_CLASS_MEMBERS(vec_view_it_t,)

    // vector result
    template<typename F> const vec_view_it_t& operator=(vector_result<F>&& vr) const { vr.r(ptr, this_block_tag()); return *this; }
    template<typename F>       vec_view_it_t& operator=(vector_result<F>&& vr)       { vr.r(ptr, this_block_tag()); return *this; }
};


template<typename _block_tag>
class cmat_view_t
    : public matrix_trait<_block_tag>
{
public:
    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    VECTOR_BLOCK_TAG_CLASS_MEMBERS(cmat_view_t)

    cm_ptr ptr;
    
    cmat_view_t(): ptr() {}
    cmat_view_t(const cm_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    cmat_view_t(const cmat_view_t& ) = default;
    cmat_view_t(      cmat_view_t&&) = default;
    // deleted for safety, to avoid confusion
    cmat_view_t& operator=(const cmat_view_t& ) = delete;
    cmat_view_t& operator=(      cmat_view_t&&) = delete;

    // view management
    void reset(const cm_ptr& p) { ptr = p; }
    void reset(const cmat_view_t& m) { ptr = m.ptr; }
    
    template<size_t bits, bool masked>
    cvec_view_it_t<block_tag<bits,masked>> subvector(size_t row, size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cvec_view_it_t<block_tag<bits,masked>>(ptr.subvectorit(row, coloffset, cols)); }
    cvec_view_it_t<default_block_tag>      subvector(size_t row, size_t coloffset, size_t cols)                         const { return cvec_view_it_t<default_block_tag>     (ptr.subvectorit(row, coloffset, cols)); }
    
    template<size_t bits, bool masked>
    cmat_view_t<block_tag<bits,masked>> submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cmat_view_t<block_tag<bits,masked>>(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
    cmat_view_t<default_block_tag>      submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols)                         const { return cmat_view_t<default_block_tag>     (ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    cvec_view_it_t<this_block_tag> operator[](size_t r) const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
    cvec_view_it_t<this_block_tag> operator()(size_t r) const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
    cvec_view_it_t<this_block_tag> begin()              const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(0)); }
    cvec_view_it_t<this_block_tag> end()                const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(rows())); }

    // automatic conversion
    operator const cmat_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const cmat_view_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
          cmat_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<      cmat_view_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const cmat_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const cmat_view_t<block_tag<bits,masked>>*>(this); }

    // common matrix API class members
    CONST_MATRIX_CLASS_MEMBERS
};

template<typename _block_tag>
class mat_view_t
    : public matrix_trait<_block_tag>
{
public:
    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    VECTOR_BLOCK_TAG_CLASS_MEMBERS(mat_view_t)

    m_ptr ptr;
    
    mat_view_t(): ptr() {}
    mat_view_t(const m_ptr& p): ptr(p) {}
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    mat_view_t(const mat_view_t& ) = default;
    mat_view_t(      mat_view_t&&) = default;
    mat_view_t& operator=(const mat_view_t& ) = default;
    mat_view_t& operator=(      mat_view_t&&) = default;

    // view management
    void reset(const m_ptr& p) { ptr = p; }
    void reset(const mat_view_t& m) { ptr = m.ptr; }

    template<size_t bits, bool masked>
    vec_view_it_t<block_tag<bits,masked>> subvector(size_t row, size_t coloffset, size_t cols, block_tag<bits,masked>) const { return vec_view_it_t<block_tag<bits,masked>>(ptr.subvectorit(row, coloffset, cols)); }
    vec_view_it_t<default_block_tag>      subvector(size_t row, size_t coloffset, size_t cols)                         const { return vec_view_it_t<default_block_tag>     (ptr.subvectorit(row, coloffset, cols)); }

    template<size_t bits, bool masked>
    mat_view_t<block_tag<bits,masked>> submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols, block_tag<bits,masked>) const { return mat_view_t<block_tag<bits,masked>>(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
    mat_view_t<default_block_tag>      submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols)                         const { return mat_view_t<default_block_tag>     (ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    vec_view_it_t<this_block_tag> operator[](size_t r) const { return vec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
    vec_view_it_t<this_block_tag> operator()(size_t r) const { return vec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
    vec_view_it_t<this_block_tag> begin()              const { return vec_view_it_t<this_block_tag>(ptr.subvectorit(0)); }
    vec_view_it_t<this_block_tag> end()                const { return vec_view_it_t<this_block_tag>(ptr.subvectorit(rows())); }

    // automatic conversion
    operator       cmat_view_t<this_block_tag>&()       { return *reinterpret_cast<      cmat_view_t<this_block_tag>*>(this); }
    operator const cmat_view_t<this_block_tag>&() const { return *reinterpret_cast<const cmat_view_t<this_block_tag>*>(this); }

    operator const  mat_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const  mat_view_t<cond_default_block_tag>* > (this); }
    operator const cmat_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const cmat_view_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
          mat_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<      mat_view_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const mat_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const mat_view_t<block_tag<bits,masked>>*>(this); }

    // common matrix API class members
    CONST_MATRIX_CLASS_MEMBERS
    MATRIX_CLASS_MEMBERS(mat_view_t,const)
    MATRIX_CLASS_MEMBERS(mat_view_t,)

    // matrix result
    template<typename F> const mat_view_t& operator=(matrix_result<F>&& mr) const { mr.r(ptr,this_block_tag()); return *this; }
    template<typename F>       mat_view_t& operator=(matrix_result<F>&& mr)       { mr.r(ptr,this_block_tag()); return *this; }
};


template<typename _block_tag>
class vec_t
    : public vector_trait< _block_tag >
{
public:
    v_ptr ptr;
private:
    std::vector<uint64_t> mem;
public:
    static const size_t bit_alignment = 512;
    static const size_t byte_alignment = bit_alignment/8;
    static const size_t word_alignment = bit_alignment/64;

    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    static const size_t block_bits = this_block_tag::bits;
    static const bool maskedlastblock = this_block_tag::maskedlastblock;
    static this_block_tag tag() { return this_block_tag(); }

    
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)>
    void assign(const vector_t& v)
    {
        resize(v.columns());
        copy(v);
    }
    
    void resize(size_t _columns, bool value = false)
    {
        if (_columns == columns())
            return;
        if (_columns == 0)
        {
            ptr = v_ptr();
            mem.clear();
            return;
        }
        if (mem.empty())
        {
            size_t rowwords = (_columns+63)/64;
            rowwords = (rowwords + word_alignment - 1) & ~(word_alignment-1);
            mem.resize(rowwords + word_alignment, value ? ~uint64_t(0) : uint64_t(0));
            auto p = (uintptr_t(&mem[0]) + byte_alignment-1) & ~uintptr_t(byte_alignment-1);
            ptr.reset(reinterpret_cast<uint64_t*>(p), _columns);
            return;
        }
        vec_t tmp(_columns, value);
        size_t mincols = std::min(_columns, columns());
        tmp.subvector(0, mincols).copy(subvector(0, mincols));
        tmp.swap(*this);
    }
    
    void swap(vec_t& v)
    {
        std::swap(mem, v.mem);
        std::swap(ptr, v.ptr);
    }

    vec_t(): ptr(), mem() {}
    vec_t(size_t _columns, bool value = false): ptr(), mem() { resize(_columns,value); }
    
    // copy/move constructors & assignment copy/move the view parameters, not the view contents
    vec_t(const vec_t&  v): ptr(), mem() { assign(v); }
    vec_t(      vec_t&& v): ptr(), mem() { swap(v); }
    
    vec_t& operator=(const vec_t&  v) { assign(v); return *this; }
    vec_t& operator=(      vec_t&& v) { swap(v); return *this; }

    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> vec_t(const vector_t& v): ptr(), mem() { assign(v); }
    template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> vec_t& operator=(const vector_t& v) { assign(v); return *this; }

    template<typename F> vec_t(vector_result<F>&& vr) { vr.r.resize_me(*this); vr.r(ptr, tag()); }

    // view management
    template<size_t bits, bool masked>
    cvec_view_t<block_tag<bits,masked>> subvector(size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cvec_view_t<block_tag<bits,masked>>(ptr.subvector(coloffset, cols)); }
    cvec_view_t<default_block_tag>      subvector(size_t coloffset, size_t cols)                         const { return cvec_view_t<default_block_tag>     (ptr.subvector(coloffset, cols)); }
    template<size_t bits, bool masked>
    vec_view_t<block_tag<bits,masked>> subvector(size_t coloffset, size_t cols, block_tag<bits,masked>)       { return  vec_view_t<block_tag<bits,masked>>(ptr.subvector(coloffset, cols)); }
    vec_view_t<default_block_tag>      subvector(size_t coloffset, size_t cols)                               { return  vec_view_t<default_block_tag>     (ptr.subvector(coloffset, cols)); }
    
    // automatic conversion
    operator const cvec_view_t<this_block_tag>&() const { return *reinterpret_cast<const cvec_view_t<this_block_tag>*>(this); }
    operator const  vec_view_t<this_block_tag>&()       { return *reinterpret_cast<const  vec_view_t<this_block_tag>*>(this); }

    operator const cvec_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const cvec_view_t<cond_default_block_tag>* > (this); }
    operator const  vec_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const  vec_view_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
           vec_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<       vec_view_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const cvec_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const cvec_view_t<block_tag<bits,masked>>*>(this); }

    // common matrix API class members
    CONST_VECTOR_CLASS_MEMBERS
    VECTOR_CLASS_MEMBERS(vec_t,)

    // vector result
    template<typename F> vec_t& operator=(vector_result<F>&& vr) { vr.r.resize_me(*this); vr.r(ptr, tag()); return *this; }
};

template<typename _block_tag>
class mat_t
    : public matrix_trait< _block_tag >
{
public:
    m_ptr ptr;
private:
    std::vector<uint64_t> mem;

public:
    static const size_t bit_alignment = 512;
    static const size_t byte_alignment = bit_alignment/8;
    static const size_t word_alignment = bit_alignment/64;

    typedef _block_tag this_block_tag;
    typedef uint64_block_t<this_block_tag::bits> this_block_t;
    typedef typename std::conditional< std::is_same<this_block_tag,default_block_tag>::value, void_block_tag, default_block_tag >::type cond_default_block_tag;

    static const size_t block_bits = this_block_tag::bits;
    static const bool maskedlastblock = this_block_tag::maskedlastblock;
    static this_block_tag tag() { return this_block_tag(); }

    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
    void assign(const matrix_t& m)
    {
        resize(m.rows(), m.columns());
        copy(m);
    }
    
    void resize(size_t _rows, size_t _columns, bool value = false)
    {
        if (_rows == rows() && _columns == columns())
            return;
        if (_rows == 0 && _columns == 0)
        {
            ptr = m_ptr();
            mem.clear();
            return;
        }
        if (mem.empty())
        {
            size_t rowwords = (_columns+63)/64;
            rowwords = (rowwords + word_alignment - 1) & ~(word_alignment-1);
            mem.resize(_rows * rowwords + word_alignment, value ? ~uint64_t(0) : uint64_t(0));
            auto p = (uintptr_t(&mem[0]) + byte_alignment-1) & ~uintptr_t(byte_alignment-1);
            ptr.reset(reinterpret_cast<uint64_t*>(p), _columns, rowwords, _rows);
            return;
        }
        mat_t tmp(_rows, _columns, value);
        size_t minrows = std::min(_rows, rows()), mincols = std::min(_columns, columns());
        tmp.submatrix(0, minrows, 0, mincols).copy(this->submatrix(0, minrows, 0, mincols));
        tmp.swap(*this);
    }
    
    void swap(mat_t& m)
    {
        std::swap(mem, m.mem);
        std::swap(ptr, m.ptr);
    }

    mat_t(): ptr(), mem() {}
    mat_t(const size_t rows, const size_t columns, bool value = false): ptr(), mem() { resize(rows, columns, value); }
    
    // copy/move constructors & assignment copy/move the contents
    mat_t(const mat_t&  m): ptr(), mem() { assign(m); }
    mat_t(      mat_t&& m): ptr(), mem() { swap(m); }
    mat_t& operator=(const mat_t&  m) { assign(m); return *this; }
    mat_t& operator=(      mat_t&& m) { assign(m); return *this; }

    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> mat_t(const matrix_t& m): ptr(), mem() { assign(m); }
    template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> mat_t& operator=(const matrix_t& m) { assign(m); return *this; }

    template<typename F> mat_t(matrix_result<F>&& mr) { mr.r.resize_me(*this); mr.r(ptr, tag()); }

    // view management
    template<size_t bits, bool masked>
    cvec_view_it_t<block_tag<bits,masked>> subvector(size_t row, size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cvec_view_it_t<block_tag<bits,masked>>(ptr.subvectorit(row, coloffset, cols)); }
    cvec_view_it_t<default_block_tag>      subvector(size_t row, size_t coloffset, size_t cols)                         const { return cvec_view_it_t<default_block_tag>     (ptr.subvectorit(row, coloffset, cols)); }
    template<size_t bits, bool masked>
     vec_view_it_t<block_tag<bits,masked>> subvector(size_t row, size_t coloffset, size_t cols, block_tag<bits,masked>)       { return  vec_view_it_t<block_tag<bits,masked>>(ptr.subvectorit(row, coloffset, cols)); }
     vec_view_it_t<default_block_tag>      subvector(size_t row, size_t coloffset, size_t cols)                               { return  vec_view_it_t<default_block_tag>     (ptr.subvectorit(row, coloffset, cols)); }
    
    template<size_t bits, bool masked>
    cmat_view_t<block_tag<bits,masked>> submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols, block_tag<bits,masked>) const { return cmat_view_t<block_tag<bits,masked>>(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
    cmat_view_t<default_block_tag>      submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols)                         const { return cmat_view_t<default_block_tag>     (ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
    template<size_t bits, bool masked>
     mat_view_t<block_tag<bits,masked>> submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols, block_tag<bits,masked>)       { return  mat_view_t<block_tag<bits,masked>>(ptr.submatrix(rowoffset, _rows, coloffset, cols)); }
     mat_view_t<default_block_tag>      submatrix(size_t rowoffset, size_t _rows, size_t coloffset, size_t cols)                               { return  mat_view_t<default_block_tag>     (ptr.submatrix(rowoffset, _rows, coloffset, cols)); }

    cvec_view_it_t<this_block_tag> operator[](size_t r) const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
     vec_view_it_t<this_block_tag> operator[](size_t r)       { return  vec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
    cvec_view_it_t<this_block_tag> operator()(size_t r) const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }
     vec_view_it_t<this_block_tag> operator()(size_t r)       { return  vec_view_it_t<this_block_tag>(ptr.subvectorit(r)); }

    cvec_view_it_t<this_block_tag> begin() const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(0)); }
    cvec_view_it_t<this_block_tag> end()   const { return cvec_view_it_t<this_block_tag>(ptr.subvectorit(rows())); }
     vec_view_it_t<this_block_tag> begin()       { return vec_view_it_t<this_block_tag>(ptr.subvectorit(0)); }
     vec_view_it_t<this_block_tag> end()         { return vec_view_it_t<this_block_tag>(ptr.subvectorit(rows())); }

    // automatic conversion
    operator const cmat_view_t<this_block_tag>&() const { return *reinterpret_cast<const cmat_view_t<this_block_tag>*>(this); }
    operator const  mat_view_t<this_block_tag>&()       { return *reinterpret_cast<const  mat_view_t<this_block_tag>*>(this); }

    operator const cmat_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const cmat_view_t<cond_default_block_tag>* > (this); }
    operator const  mat_view_t<cond_default_block_tag>& () const { return *reinterpret_cast< const  mat_view_t<cond_default_block_tag>* > (this); }

    template<size_t bits = 64, bool masked = true>
           mat_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>)       { return *reinterpret_cast<       mat_view_t<block_tag<bits,masked>>*>(this); }
    template<size_t bits = 64, bool masked = true>
    const cmat_view_t<block_tag<bits,masked>>& as(block_tag<bits,masked>) const { return *reinterpret_cast<const cmat_view_t<block_tag<bits,masked>>*>(this); }

    // common matrix API class members
    CONST_MATRIX_CLASS_MEMBERS
    MATRIX_CLASS_MEMBERS(mat_t,)

    // vector result
    template<typename F> mat_t& operator=(matrix_result<F>&& mr) { mr.r.resize_me(*this); mr.r(ptr, tag()); return *this; }
};

template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)>
inline bool operator==(const vector1_t& v1, const vector2_t& v2) { return v1.ptr == v2.ptr; }
template<typename vector1_t, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector1_t), MCCL_ENABLE_IF_VECTOR(vector2_t)>
inline bool operator!=(const vector1_t& v1, const vector2_t& v2) { return v1.ptr != v2.ptr; }

template<typename tag1, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector2_t)>
inline bool operator==(const vec_t<tag1>& v1, const vector2_t& v2) { return v1.isequal(v2); }
template<typename tag1, typename vector2_t, MCCL_ENABLE_IF_VECTOR(vector2_t)>
inline bool operator!=(const vec_t<tag1>& v1, const vector2_t& v2) { return !v1.isequal(v2); }
template<typename vector1_t, typename tag2, MCCL_ENABLE_IF_VECTOR(vector1_t)>
inline bool operator==(const vector1_t& v1, const vec_t<tag2>& v2) { return v1.isequal(v2); }
template<typename vector1_t, typename tag2, MCCL_ENABLE_IF_VECTOR(vector1_t)>
inline bool operator!=(const vector1_t& v1, const vec_t<tag2>& v2) { return !v1.isequal(v2); }

template<typename matrix1_t, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix1_t), MCCL_ENABLE_IF_MATRIX(matrix2_t)>
inline bool operator==(const matrix1_t& m1, const matrix2_t& m2) { return m1.ptr == m2.ptr; }
template<typename matrix1_t, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix1_t), MCCL_ENABLE_IF_MATRIX(matrix2_t)>
inline bool operator!=(const matrix1_t& m1, const matrix2_t& m2) { return m1.ptr != m2.ptr; }

template<typename tag1, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix2_t)>
inline bool operator==(const mat_t<tag1>& m1, const matrix2_t& m2) { return m1.isequal(m2); }
template<typename tag1, typename matrix2_t, MCCL_ENABLE_IF_MATRIX(matrix2_t)>
inline bool operator!=(const mat_t<tag1>& m1, const matrix2_t& m2) { return !m1.isequal(m2); }
template<typename matrix1_t, typename tag2, MCCL_ENABLE_IF_MATRIX(matrix1_t)>
inline bool operator==(const matrix1_t& m1, const mat_t<tag2>& m2) { return m1.isequal(m2); }
template<typename matrix1_t, typename tag2, MCCL_ENABLE_IF_MATRIX(matrix1_t)>
inline bool operator!=(const matrix1_t& m1, const mat_t<tag2>& m2) { return !m1.isequal(m2); }

template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)>
inline std::ostream& operator<<(std::ostream& o, const vector_t& v) { detail::v_print(o, v.ptr); return o; }
template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
inline std::ostream& operator<<(std::ostream& o, const matrix_t& m) { detail::m_print(o, m.ptr); return o; }

#define MCCL_VECTOR_RESULT_FUNCTION_OP2(func) \
template<typename _block_tag2> \
struct v_ptr_op2_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    const cv_ptr* v2; \
    v_ptr_op2_result_ ## func (): v2(nullptr) {} \
    v_ptr_op2_result_ ## func (const cv_ptr& _v2) { v2 = &_v2; } \
    v_ptr_op2_result_ ## func (const v_ptr_op2_result_ ## func &) = default; \
    v_ptr_op2_result_ ## func (      v_ptr_op2_result_ ## func &&) = default; \
    v_ptr_op2_result_ ## func & operator=(const v_ptr_op2_result_ ## func &) = default; \
    v_ptr_op2_result_ ## func & operator=(      v_ptr_op2_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const v_ptr& v1, block_tag<bits1,masked1>) { detail::  func (v1,*v2, block_tag<bits1,masked1>(), block_tag2()); } \
    void resize_me(vec& v) { v.resize(v2->columns); } \
}; \
template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)> \
inline vector_result<v_ptr_op2_result_ ## func <typename vector_t::this_block_tag> > func (const vector_t& v2) \
{ \
    return vector_result<v_ptr_op2_result_ ## func <typename vector_t::this_block_tag> >(v2.ptr); \
}

MCCL_VECTOR_RESULT_FUNCTION_OP2(v_copy)
MCCL_VECTOR_RESULT_FUNCTION_OP2(v_copynot)

#define MCCL_VECTOR_RESULT_FUNCTION_OP3(func) \
template<typename _block_tag2, typename _block_tag3> \
struct v_ptr_op3_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    typedef _block_tag3 block_tag3; \
    const cv_ptr* v2; \
    const cv_ptr* v3; \
    v_ptr_op3_result_ ## func (): v2(nullptr), v3(nullptr) {} \
    v_ptr_op3_result_ ## func (const cv_ptr& _v2, const cv_ptr& _v3) { v2 = &_v2; v3 = &_v3; } \
    v_ptr_op3_result_ ## func (const v_ptr_op3_result_ ## func &) = default; \
    v_ptr_op3_result_ ## func (      v_ptr_op3_result_ ## func &&) = default; \
    v_ptr_op3_result_ ## func & operator=(const v_ptr_op3_result_ ## func &) = default; \
    v_ptr_op3_result_ ## func & operator=(      v_ptr_op3_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const v_ptr& v1, block_tag<bits1,masked1>) { detail::  func (v1,*v2,*v3, block_tag<bits1,masked1>(), block_tag2(), block_tag3()); } \
    void resize_me(vec& v) { v.resize(v2->columns); } \
}; \
template<typename vector_t2, typename vector_t3, MCCL_ENABLE_IF_VECTOR(vector_t2), MCCL_ENABLE_IF_VECTOR(vector_t3)> \
inline vector_result<v_ptr_op3_result_ ## func <typename vector_t2::this_block_tag, typename vector_t3::this_block_tag>> func (const vector_t2& v2, const vector_t3& v3) \
{ \
    return vector_result<v_ptr_op3_result_ ## func <typename vector_t2::this_block_tag, typename vector_t3::this_block_tag> >(v2.ptr, v3.ptr); \
}

MCCL_VECTOR_RESULT_FUNCTION_OP3(v_and)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_or)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_xor)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_nand)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_nor)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_nxor)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_andin)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_andni)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_orin)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_orni)


#define MCCL_MATRIX_RESULT_FUNCTION_OP2(func) \
template<typename _block_tag2> \
struct m_ptr_op2_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    const cm_ptr* m2; \
    m_ptr_op2_result_ ## func (): m2(nullptr) {} \
    m_ptr_op2_result_ ## func (const cm_ptr& _m2) { m2 = &_m2; } \
    m_ptr_op2_result_ ## func (const m_ptr_op2_result_ ## func &) = default; \
    m_ptr_op2_result_ ## func (      m_ptr_op2_result_ ## func &&) = default; \
    m_ptr_op2_result_ ## func & operator=(const m_ptr_op2_result_ ## func &) = default; \
    m_ptr_op2_result_ ## func & operator=(      m_ptr_op2_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const m_ptr& m1, block_tag<bits1,masked1>) { detail:: func (m1,*m2, block_tag<bits1,masked1>(), block_tag2()); } \
    void resize_me(mat& m) { m.resize(m2->rows, m2->columns); } \
}; \
template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)> \
inline matrix_result<m_ptr_op2_result_ ## func <typename matrix_t::this_block_tag>> func (const matrix_t& m2) \
{ \
    return matrix_result<m_ptr_op2_result_ ## func <typename matrix_t::this_block_tag> >(m2.ptr); \
}

MCCL_MATRIX_RESULT_FUNCTION_OP2(m_copy)
MCCL_MATRIX_RESULT_FUNCTION_OP2(m_copynot)

struct m_ptr_op2_result_m_transpose
{
    const cm_ptr* m2;
    m_ptr_op2_result_m_transpose(): m2(nullptr) {}
    m_ptr_op2_result_m_transpose(const cm_ptr& _m2) { m2 = &_m2; }
    m_ptr_op2_result_m_transpose(const m_ptr_op2_result_m_transpose&) = default;
    m_ptr_op2_result_m_transpose(      m_ptr_op2_result_m_transpose&&) = default;
    m_ptr_op2_result_m_transpose& operator=(const m_ptr_op2_result_m_transpose&) = default;
    m_ptr_op2_result_m_transpose& operator=(      m_ptr_op2_result_m_transpose&&) = default;
    template<size_t bits1, bool masked1> void operator()(const m_ptr& m1, block_tag<bits1,masked1>) { detail::m_transpose(m1,*m2); }
    void resize_me(mat& m) { m.resize(m2->columns, m2->rows); }
};
template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
inline matrix_result<m_ptr_op2_result_m_transpose> m_transpose(const matrix_t& m2)
{
    return matrix_result<m_ptr_op2_result_m_transpose>(m2.ptr);
}

#define MCCL_MATRIX_RESULT_FUNCTION_OP3(func) \
template<typename _block_tag2, typename _block_tag3> \
struct m_ptr_op3_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    typedef _block_tag3 block_tag3; \
    const cm_ptr* m2; \
    const cm_ptr* m3; \
    m_ptr_op3_result_ ## func (): m2(nullptr), m3(nullptr) {} \
    m_ptr_op3_result_ ## func (const cm_ptr& _m2, const cm_ptr& _m3) { m2 = &_m2; m3 = &_m3; } \
    m_ptr_op3_result_ ## func (const m_ptr_op3_result_ ## func &) = default; \
    m_ptr_op3_result_ ## func (      m_ptr_op3_result_ ## func &&) = default; \
    m_ptr_op3_result_ ## func & operator=(const m_ptr_op3_result_ ## func &) = default; \
    m_ptr_op3_result_ ## func & operator=(      m_ptr_op3_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const m_ptr& m1, block_tag<bits1,masked1>) { detail:: func (m1,*m2,*m3, block_tag<bits1,masked1>(), block_tag2(), block_tag3()); } \
    void resize_me(mat& m) { m.resize(m2->rows, m2->columns); } \
}; \
template<typename matrix_t2, typename matrix_t3, MCCL_ENABLE_IF_MATRIX(matrix_t2), MCCL_ENABLE_IF_MATRIX(matrix_t3)> \
inline matrix_result<m_ptr_op3_result_ ## func <typename matrix_t2::this_block_tag, typename matrix_t3::this_block_tag>> func (const matrix_t2& m2, const matrix_t3& m3) \
{ \
    return matrix_result<m_ptr_op3_result_ ## func <typename matrix_t2::this_block_tag, typename matrix_t3::this_block_tag> >(m2.ptr, m3.ptr); \
}

MCCL_MATRIX_RESULT_FUNCTION_OP3(m_and)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_or)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_xor)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_nand)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_nor)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_nxor)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_andin)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_andni)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_orin)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_orni)


template<typename vector_t2, typename vector_t3, MCCL_ENABLE_IF_VECTOR(vector_t2), MCCL_ENABLE_IF_VECTOR(vector_t3)>
inline auto operator & (const vector_t2& v2, const vector_t3& v3) -> decltype(v_and(v2,v3)) { return v_and(v2, v3); }
template<typename vector_t2, typename vector_t3, MCCL_ENABLE_IF_VECTOR(vector_t2), MCCL_ENABLE_IF_VECTOR(vector_t3)>
inline auto operator | (const vector_t2& v2, const vector_t3& v3) -> decltype(v_or (v2,v3)) { return v_or (v2, v3); }
template<typename vector_t2, typename vector_t3, MCCL_ENABLE_IF_VECTOR(vector_t2), MCCL_ENABLE_IF_VECTOR(vector_t3)>
inline auto operator ^ (const vector_t2& v2, const vector_t3& v3) -> decltype(v_xor(v2,v3)) { return v_xor(v2, v3); }

template<typename matrix_t2, typename matrix_t3, MCCL_ENABLE_IF_MATRIX(matrix_t2), MCCL_ENABLE_IF_MATRIX(matrix_t3)>
inline auto operator & (const matrix_t2& m2, const matrix_t3& m3) -> decltype(m_and(m2,m3)) { return m_and(m2, m3); }
template<typename matrix_t2, typename matrix_t3, MCCL_ENABLE_IF_MATRIX(matrix_t2), MCCL_ENABLE_IF_MATRIX(matrix_t3)>
inline auto operator | (const matrix_t2& m2, const matrix_t3& m3) -> decltype(m_or (m2,m3)) { return m_or (m2, m3); }
template<typename matrix_t2, typename matrix_t3, MCCL_ENABLE_IF_MATRIX(matrix_t2), MCCL_ENABLE_IF_MATRIX(matrix_t3)>
inline auto operator ^ (const matrix_t2& m2, const matrix_t3& m3) -> decltype(m_xor(m2,m3)) { return m_xor(m2, m3); }

template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
inline size_t hammingweight(const matrix_t& m) { return m.hw(); }

template<typename vector_t, MCCL_ENABLE_IF_VECTOR(vector_t)>
inline size_t hammingweight(const vector_t& v) { return v.hw(); }

template<typename vector_t1, typename vector_t2, MCCL_ENABLE_IF_VECTOR(vector_t1), MCCL_ENABLE_IF_VECTOR(vector_t2)>
inline size_t hammingweight_and(const vector_t1& v1, const vector_t2& v2) { return detail::v_hw_and(v1.ptr, v2.ptr); }
template<typename vector_t1, typename vector_t2, MCCL_ENABLE_IF_VECTOR(vector_t1), MCCL_ENABLE_IF_VECTOR(vector_t2)>
inline size_t hammingweight_or (const vector_t1& v1, const vector_t2& v2) { return detail::v_hw_or (v1.ptr, v2.ptr); }
template<typename vector_t1, typename vector_t2, MCCL_ENABLE_IF_VECTOR(vector_t1), MCCL_ENABLE_IF_VECTOR(vector_t2)>
inline size_t hammingweight_xor(const vector_t1& v1, const vector_t2& v2) { return detail::v_hw_xor(v1.ptr, v2.ptr); }


MCCL_END_NAMESPACE

#endif
