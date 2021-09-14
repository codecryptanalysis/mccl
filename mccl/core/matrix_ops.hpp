#ifndef MCCL_CORE_MATRIX_OPS_HPP
#define MCCL_CORE_MATRIX_OPS_HPP

#include <mccl/config/config.hpp>

#include <mccl/core/matrix_base.hpp>

#include <iostream>
#include <array>

MCCL_BEGIN_NAMESPACE

namespace detail {

inline       uint64_t& m_getword(const  m_ptr& m, size_t r, size_t c) { return m.ptr[m.stride*r + (c/64)]; }
inline const uint64_t& m_getword(const cm_ptr& m, size_t r, size_t c) { return m.ptr[m.stride*r + (c/64)]; }
inline       uint64_t& v_getword(const  v_ptr& v, size_t c) { return v.ptr[(c/64)]; }
inline const uint64_t& v_getword(const cv_ptr& v, size_t c) { return v.ptr[(c/64)]; }

inline bool m_getbit(const cm_ptr& m, size_t r, size_t c) { return (m_getword(m,r,c) >> (c%64)) & 1; }
inline void m_clearbit(const m_ptr& m, size_t r, size_t c) { m_getword(m,r,c) &= ~(uint64_t(1) << (c%64)); }
inline void m_flipbit(const m_ptr& m, size_t r, size_t c) { m_getword(m,r,c) ^= uint64_t(1) << (c%64); }
inline void m_setbit(const m_ptr& m, size_t r, size_t c) { m_getword(m,r,c) |= uint64_t(1) << (c%64); }
inline void m_setbit(const m_ptr& m, size_t r, size_t c, bool b) { uint64_t x = uint64_t(1)<<(c%64); uint64_t& w = m_getword(m,r,c); w &= ~x; w |= b?x:0;}

inline bool v_getbit(const cv_ptr& v, size_t c) { return (v_getword(v,c) >> (c%64)) & 1; }
inline void v_clearbit(const v_ptr& m, size_t c) { v_getword(m,c) &= ~(uint64_t(1) << (c%64)); }
inline void v_flipbit(const v_ptr& m, size_t c) { v_getword(m,c) ^= uint64_t(1) << (c%64); }
inline void v_setbit(const v_ptr& m, size_t c) { v_getword(m,c) |= uint64_t(1) << (c%64); }
inline void v_setbit(const v_ptr& m, size_t c, bool b) { uint64_t x = uint64_t(1)<<(c%64); uint64_t& w = v_getword(m,c); w &= ~x; w |= b?x:0;}

// matrix_ops.inl
inline void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols, bool b);
inline void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols);
inline void v_clearcolumns(const v_ptr& v, size_t coloffset, size_t cols);
inline void v_flipcolumns(const v_ptr& v, size_t coloffset, size_t cols);

inline size_t v_hw(const cv_ptr& v);
inline size_t v_hw_and(const cv_ptr& v1, const cv_ptr& v2);
inline size_t v_hw_or(const cv_ptr& v1, const cv_ptr& v2);
inline size_t v_hw_xor(const cv_ptr& v1, const cv_ptr& v2);

template<size_t bits, bool masked> inline void v_set(const v_ptr& v, block_tag<bits,masked>);
template<size_t bits, bool masked> inline void v_clear(const v_ptr& v, block_tag<bits,masked>);
template<size_t bits, bool masked> inline void v_set(const v_ptr& v, bool b, block_tag<bits,masked>);
template<size_t bits, bool masked> inline void v_not(const v_ptr& v, block_tag<bits,masked>);

template<size_t bits, bool masked>
inline bool v_isequal(const cv_ptr& v1, const cv_ptr& v2, block_tag<bits,masked>);
template<size_t bits1, bool masked1, size_t bits2, bool masked2>
inline bool v_isequal(const cv_ptr& v1, const cv_ptr& v2, block_tag<bits1,masked1>, block_tag<bits2,masked2>)
{
    return v_isequal(v1, v2, block_tag<(bits1<bits2?bits1:bits2),true>()); // always use maskedlastword
}

template<size_t bits, bool masked>
inline void v_swap(const v_ptr& v1, const v_ptr& v2, block_tag<bits,masked>);
template<size_t bits1, bool masked1, size_t bits2, bool masked2>
inline void v_swap(const v_ptr& v1, const v_ptr& v2, block_tag<bits1,masked1>, block_tag<bits2,masked2>)
{
    v_swap(v1, v2, block_tag<(bits1<bits2?bits1:bits2),(masked1|masked2)>()); // use maskedlastword if either does
}


#define MCCL_VECTOR_2OP_FUNC_DEF(func) \
   template<size_t bits, bool masked> \
   inline void func (const v_ptr&, const cv_ptr&, block_tag<bits,masked>); \
   template<size_t bits1, bool masked1, size_t bits2, bool masked2> \
   inline void func (const v_ptr& dst, const cv_ptr& v2, block_tag<bits1,masked1>, block_tag<bits2,masked2>) \
   { \
      func (dst, v2, block_tag<(bits1<bits2?bits1:bits2),masked1>()); \
   }

MCCL_VECTOR_2OP_FUNC_DEF(v_copy)
MCCL_VECTOR_2OP_FUNC_DEF(v_copynot)
MCCL_VECTOR_2OP_FUNC_DEF(v_and)
MCCL_VECTOR_2OP_FUNC_DEF(v_or)
MCCL_VECTOR_2OP_FUNC_DEF(v_xor)
MCCL_VECTOR_2OP_FUNC_DEF(v_nand)
MCCL_VECTOR_2OP_FUNC_DEF(v_nor)
MCCL_VECTOR_2OP_FUNC_DEF(v_nxor)
MCCL_VECTOR_2OP_FUNC_DEF(v_andin)
MCCL_VECTOR_2OP_FUNC_DEF(v_andni)
MCCL_VECTOR_2OP_FUNC_DEF(v_orin)
MCCL_VECTOR_2OP_FUNC_DEF(v_orni)

#define MCCL_VECTOR_3OP_FUNC_DEF(func) \
   template<size_t bits, bool masked> \
   inline void func (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2, block_tag<bits,masked>); \
   template<size_t bits1, bool masked1, size_t bits2, bool masked2, size_t bits3, bool masked3> \
   inline void func (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2, block_tag<bits1,masked1>, block_tag<bits2,masked2>, block_tag<bits3,masked3>) \
   { \
      const size_t bits = (bits1 < bits2) ? (bits3 < bits1 ? bits3 : bits1) : (bits3 < bits2 ? bits3 : bits2); \
      func (dst, v1, v2, block_tag<bits,masked1>() ); \
   }

MCCL_VECTOR_3OP_FUNC_DEF(v_and)
MCCL_VECTOR_3OP_FUNC_DEF(v_or)
MCCL_VECTOR_3OP_FUNC_DEF(v_xor)
MCCL_VECTOR_3OP_FUNC_DEF(v_nand)
MCCL_VECTOR_3OP_FUNC_DEF(v_nor)
MCCL_VECTOR_3OP_FUNC_DEF(v_nxor)
MCCL_VECTOR_3OP_FUNC_DEF(v_andin)
MCCL_VECTOR_3OP_FUNC_DEF(v_andni)
MCCL_VECTOR_3OP_FUNC_DEF(v_orin)
MCCL_VECTOR_3OP_FUNC_DEF(v_orni)


// matrix_ops.cpp
void v_print(std::ostream& o, const cv_ptr& v);
void m_print(std::ostream& o, const cm_ptr& m, bool transpose = false);

size_t m_hw(const cm_ptr& m);

void m_swapcolumns(const m_ptr& m, size_t c1, size_t c2);
void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols, bool b);
void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols);
void m_clearcolumns(const m_ptr& m, size_t coloffset, size_t cols);
void m_flipcolumns(const m_ptr& m, size_t coloffset, size_t cols);

void m_transpose(const m_ptr& dst, const cm_ptr& src);

template<size_t bits, bool masked> void m_set(const m_ptr& m, block_tag<bits,masked>);
template<size_t bits, bool masked> void m_clear(const m_ptr& m, block_tag<bits,masked>);
template<size_t bits, bool masked> void m_set(const m_ptr& m, bool b, block_tag<bits,masked>) { if (b) m_set(m, block_tag<bits,masked>()); else m_clear(m, block_tag<bits,masked>()); }
template<size_t bits, bool masked> void m_not(const m_ptr& m, block_tag<bits,masked>);

template<size_t bits, bool masked>
bool m_isequal(const cm_ptr& m1, const cm_ptr& m2, block_tag<bits,masked>);

template<size_t bits1, bool masked1, size_t bits2, bool masked2>
inline bool m_isequal(const cm_ptr& m1, const cm_ptr& m2, block_tag<bits1,masked1>, block_tag<bits2,masked2>)
{
    return m_isequal(m1, m2, block_tag<(bits1<bits2?bits1:bits2),true>());
}

#define MCCL_MATRIX_2OP_FUNC_DEF(func) \
   template<size_t bits, bool masked> \
   void func (const m_ptr&, const cm_ptr&, block_tag<bits,masked>); \
   template<size_t bits1, bool masked1, size_t bits2, bool masked2> \
   inline void func (const m_ptr& dst, const cm_ptr& m2, block_tag<bits1,masked1>, block_tag<bits2,masked2>) \
   { \
      func (dst, m2, block_tag<(bits1<bits2?bits1:bits2),masked1>()); \
   }

MCCL_MATRIX_2OP_FUNC_DEF(m_copy)
MCCL_MATRIX_2OP_FUNC_DEF(m_copynot)
MCCL_MATRIX_2OP_FUNC_DEF(m_and)
MCCL_MATRIX_2OP_FUNC_DEF(m_or)
MCCL_MATRIX_2OP_FUNC_DEF(m_xor)
MCCL_MATRIX_2OP_FUNC_DEF(m_nand)
MCCL_MATRIX_2OP_FUNC_DEF(m_nor)
MCCL_MATRIX_2OP_FUNC_DEF(m_nxor)
MCCL_MATRIX_2OP_FUNC_DEF(m_andin)
MCCL_MATRIX_2OP_FUNC_DEF(m_andni)
MCCL_MATRIX_2OP_FUNC_DEF(m_orin)
MCCL_MATRIX_2OP_FUNC_DEF(m_orni)

#define MCCL_MATRIX_3OP_FUNC_DEF(func) \
   template<size_t bits, bool masked> \
   void func (const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2, block_tag<bits,masked>); \
   template<size_t bits1, bool masked1, size_t bits2, bool masked2, size_t bits3, bool masked3> \
   inline void func (const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2, block_tag<bits1,masked1>, block_tag<bits2,masked2>, block_tag<bits3,masked3>) \
   { \
      const size_t bits = (bits1 < bits2) ? (bits3 < bits1 ? bits3 : bits1) : (bits3 < bits2 ? bits3 : bits2); \
      func (dst, m1, m2, block_tag<bits,masked1>() ); \
   }

MCCL_MATRIX_3OP_FUNC_DEF(m_and)
MCCL_MATRIX_3OP_FUNC_DEF(m_or)
MCCL_MATRIX_3OP_FUNC_DEF(m_xor)
MCCL_MATRIX_3OP_FUNC_DEF(m_nand)
MCCL_MATRIX_3OP_FUNC_DEF(m_nor)
MCCL_MATRIX_3OP_FUNC_DEF(m_nxor)
MCCL_MATRIX_3OP_FUNC_DEF(m_andin)
MCCL_MATRIX_3OP_FUNC_DEF(m_andni)
MCCL_MATRIX_3OP_FUNC_DEF(m_orin)
MCCL_MATRIX_3OP_FUNC_DEF(m_orni)

}

MCCL_END_NAMESPACE

#include "matrix_ops.inl"

#endif

