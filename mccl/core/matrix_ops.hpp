#ifndef MCCL_CORE_MATRIX_OPS_HPP
#define MCCL_CORE_MATRIX_OPS_HPP

#include <mccl/config/config.hpp>

#include <mccl/core/matrix_base.hpp>

#include <iostream>
#include <array>

MCCL_BEGIN_NAMESPACE

namespace detail {

template<size_t bits>
struct alignas(bits/8) block_t {
    static const size_t size = bits/64;
    std::array<uint64_t,size> v;
    block_t& operator&=(const block_t& v2) { for (size_t i = 0; i < size; ++i) v[i] &= v2.v[i]; return *this; }
    block_t& operator^=(const block_t& v2) { for (size_t i = 0; i < size; ++i) v[i] ^= v2.v[i]; return *this; }
    block_t& operator|=(const block_t& v2) { for (size_t i = 0; i < size; ++i) v[i] |= v2.v[i]; return *this; }
    block_t operator&(const block_t& v2) const { block_t tmp(*this); return tmp &= v2; }
    block_t operator^(const block_t& v2) const { block_t tmp(*this); return tmp ^= v2; } 
    block_t operator|(const block_t& v2) const { block_t tmp(*this); return tmp |= v2; }
    block_t operator~() const { block_t tmp; for (size_t i = 0; i < size; ++i) tmp.v[i] = ~v[i]; return tmp; }
    bool operator==(const block_t& v2) const { for (size_t i = 0; i < size; ++i) if (v[i] != v2.v[i]) return false; return true; }
};
typedef block_t<256> block256_t;
typedef block_t<512> block512_t;

template<size_t bits>
struct aligned_tag {};

typedef aligned_tag<256> aligned256_tag; // for avx2
typedef aligned_tag<512> aligned512_tag; // cacheline, 2x avx2
typedef aligned_tag<64>  fullword_tag;


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
static inline bool v_isequal(const cv_ptr& v1, const cv_ptr& v2);
static inline void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols, bool b);
static inline void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols);
static inline void v_clearcolumns(const v_ptr& v, size_t coloffset, size_t cols);
static inline void v_flipcolumns(const v_ptr& v, size_t coloffset, size_t cols);
static inline void v_set(const v_ptr& v);
static inline void v_clear(const v_ptr& v);
static inline void v_set(const v_ptr& v, bool b);
static inline void v_not(const v_ptr& v);

static inline void v_copy   (const v_ptr& dst, const cv_ptr& src);
static inline void v_copynot(const v_ptr& dst, const cv_ptr& src);
static inline void v_and    (const v_ptr& dst, const cv_ptr& v2);
static inline void v_or     (const v_ptr& dst, const cv_ptr& v2);
static inline void v_xor    (const v_ptr& dst, const cv_ptr& v2);
static inline void v_nand   (const v_ptr& dst, const cv_ptr& v2);
static inline void v_nor    (const v_ptr& dst, const cv_ptr& v2);
static inline void v_nxor   (const v_ptr& dst, const cv_ptr& v2);
static inline void v_andin  (const v_ptr& dst, const cv_ptr& v2);
static inline void v_andni  (const v_ptr& dst, const cv_ptr& v2);
static inline void v_orin   (const v_ptr& dst, const cv_ptr& v2);
static inline void v_orni   (const v_ptr& dst, const cv_ptr& v2);
static inline void v_and(const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_or (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_xor(const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_nand(const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_nor (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_nxor(const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_andin(const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_andni(const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_orin (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);
static inline void v_orni (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2);

static inline void v_swap(const v_ptr& v1, const v_ptr& v2);
static inline size_t v_hw(const cv_ptr& v);
static inline size_t v_hw_and(const cv_ptr& v1, const cv_ptr& v2);
static inline size_t v_hw_or(const cv_ptr& v1, const cv_ptr& v2);
static inline size_t v_hw_xor(const cv_ptr& v1, const cv_ptr& v2);

// matrix_ops.cpp
void v_print(std::ostream& o, const cv_ptr& v);

void m_print(std::ostream& o, const cm_ptr& m, bool transpose = false);
bool m_isequal(const cm_ptr& m1, const cm_ptr& m2);
void m_swapcolumns(const m_ptr& m, size_t c1, size_t c2);
void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols, bool b);
void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols);
void m_clearcolumns(const m_ptr& m, size_t coloffset, size_t cols);
void m_flipcolumns(const m_ptr& m, size_t coloffset, size_t cols);
void m_set(const m_ptr& m);
void m_clear(const m_ptr& m);
void m_set(const m_ptr& m, bool b);
void m_not(const m_ptr& m);
void m_copy   (const m_ptr& dst, const cm_ptr& src);
void m_copynot(const m_ptr& dst, const cm_ptr& src);
void m_and    (const m_ptr& dst, const cm_ptr& m2);
void m_or     (const m_ptr& dst, const cm_ptr& m2);
void m_xor    (const m_ptr& dst, const cm_ptr& m2);
void m_nand   (const m_ptr& dst, const cm_ptr& m2);
void m_nor    (const m_ptr& dst, const cm_ptr& m2);
void m_nxor   (const m_ptr& dst, const cm_ptr& m2);
void m_andin  (const m_ptr& dst, const cm_ptr& m2);
void m_andni  (const m_ptr& dst, const cm_ptr& m2);
void m_orin   (const m_ptr& dst, const cm_ptr& m2);
void m_orni   (const m_ptr& dst, const cm_ptr& m2);
void m_and(const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_or (const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_xor(const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_nand(const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_nor (const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_nxor(const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_andin(const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_andni(const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_orin (const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);
void m_orni (const m_ptr& dst, const cm_ptr& m1, const cm_ptr& m2);

size_t m_hw(const cm_ptr& m);
void m_transpose(const m_ptr& dst, const cm_ptr& src);

}

using namespace detail;

MCCL_END_NAMESPACE

#include "matrix_ops.inl"

#endif

