#ifndef MCCL_CORE_MATRIX_M4RI_HPP
#define MCCL_CORE_MATRIX_M4RI_HPP

#include <mccl/core/matrix.hpp>
#include <stdlib.h>

MCCL_BEGIN_NAMESPACE

// m4ri uses its own data structures, this is a pointer to a m4ri handle
struct m4ri_data_t;
typedef m4ri_data_t* m4ri_handle_t;

/* create and free m4ri handles to mccl matrix_ref_t */
m4ri_handle_t create_m4ri_handle(const mat_view& m);
inline void free_m4ri_handle(m4ri_handle_t h) { free(h); }

/* m4ri functions taking m4ri handles */
// basic functions
void m4ri_transpose(m4ri_handle_t dst, const m4ri_handle_t src);
void m4ri_add(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B);
void m4ri_mul_naive(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B);
void m4ri_addmul_naive(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B);
void m4ri_gauss_delayed(m4ri_handle_t M, unsigned int startcol, bool full = false);
void m4ri_echelonize_naive(m4ri_handle_t M, bool full = false);
void m4ri_invert_naive(m4ri_handle_t dst, const m4ri_handle_t src, const m4ri_handle_t I);

// optimized complex operations
void m4ri_echelonize(m4ri_handle_t M, bool full =  false);
void m4ri_echelonize_pluq(m4ri_handle_t M, bool full = false);
void m4ri_echelonize_m4ri(m4ri_handle_t M, bool full = false, int k = 0);
void m4ri_mul(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B);
void m4ri_addmul(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B);

/* helper struct to manage m4ri handles */
struct m4ri_data_helper_t
{
    m4ri_data_helper_t(const mat_view& m): h(create_m4ri_handle(m)) {}
    ~m4ri_data_helper_t() { free_m4ri_handle(h); }
    
    m4ri_handle_t h;
    
    operator m4ri_handle_t&() { return h; }
    operator const m4ri_handle_t&() const { return h; }
};

/* m4ri functions taking mccl matrix_ref_t, which create temporary m4ri handles */
static inline void m4ri_transpose(const mat_view& dst, const mat_view& src)
{
    m4ri_data_helper_t h_dst(dst), h_src(src);
    m4ri_transpose(h_dst, h_src);
}
static inline void m4ri_add(const mat_view& dst, const mat_view& A, const mat_view& B)
{
    m4ri_data_helper_t h_dst(dst), h_A(A), h_B(B);
    m4ri_add(h_dst, h_A, h_B);
}
static inline void m4ri_mul_naive(const mat_view& dst, const mat_view& A, const mat_view& B)
{
    m4ri_data_helper_t h_dst(dst), h_A(A), h_B(B);
    m4ri_mul_naive(h_dst, h_A, h_B);
}
static inline void m4ri_addmul_naive(const mat_view& dst, const mat_view& A, const mat_view& B)
{
    m4ri_data_helper_t h_dst(dst), h_A(A), h_B(B);
    m4ri_addmul_naive(h_dst, h_A, h_B);
}
static inline void m4ri_gauss_delayed(const mat_view& M, unsigned int startcol = 0, bool full = false)
{
    m4ri_data_helper_t h(M);
    m4ri_gauss_delayed(M, startcol, full);
}
static inline void m4ri_echelonize_naive(const mat_view& M, bool full = false)
{
    m4ri_data_helper_t h(M);
    m4ri_echelonize_naive(M, full);
}
static inline void m4ri_invert_naive(const mat_view& dst, const mat_view& src, const mat_view& I)
{
    m4ri_data_helper_t h_dst(dst), h_src(src), h_I(I);
    m4ri_invert_naive(h_dst, h_src, h_I);
}

static inline void m4ri_echelonize(const mat_view& M, bool full = false)
{
    m4ri_data_helper_t h(M);
    m4ri_echelonize(h, full);
}
static inline void m4ri_echelonize_pluq(const mat_view& M, bool full = false)
{
    m4ri_data_helper_t h(M);
    m4ri_echelonize_pluq(h, full);
}
static inline void m4ri_echelonize_m4ri(const mat_view& M, bool full = false, int k = 0)
{
    m4ri_data_helper_t h(M);
    m4ri_echelonize_m4ri(h, full, k);
}
static inline void m4ri_mul(const mat_view& dst, const mat_view& A, const mat_view& B)
{
    m4ri_data_helper_t h_dst(dst), h_A(A), h_B(B);
    m4ri_mul(h_dst, h_A, h_B);
}
static inline void m4ri_addmul(const mat_view& dst, const mat_view& A, const mat_view& B)
{
    m4ri_data_helper_t h_dst(dst), h_A(A), h_B(B);
    m4ri_addmul(h_dst, h_A, h_B);
}

MCCL_END_NAMESPACE

#endif
