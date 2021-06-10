#include <mccl/core/matrix_m4ri.hpp>
#include <m4ri/m4ri.h>

MCCL_BEGIN_NAMESPACE

struct m4ri_data_t
{
    mzd_t mzd;
    mzd_block_t mzd_block;
    std::vector<uint64_t*> rows;
    
    // automatic conversions which allows m4ri_data_t to be directly used as input to m4ri functions
    operator mzd_t*()
    {
        return &mzd;
    }
    operator const mzd_t*() const
    {
        return &mzd;
    }
};
    
    void make_mzd(m4ri_data_t& dst, const m_ptr& src)
    {
        dst.mzd.nrows = src.rows;
        dst.mzd.ncols = src.columns;
        dst.mzd.width = (src.columns+63)/64;
        dst.mzd.rowstride = src.stride;
        dst.mzd.offset_vector = 0;
        dst.mzd.row_offset = 0;
        dst.mzd.flags = mzd_flag_nonzero_excess | mzd_flag_windowed_zerooffset;
        dst.mzd.flags |= (src.columns % 64 == 0) ? mzd_flag_windowed_zeroexcess : mzd_flag_nonzero_excess;
        dst.mzd.blockrows_log = 31 - __builtin_clz(src.rows);
        dst.mzd.high_bitmask = lastwordmask(src.columns);
        
        dst.mzd_block.size = src.rows * src.stride;
        dst.mzd_block.begin = src.data();
        dst.mzd_block.end = src.data(src.rows);
        
        dst.mzd.blocks = &dst.mzd_block;
        dst.rows.resize(src.rows);
        for (size_t i = 0; i < src.rows; ++i)
            dst.rows[i] = src.data(i);
        dst.mzd.rows = src.rows==0 ? nullptr : &dst.rows[0];
    }


m4ri_handle_t create_m4ri_handle(const mat_view& m)
{
    m4ri_data_t* h = new m4ri_data_t;
    make_mzd(*h, m.ptr);
    return h;
}

void m4ri_transpose(m4ri_handle_t dst, const m4ri_handle_t src)
{
    mzd_transpose(*dst, *src);
}
void m4ri_add(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B)
{
    mzd_add(*dst, *A, *B);
}
void m4ri_mul_naive(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B)
{
    mzd_mul_naive(*dst, *A, *B);
}
void m4ri_addmul_naive(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B)
{
    mzd_addmul_naive(*dst, *A, *B);
}
void m4ri_gauss_delayed(m4ri_handle_t M, unsigned int startcol, bool full)
{
    mzd_gauss_delayed(*M, startcol, full ? TRUE : FALSE);
}
void m4ri_echelonize_naive(m4ri_handle_t M, bool full)
{
    mzd_echelonize_naive(*M, full ? TRUE : FALSE);
}
void m4ri_invert_naive(m4ri_handle_t M, const m4ri_handle_t A, const m4ri_handle_t I)
{
    mzd_invert_naive(*M, *A, *I);
}

void m4ri_echelonize(m4ri_handle_t M, bool full)
{
    mzd_echelonize(*M, full ? TRUE : FALSE);
}
void m4ri_echelonize_pluq(m4ri_handle_t M, bool full)
{
    mzd_echelonize_pluq(*M, full ? TRUE : FALSE);
}
void m4ri_echelonize_m4ri(m4ri_handle_t M, bool full, int k)
{
    mzd_echelonize_m4ri(*M, full ? TRUE : FALSE, k);
}
void m4ri_mul(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B)
{
    mzd_mul(*dst, *A, *B, __M4RI_STRASSEN_MUL_CUTOFF);
}
void m4ri_addmul(m4ri_handle_t dst, const m4ri_handle_t A, const m4ri_handle_t B)
{
    mzd_addmul(*dst, *A, *B, __M4RI_STRASSEN_MUL_CUTOFF);
}

MCCL_END_NAMESPACE
