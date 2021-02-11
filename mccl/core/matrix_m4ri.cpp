#include <mccl/core/matrix_m4ri.hpp>
#include <m4ri/m4ri.h>

MCCL_BEGIN_NAMESPACE

struct m4ri_data_t
{
    mzd_t mzd;
    mzd_block_t mzd_block;
    std::vector<uint64_t*> rows;
};
    
    void make_mzd(m4ri_data_t& dst, const detail::matrix_base_ref_t<uint64_t>& src)
    {
        dst.mzd.nrows = src.rows;
        dst.mzd.ncols = src.columns;
        dst.mzd.width = (src.columns+src.word_bits-1)/src.word_bits;
        dst.mzd.rowstride = src.stride;
        dst.mzd.offset_vector = 0;
        dst.mzd.row_offset = 0;
        dst.mzd.flags = mzd_flag_nonzero_excess | mzd_flag_windowed_zerooffset;
        dst.mzd.blockrows_log = 31 - __builtin_clz(src.rows);
        dst.mzd.high_bitmask = src.lastwordmask();
        
        dst.mzd_block.size = src.rows * src.stride;
        dst.mzd_block.begin = src.data();
        dst.mzd_block.end = src.data(src.rows);
        
        dst.mzd.blocks = &dst.mzd_block;
        dst.rows.resize(src.rows);
        for (size_t i = 0; i < src.rows; ++i)
            dst.rows[i] = src.data(i);
        dst.mzd.rows = &dst.rows[0]; // <= try to get away with not providing pointers to rows structure
    }


m4ri_handle_t create_m4ri_handle(m4ri_ref_t& m)
{
    m4ri_data_t* h = new m4ri_data_t;
    make_mzd(*h, m.base());
    return h;
}

void transpose(m4ri_handle_t dst, const m4ri_handle_t src)
{
    mzd_transpose(&static_cast<m4ri_data_t*>(dst)->mzd, &static_cast<const m4ri_data_t*>(src)->mzd);
}


MCCL_END_NAMESPACE
