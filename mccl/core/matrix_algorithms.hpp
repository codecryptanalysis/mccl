#ifndef MCCL_CORE_MATRIX_ALGORITHMS_HPP
#define MCCL_CORE_MATRIX_ALGORITHMS_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix.hpp>
#include <mccl/core/random.hpp>

MCCL_BEGIN_NAMESPACE

template<typename Func = std::function<bool(size_t,size_t)>>
void fill(const mat_view& m, Func& f)
{
    for (size_t r = 0; r < m.rows(); ++r)
        for (size_t c = 0; c < m.columns(); ++c)
            m.setbit(r,c, f(r,c));
}
template<typename Func = std::function<bool(size_t)>>
void fill(const vec_view& v, Func& f)
{
    for (size_t c = 0; c < v.columns(); ++c)
        v.setbit(c, f(c));
}

template<typename Func = std::function<uint64_t(size_t,size_t)>>
void fillword(const mat_view& m, Func& f)
{
    if (m.rows() == 0 || m.columns() == 0)
        return;
    const size_t words = m.rowwords();
    const uint64_t lwm = detail::lastwordmask(m.columns());
    for (size_t r = 0; r < m.rows(); ++r)
    {
        uint64_t* first = m.ptr.data(r);
        uint64_t* last = first + words - 1;
        size_t w = 0;
        for (; first != last; ++first,++w)
            *first = f(r,w);
        *first = (f(r,w) & lwm) | (*first & ~lwm);
    }
}
template<typename Func = std::function<uint64_t(size_t)>>
void fillword(const vec_view& v, Func& f)
{
    if (v.columns() == 0)
        return;
    const size_t words = v.rowwords();
    const uint64_t lwm = detail::lastwordmask(v.columns());
    uint64_t* first = v.ptr.data(0);
    uint64_t* last = first + words - 1;
    size_t w = 0;
    for (; first != last; ++first,++w)
        *first = f(w);
    *first = (f(w) & lwm) | (*first & ~lwm);
}


template<typename Generator>
void fillgenerator(const mat_view& m, Generator& g)
{
    if (m.rows() == 0 || m.columns() == 0)
        return;
    const size_t words = m.rowwords();
    const uint64_t lwm = detail::lastwordmask(m.columns());
    for (size_t r = 0; r < m.rows(); ++r)
    {
        uint64_t* first = m.ptr.data(r);
        uint64_t* last = first + words - 1;
        for (; first != last; ++first)
            g(*first);
        uint64_t oldval = *first;
        g(*first);
        *first = (*first & lwm) | (oldval & ~lwm);
    }
}
template<typename Generator>
void fillgenerator(const vec_view& v, Generator& g)
{
    if (v.columns() == 0)
        return;
    const size_t words = v.rowwords();
    const uint64_t lwm = detail::lastwordmask(v.columns());
    uint64_t* first = v.ptr.data(0);
    uint64_t* last = first + words - 1;
    for (; first != last; ++first)
        g(*first);
    uint64_t oldval = *first;
    g(*first);
    *first = (*first & lwm) | (oldval & ~lwm);
}

inline void fillrandom(const mat_view& m)
{
    mccl_base_random_generator gen;
    fillgenerator(m, gen);
}
inline void fillrandom(const mat_view& m, mccl_base_random_generator& gen)
{
    fillgenerator(m, gen);
}
inline void fillrandom(const vec_view& v)
{
    mccl_base_random_generator gen;
    fillgenerator(v, gen);
}
inline void fillrandom(const vec_view& v, mccl_base_random_generator& gen)
{
    fillgenerator(v, gen);
}


// full row reduction of matrix m over columns [column_start,column_end)
// pivots may be selected from rows [pivot_start,rows())
// returns pivotend = pivot_start + nrnewrowpivots
template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
size_t echelonize(matrix_t& m, size_t column_start = 0, size_t column_end = ~size_t(0), size_t pivot_start = 0)
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

// full row reduction on *transposed* matrix
// aka full column reduction of matrix m over rows [row_start,row_end)
// pivots may be selected from columns [pivot_start,columns())
// returns pivotend = pivot_start + nrnewcolpivots
template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
size_t echelonize_col(matrix_t& m, size_t row_start = 0, size_t row_end = ~size_t(0), size_t pivot_start = 0)
{
    if (row_end > m.rows())
        row_end = m.rows();
    for (size_t r = row_start; r < row_end; ++r)
    {
        // find pivot for row r
        size_t p = pivot_start;
        for (; p < m.columns() && m(r,p) == false; ++p)
            ;
        // if no pivot found the continue with next row
        if (p >= m.columns())
            continue;
        // swap column if necessary
        if (p != pivot_start)
            m.swapcolumns(p, pivot_start);
        // reduce row r with column pivot_start
        // note: first clear bit pivot_start in row r to prevent row r & column pivot_start to be changed
        vec_view pivotrow(m[r]);
        pivotrow.clearbit(pivot_start);
        auto mrowit = m[0];
        for (size_t r2 = 0; r2 < m.rows(); ++r2,++mrowit)
            if (m(r2,pivot_start))
                mrowit.vxor(pivotrow);
        // now just set pivotrow to zero except for column pivot_start
        pivotrow.clear();
        pivotrow.setbit(pivot_start);
        // increase pivot_start for next row
        ++pivot_start;
    }
    return pivot_start;
}

// full *column* reduction of matrix m over rows [row_start,row_end) with *reverse* column ordering
// pivots may be selected from columns [0, pivot_start)
// returns pivotend = pivot_start - nrnewcolpivots
template<typename matrix_t, MCCL_ENABLE_IF_MATRIX(matrix_t)>
size_t echelonize_col_rev(matrix_t& m, size_t row_start = 0, size_t row_end = ~size_t(0), size_t pivot_start = ~size_t(0))
{
    if (row_end > m.rows())
        row_end = m.rows();
    if (pivot_start > m.columns())
        pivot_start = m.columns();
    for (size_t r = row_start; r < row_end; ++r)
    {
        // find pivot for row r
        size_t p = pivot_start;
        for (; p > 0 && m(r,p-1) == false; --p)
            ;
        // if no pivot found the continue with next row
        if (p == 0)
            continue;
        // operate on column pivot_start-1 and p-1 instead
        --p; --pivot_start;
        // swap column if necessary
        if (p != pivot_start)
            m.swapcolumns(p, pivot_start);
        // reduce row r with column pivot_start
        // note: first clear bit pivot_start in row r to prevent row r & column pivot_start to be changed
        vec_view pivotrow(m[r]);
        pivotrow.clearbit(pivot_start);
        auto mrowit = m[0];
        for (size_t r2 = 0; r2 < m.rows(); ++r2,++mrowit)
            if (m(r2,pivot_start))
                mrowit.vxor(pivotrow);
        // now just set pivotrow to zero except for column pivot_start
        pivotrow.clear();
        pivotrow.setbit(pivot_start);
        // need to decrease pivot_start by 1 for next row, but already done
    }
    return pivot_start;
}



inline mat dual_matrix(const cmat_view& m)
{
        mat msf(m);
        // echelonize msf
        size_t pr = echelonize(msf);
        // remove zero rows
        msf.resize( pr, msf.columns() );
        size_t rows = msf.rows(), columns = msf.columns();

        // compute transpose over which we'll compute
        mat msfT = m_transpose(msf);

        // we want to swap columns such that msf = ( I_n | P )
        // => we will swap rows of msfT such that msfT = (I_n | P)^T
        std::vector< std::pair<size_t,size_t> > columnswaps;
        vec tmp;
        for (size_t p = 0; p < rows; ++p)
        {
                // find msf column = msfT row with single bit set at position p
                size_t c = p;
                for (; c < columns; ++c)
                        if (hammingweight(msfT[c]) == 1 && msfT(c,p) == true)
                                break;
                if (c == p)
                        continue;
                if (c == columns)
                        throw std::runtime_error("dual_matrix(): internal error 1");
                // swap columns
                columnswaps.emplace_back(p, c);
                tmp = msfT[p] ^ msfT[c];
                msfT[p] ^= tmp;
                msfT[c] ^= tmp;
        }
        // we should now have a identity matrix as left submatrix
        // msf = (I_n | P), so msfdual = ( P^T | I_(n-k) )
        mat dual(columns - rows, columns);
        // write P^T
        dual.submatrix(0, dual.rows(), 0, msf.rows()) = m_copy( msfT.submatrix(msf.rows(), dual.rows(), 0, msf.rows()));
        // write I_(n-k)
        for (size_t r = 0; r < dual.rows(); ++r)
                dual.setbit(r, rows + r, true);
        // undo column swaps
        while (!columnswaps.empty())
        {
                auto pc = columnswaps.back();
                columnswaps.pop_back();
                dual.swapcolumns(pc.first, pc.second);
        }
        return dual;
}

inline mat prepend_identity(const cmat_view& m)
{
        mat retT(m.rows() + m.columns(), m.rows());
        retT.setidentity();
        retT.submatrix(m.rows(), m.columns(), 0, m.rows()) = m_transpose(m);
        mat ret = m_transpose(retT);
        return ret;
}

MCCL_END_NAMESPACE

#endif
