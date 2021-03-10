#ifndef MCCL_CORE_MATRIX_ECHELON_HPP
#define MCCL_CORE_MATRIX_ECHELON_HPP

#include <mccl/core/matrix.hpp>
#include <vector>

MCCL_BEGIN_NAMESPACE

namespace detail {

template<unsigned int bits>
struct bitfield_type {};
template<> struct bitfield_type< 8> { typedef uint8_t type; }
template<> struct bitfield_type<16> { typedef uint16_t type; }
template<> struct bitfield_type<32> { typedef uint32_t type; }
template<> struct bitfield_type<64> { typedef uint64_t type; }

/* helper struct to extract columns from matrix */
template<typename _word_t, typename _matref_t>
struct columns_extractor
{
    typedef _matref_t matref_t;
    typedef typename matref_t::base_ref_t::data_t matword_t;
    typedef _word_t rowword_t;
    static const unsigned int maxcolumns = sizeof(rowword_t)*8;
    // maximum number of matrix words to read: round rowword_t up and add 1
    static const unsigned int maxmatwords = (sizeof(rowword_t) + sizeof(matword_t) - 1) / sizeof(matword_t) + 1;
    
    std::vector<rowword_t> rows;
    unsigned int columns;
    
    columns_extractor() { clear(); }
    
    void clear()
    {
        columns = 0;
        rows.clear();
    }

    template<unsigned int matwords>
    void _extract_columns(const matref_t& m, size_t wordoffset, unsigned int* wordshift, rowword_t* wordmask)
    {
        if (matwords <= maxmatwords)
        {
            auto endwordptr = m.data(m.rows(),0) + wordoffset;
            rowword_t* ptr = &rows[0];
            for (auto wordptr = m.data(0,0) + wordoffset; wordptr != endwordptr; wordptr += m.rowstride, ++ptr)
            {
                *ptr = static_cast<rowword_t>(wordptr[0]>>wordshift[0]) & wordmask[0];
                for (unsigned int j = 1; j < matwords; ++j)
                    *ptr ^= (static_cast<rowword_t>(wordptr[j])<<wordshift[j]) & wordmask[j];
            }
        }
    }
    
    void extract_columns(const matref_t& m, size_t columnstart, size_t _columns)
    {
        if (_columns > maxcolumns)
            _columns = maxcolumns;
        clear();
        columns = _columns;
        if (columns == 0)
            return;
            
        /* determine operations to extract columns */
        unsigned int wordshift[maxmatwords]; // first shift is right, other shifts are left
        rowword_t wordmask[maxmatwords];
        unsigned int matwords = 0;
        static const unsigned int matwordbits = sizeof(matword_t)*8;
        
        for (size_t cols = 0; cols < columns; )
        {
            unsigned int bitsextracted = columns - cols;
            if (cols == 0)
            {
                wordshift[0] = columnstart % (sizeof(matword_t)*8);
                unsigned int bitsextracted = std::min<unsigned int>(matwordbits - wordshift[0], bitsextracted);
            }
            else
            {
                wordshift[matwords] = cols;
                unsigned int bitsextracted = std::min<unsigned int>(matwordbits, bitsextracted);
            }
            // wordmask has bitsextracted # one bits starting at bit position cols, zero bits elsewhere
            wordmask[matwords] = (~((~rowword_t(0)) << bitsextracted)) << cols;
            cols += bitsextracted;
            ++matwords;
        }
        
        /* extract columns */
        index.resize(m.rows());
        const size_t wordoffset = columns / matwordbits;
             if (1 <= maxmatwords && 1==matwords) _extract_columns<1>(m, wordoffset, wordshift, wordmask);
        else if (2 <= maxmatwords && 2==matwords) _extract_columns<2>(m, wordoffset, wordshift, wordmask);
        // unlikely rowword_t is bigger than matword_t, so 2 should be max
//        else if (3 <= maxmatwords && 3==matwords) _extract_columns<3>(m, wordoffset, wordshift, wordmask);
//        else if (4 <= maxmatwords && 4==matwords) _extract_columns<4>(m, wordoffset, wordshift, wordmask);
        else throw std::runtime_error("echelon_helper: extract_columns failed");
    }

};





template<unsigned int _maxbits = 64>
struct local_rowreduce
{
    static_assert(_maxbits <= 64, "local_rowreduce is limited to 64 bits at a time");
    static const unsigned int maxbits = _maxbits;

    // amount of total rows
    size_t rows;
    // which block of rows to process
    size_t pivotstart, pivots;

    // first swap these rows
    std::pair<size_t,size_t> rowswap[maxbits];
    unsigned int rowswaps;

    // then compute the new rows as sums of old rows
    // oldrow pivotstart+i should be added to newrows add[i][0], ... , add[i][addlen[i]-1]
    uint8_t add[maxbits][maxbits]; 
    uint8_t addlen[maxbits];
    
    uint64_t activebitmask;
    unsigned int bits;

    local_echelon() { clear(); }
    
    void clear()
    {
        rows = pivotstart = pivots = 0;
        rowswaps = 0;
        activebits = 0;
    }
    
    template<typename word_t>
    void rowreduce(word_t* first, size_t _rows, size_t pivotrow_min, unsigned _bits)
    {
        if (_bits > maxbits || _bits > sizeof(word_t)*8)
            throw;
        bits = _bits;

        // reset state
        rows = _rows;
        pivotstart = pivotrow_min;
        pivots = 0;
        rowswaps = 0;
        activebitmask = 0;
        
        // which row to reduce bit b: first[pivotrow[b]] if pivotrow[b]!=-1
        size_t pivotrow[maxbits];
        // transformation matrix: bit p corresponds to pivot p
        word_t U[maxbits];

        // do row reduction over `bits` columns 
        // swap rows where necessary, and update U
        for (unsigned int bit = 0; bit < bits; ++bit)
        {
            const word_t bitval = word_t(1) << bit;

            size_t p = pivotrow_min + pivots, r = p;
            for (; r < rows ; ++r)
            {
                word_t rowr = first[r];
                word_t rowU = word_t(1) << pivots;
                // reduce with previous pivots
                for (unsigned int b = 0; b < bit; ++b)
                    if (((rowr>>b)&1) && pivotrow[b] < maxbits)
                    {
                        rowr ^= first[ pivotrow_min + pivotrow[b] ];
                        rowU ^= U[ pivotrow[b] ];
                    }
                // check if this row can be pivot for this bit
                if (rowr & bitval)
                {
                    U[pivots] = rowU;
                    first[r] = rowr;
                    break;
                }
            }
            // if no pivot row found then we cannot row-reduce this bit position
            if (r == rows)
            {
                pivotrow[bit] = ~size_t(0);
                continue;
            }
            // found pivot for bit `bit`
            pivotrow[bit] = pivots;
            activebitmask |= static_cast<uint64_t>(bitval);
            // swap rows if needed
            if (r != p)
            {
                row_swaps.emplace(p, r);
                std::swap(first[p], first[r]);
            }
            // reduce previous pivots
            for (size_t i = 0; i < pivots; ++i)
            {
                if (first[pivotrow_min + i] & bitval)
                {
                    first[rowpivot_min + i] ^= first[p];
                    U[i] ^= U[pivots];
                }
            }
            ++pivots;
        }

        // now create transformation tables
        for (size_t i = 0; i < pivots; ++i)
        {
            addlen[i] = 0;
            for (unsigned b = 0; b < pivots; ++b)
            {
                if ((U[b]>>i)&1)
                {
                    add[i][addlen[i]] = b;
                    ++addlen[i];
                }
            }
        }
    }


    template<typename T>
    void apply_pivotstransformation(T* m, size_t rowstride)
    {
        // apply row swaps
        for (unsigned i = 0; i < rowswaps; ++i)
            std::swap(m[rowswap[i].first*rowstride], m[rowswap[i].second*rowstride]);
        // apply sub_reduction
        {
            T tmp[maxbits];
            memset(tmp, 0, sizeof(T)*maxbits);
            T* m2 = m + pivotstart*rowstride;
            for (unsigned i = 0; i < pivots; ++i, m2+=rowstride)
            {
                for (unsigned j = 0; j < addlen[i]; ++j)
                    tmp[add[i][j]] ^= *m2;
            }
            m2 = m + rowstride*pivotstart;
            for (unsigned i = 0; i < pivots; ++i, m2+=rowstride)
                *m2 = tmp[i];
        }
    }

    // preparation stage for method of 4 russians
    // copy pivots from m to p, interlaced with zero words, such that bit positions match pivot index
    template<typename T>
    void prepare_pivots_m4r(T* m, size_t rowstride, T* p)
    {
        T zero(0);
        m += pivotstart * rowstride;
        for (unsigned b = 0; b < bits; ++b, ++p)
        {
            if ((activebitmask >> b)&1)
            {
                *p = *m;
                m += rowstride;
            } else
            {
                *p = zero;
            }
        }
    }
};

}

template<typename _matrix_ref_t, unsigned int _tables, unsigned int _tablebits = 8>
struct echelon_helper
{
    typedef _matrix_ref_t matref_t;
    typedef matref_t::base_ref_t::data_t word_t;
    
    static const unsigned int tablebits = _tablebits;
    static const unsigned int tables = _tables;
    static const unsigned int totalbits = tablebits * tables;
    static const unsigned int rowbits = totalbits; 
    static const unsigned int word_bits = matref_t::base_ref_t::word_bits;
    typedef typename echelon_helper_traits< (rowbits<=8) ? 8 : (rowbits<=16) ? 16 : (rowbits<=32) ? 32 : 64>::value_t row_t;
    static const unsigned int maxmatwords = (sizeof(row_t)*8 + word_bits - 1)/word_bits + 1;
    

};

MCCL_END_NAMESPACE

#endif
