#ifndef MCCL_CORE_MATRIX_PARTIALMUL_HPP
#define MCCL_CORE_MATRIX_PARTIALMUL_HPP

#include <mccl/core/matrix.hpp>

MCCL_BEGIN_NAMESPACE

namespace detail {

unsigned int alignment(uintptr_t ptr)
{
    // ptr      = ... 1 0000
    // ptr-1    = ... 0 1111
    // ~(ptr-1) = ~~~ 1 0000
    // return   = 000 1 0000
    return ptr & ~(ptr-1);
}

template<unsigned int _bytes>
struct alignas(_bytes) alignedblock
{
    static_assert(_bytes == (_bytes & ~(_bytes-1)), "bytes is not power of 2");
    static_assert(_bytes % sizeof(uint64_t) == 0, "bytes is not multiple of wordsize");
    static const unsigned int bytes = _bytes;
    static const unsigned int words = bytes/sizeof(uint64_t);
    
    uint64_t w[words];
    
    alignedblock(): w() {}
    alignedblock(const alignedblock&) = default;
    alignedblock(alignedblock&&) = default;

    alignedblock& operator=(const alignedblock&) = default;
    alignedblock& operator=(alignedblock&&) = default;

    static bool isaligned(const void* ptr) { return (uintptr_t(ptr)%bytes) == 0; }

    alignedblock& read(const void* ptr) { return *this = *reinterpret_cast<const alignedblock*>(ptr); }
    void write(void* ptr) { *reinterpret_cast<alignedblock*>(ptr) = *this; }
    
    alignedblock& copy(const uint64_t* w2) { for (unsigned i = 0; i < words; ++i) w[i] = w2[i]; return *this; }

    alignedblock& operator^=(const alignedblock& w2)
    {
        for (unsigned i = 0; i < words; ++i)
            w[i] ^= w2.w[i];
        return *this;
    }
    alignedblock& operator&=(const alignedblock& w2)
    {
        for (unsigned i = 0; i < words; ++i)
            w[i] &= w2.w[i];
        return *this;
    }
    alignedblock& operator|=(const alignedblock& w2)
    {
        for (unsigned i = 0; i < words; ++i)
            w[i] |= w2.w[i];
        return *this;
    }
    alignedblock operator^(const alignedblock& w2) const
    {
        alignedblock r(*this);
        return r ^= w2;
    }
    alignedblock operator&(const alignedblock& w2) const
    {
        alignedblock r(*this);
        return r &= w2;
    }
    alignedblock operator|(const alignedblock& w2) const
    {
        alignedblock r(*this);
        return r |= w2;
    }
};

template<size_t _generators, size_t _bytes>
struct alignedblock_table
{
    static_assert(_generators >= 3, "cacheline_table_t needs at least 3 generators");

    typedef alignedblock<_bytes> block;

    static const unsigned int generators = _generators;
    static const unsigned int size = size_t(1) << generators;

    static const unsigned int bytes = _bytes;
    static const unsigned int words = block::words;
    
    block T[size];

    // rows of any word type word_t can be processed with alignedblock and alignedblock_table
    // call with pointer to first word of block and stride in words
    // first copies generators to internal generator table and then generates table with _initialize
    template<typename word_t>
    void initialize(const word_t* ptr, size_t word_stride = 1)
    {
        block bgs[generators];
        for (unsigned i = 0; i < generators; ++i,ptr+=word_stride)
            memcpy(bgs+i, ptr, sizeof(block));
        _initialize(bgs);
    }
    // same as above, but a mask is applied to generators such that it won't affect bits or words with 0-bits in mask
    template<typename word_t>
    void initialize_masked(const word_t* ptr, size_t word_stride, const word_t mask[words])
    {
        block bgs[generators], bmask;
        memcpy(&bmask, mask, sizeof(block));
        for (unsigned i = 0; i < generators; ++i,ptr+=word_stride)
        {
            memcpy(bgs+i, ptr, sizeof(block));
            bgs[i] &= bmask;
        }
        _initialize(bgs);
    }
    // create table
    // instead of more complex grey code enumeration that has dynamic indexing which is difficult for compiler to optimize
    // this has inner loop unrolling of 8 very simple iterations for which one prior element of table has to be loaded
    void _initialize(const block* ptr)
    {
        const block b0 = ptr[0], b1 = ptr[1], b2 = ptr[2];
        T[0] = block();
        T[1] = b0;
        T[2] = b1;
        T[3] = b0^b1;
        T[4] = b2;
        T[5] = b2^b0;
        T[6] = b2^b1;
        T[7] = T[6]^b0;
        for (unsigned g = 3; g < generators; ++g)
        {
            const block bg = ptr[g];
            for (unsigned i = 0; i < (1<<g); i += 8)
            {
                block bi = T[i] ^ bg;
                T[(1<<g) + i + 0] = bi;
                T[(1<<g) + i + 1] = bi ^ b0;
                T[(1<<g) + i + 2] = bi ^ b1;
                T[(1<<g) + i + 3] = bi ^ b1 ^ b0;
                T[(1<<g) + i + 4] = bi ^ b2;
                T[(1<<g) + i + 5] = bi ^ b2 ^ b0;
                T[(1<<g) + i + 6] = bi ^ b2 ^ b1;
                T[(1<<g) + i + 7] = bi ^ b2 ^ b1 ^ b0;
            }
        }
    }
    
    const block& operator[](unsigned int i) const { return T[i]; }
    
    // apply on data rows using a corresponding index table with indices to the lookup table
    // through rightshift by given 'indexrshift' and then modulo 'size'(=1<<generators), any consecutive bits within index can be used as address
    // note that if initialize_masked was used, then bits and words corresponding to mask 0-bits will be read & written by apply but not changed
    template<typename indexword_t>
    void apply(const indexword_t* indexfirst, const indexword_t* indexlast, unsigned int indexrshift, block* datafirst, size_t block_stride)
    {
        for (; indexfirst != indexlast, ++indexfirst, datafirst+=block_stride)
            *datafirst ^= T[((*indexfirst)>>indexrshift) % size];
    }
    template<typename indexword_t, typename dataword_t>
    void apply(const indexword_t* indexfirst, const indexword_t* indexlast, unsigned int indexrshift, dataword_t* datafirst, size_t dataword_stride)
    {
        assert(block::isaligned(datafirst), "data pointer is not aligned");
        assert(block::isaligned(dataword_stride*sizeof(dataword_t), "data stride is not aligned");
        apply(indexfirst, indexlast, indexrshift, reinterpret_cast<block*>(datafirst), (dataword_stride*sizeof(dataword_t))/sizeof(block));
    }
    // non in-place versions
    template<typename indexword_t>
    void apply(const indexword_t* indexfirst, const indexword_t* indexlast, unsigned int indexrshift, const block* srcfirst, size_t srcstride, block* dstfirst, size_t dststride)
    {
        for (; indexfirst != indexlast, ++indexfirst, srcfirst+=srcstride, dstfirst+=dststride)
            *dstfirst = *srcfirst ^ T[((*indexfirst)>>indexrshift) % size];
    }
    template<typename indexword_t, typename dataword_t>
    void apply(const indexword_t* indexfirst, const indexword_t* indexlast, unsigned int indexrshift, const dataword_t* srcfirst, size_t srcstride, dataword_t* dstfirst, size_t dststride)
    {
        assert(block::isaligned(srcfirst), "src pointer is not aligned");
        assert(block::isaligned(srcstride*sizeof(dataword_t), "src stride is not aligned");
        assert(block::isaligned(dstfirst), "dst pointer is not aligned");
        assert(block::isaligned(dststride*sizeof(dataword_t), "dst stride is not aligned");
        apply(indexfirst, indexlast, indexrshift, 
            reinterpret_cast<const block*>(srcfirst), (srcstride*sizeof(dataword_t))/sizeof(block), 
            reinterpret_cast<block*>(dstfirst), (dststride*sizeof(dataword_t))/sizeof(block)
            );
    }
};

}

// performs partial matrix multiplication for 32-bit index
// uses four 8-generator alignedblock_tables
template<typename dataword_t>
void inplace_partial_matmul32(const uint32_t* indexfirst, const uint32_t* indexlast, dataword_t* datafirst, dataword_t* genfirst, size_t rowwords, size_t stride)
{
    // we only use hardcoded 32-byte (single avx2 register) and 64-byte alignedblocks (cacheline, 2 avx2 registers) for now
    struct TT {
        alignedblock_table<8,64> B64[4];
//        alignedblock_table<8,32> B32[4];
//        alignedblock_table<8,16> B16[4];
//        alignedblock_table<8,8> B8[4];
    };
    TT* t = new TT;
    // round datafirst pointer down to 64-byte alignment
    dataword_t* blockfirst64 = (dataword_t*)(uintptr_t(datafirst) & ~uintptr_t(63));
    // round dataend pointer up to 64-byte alignment
    dataword_t* blocklast64 = (dataword_t)( (uintptr_t(datafirst+rowwords)+63) & ~uintptr_t(63) );
    // now process in 64-byte aligned blocks
    const unsigned words64 = 64/sizeof(dataword_t);
    for (; blockfirst64 != blocklast64; blockfirst64 += words64)
    {
        dataword_t mask[words64];
        for (unsigned i = 0; i < words64; ++i)
            mask[i] = ~dataword_t(0);
        // we rounded pointer down, so may need to start mask with zero words
        if (blockfirst64 < datafirst)
        {
            for (unsigned i = 0; i < datafirst-blockfirst64; ++i)
                mask[i] = 0;
        }
        // we may end with a partial 64-byte block, so may need to end mask with zero words
        if (datafirst+rowwords < blockfirst64 + words64)
        {
            unsigned j = (blockfirst64+words64) - (datafirst+rowwords);
            for (unsigned i = words - j; i < words; ++i)
                mask[i] = 0;
        }
        const ptrdiff_t genoffset = blockfirst64 - datafirst;
        t->B64[0].initialize_masked(genfirst + genoffset, stride, mask);
        t->B64[1].initialize_masked(genfirst + genoffset + 8*stride, stride, mask);
        t->B64[2].initialize_masked(genfirst + genoffset + 16*stride, stride, mask);
        t->B64[3].initialize_masked(genfirst + genoffset + 24*stride, stride, mask);

        // TODO: these apply loops should be interleaved instead of separate calls
        alignedblock_table<8,64>::block* blockptr;
        size_t blockstride = (stride*sizeof(dataword_t))/64;
        for (const uint32_t* index; index != indexlast; ++index,blockptr+=blockstride)
        {
            *blockptr ^= t->B64[0].T[*index % 256];
            *blockptr ^= t->B64[1].T[(*index>>8) % 256];
            *blockptr ^= t->B64[2].T[(*index>>16) % 256];
            *blockptr ^= t->B64[3].T[(*index>>24) % 256];
        }
    }
}


MCCL_END_NAMESPACE

#endif
