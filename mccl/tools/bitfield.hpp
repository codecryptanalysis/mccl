#ifndef MCCL_TOOLS_BITFIELD_HPP
#define MCCL_TOOLS_BITFIELD_HPP

#include <mccl/config/config.hpp>

MCCL_BEGIN_NAMESPACE

// three stage collision bitfield:
// stage 1: compute all L1 values and set the first bit of the 2-bit value at the corresponding address
// stage 2: compute all L2 values, check the address. if the first bit is set (a collision with L1) then set the second bit and store the L2 value
// stage 3: compute all L1 values again, store the ones with second bit set at the corresponding address
template<bool usefilter1 = false, bool usefilter2 = false>
struct staged_bitfield
{
    // each 'address' is mapped to a bit position of a uint32_t word
    // the bitfield actually consists of a vector of uint64_t
    // the bottom half is the L1 uint32_t word
    // the top half is the L2 uint32_t word
    std::vector<uint64_t> bitfield;
    // the filter is another bitfield, but with a shorter address space
    // to obtain a speed-up:
    //  (1) the bitfield should not fit any cachelevel
    //  (2) it should be small enough to fit entirely in some cachelevel
    //  (3) it should still be big enough to filter a significant factor of look-ups
    // now each address uses only 1 bit, so filter1 for L1, filter2 for L2
    // filter1 can be used in stage 2 to quickly filter non-collisions of L2 values
    // filter2 can be used in stage 3 to quickly filter non-collisions of L1 values
    std::vector<uint64_t> filter1;
    std::vector<uint64_t> filter2;

    uint64_t addressmask_bitfield;
    uint64_t addressmask_filter1;
    uint64_t addressmask_filter2;
    uint32_t bitfield_bitshift;
    
    void clear()
    {
        std::fill(bitfield.begin(), bitfield.end(), uint64_t(0));
        std::fill(filter1.begin(), filter1.end(), uint64_t(0));
        std::fill(filter2.begin(), filter2.end(), uint64_t(0));
    }

    void resize(size_t bitfieldaddressbits, size_t filter1addressbits = 0, size_t filter2addressbits = 0)
    {
        // check inputs
        if (bitfieldaddressbits < 5)
            throw std::runtime_error("staged_bitfield::resize(): bitfieldaddressbits should be at least 5");
        if (usefilter1 == true && filter1addressbits < 6)
            throw std::runtime_error("staged_bitfield::resize(): filter1 will be used, so filter1addressbits must be >= 6");
        if (usefilter2 == true && filter2addressbits < 6)
            throw std::runtime_error("staged_bitfield::resize(): filter2 will be used, so filter2addressbits must be >= 6");
        if (usefilter1 == false && filter1addressbits != 0)
            throw std::runtime_error("staged_bitfield::resize(): filter1 will NOT be used, so filter1addressbits must be 0");
        if (usefilter2 == false && filter2addressbits != 0)
            throw std::runtime_error("staged_bitfield::resize(): filter2 will NOT be used, so filter2addressbits must be 0");

        bitfield.resize(size_t(1) << (bitfieldaddressbits - 5));
        addressmask_bitfield = uint64_t(bitfield.size() - 1);

        if (usefilter1)
        {
            filter1.resize(size_t(1) << (filter1addressbits - 6));
            addressmask_filter1 = uint64_t(filter1.size() - 1);
        }
    
        if (usefilter2)
        {
            filter2.resize(size_t(1) << (filter2addressbits - 6));
            addressmask_filter2 = uint64_t(filter2.size() - 1);
        }

        // always call clear
        clear();
    }
        
    inline void filter1set(uint64_t L1val)
    {
        if (!usefilter1)
            return;
        filter1[ (L1val/64) & addressmask_filter1 ] |= uint64_t(1) << (L1val%64);
    }
    inline void filter2set(uint64_t L2val)
    {
        if (!usefilter2)
            return;
        filter2[ (L2val/64) & addressmask_filter2 ] |= uint64_t(1) << (L2val%64);
    }
    inline bool filter1get(uint64_t L2val)
    {
        if (!usefilter1)
            return true;
        return 0 != (filter1[ (L2val/64) & addressmask_filter1 ] |= uint64_t(1) << (L2val%64));
    }
    inline bool filter2get(uint64_t L1val)
    {
        if (!usefilter2)
            return true;
        return 0 != (filter2[ (L1val/64) & addressmask_filter2 ] |= uint64_t(1) << (L1val%64));
    }
    
    inline void stage1(uint64_t L1val)
    {
        bitfield[ (L1val/32) & addressmask_bitfield ] |= uint64_t(1) << (L1val%32);
        filter1set(L1val);
    }
    inline bool stage2(uint64_t L2val)
    {
        if (!filter1get(L2val))
            return false;
        uint64_t& x = bitfield[ (L2val/32) & addressmask_bitfield ];
        uint64_t L1bitval = uint64_t(1) << (L2val%32);
        if (0 == (x & L1bitval))
            return false;
        x |= L1bitval<<32;
        filter2set(L2val);
        return true;
    }
    inline bool stage3(uint64_t L1val)
    {
        if (!filter2get(L1val))
            return false;
        return 0 != (bitfield[ (L1val/32) & addressmask_bitfield ] & ((uint64_t(1)<<32) << (L1val%32)));
    }
};

MCCL_END_NAMESPACE

#endif
