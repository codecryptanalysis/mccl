#include <mccl/config/config.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <cassert>

MCCL_BEGIN_NAMESPACE

namespace detail {

inline size_t hammingweight(uint64_t n)
{
	return __builtin_popcountll(n);
}
inline size_t hammingweight(uint32_t n)
{
	return __builtin_popcountl(n);
}
template<size_t bits>
inline size_t hammingweight(const uint64_block_t<bits>& v)
{
	size_t hw = 0;
	for (unsigned i = 0; i < v.size; ++i)
		hw += __builtin_popcountll(v.v[i]);
	return hw;
}

inline uint64_t rotate_right(uint64_t x, unsigned n)
{
	return (x>>n)|(x<<(64-n));
}
inline uint64_t rotate_left(uint64_t x, unsigned n)
{
	return (x<<n)|(x>>(64-n));
}

#ifdef _MSC_VER
inline unsigned int trailing_zeroes(uint32_t n)
{
	unsigned long ret = 0;
	_BitScanForward(&ret, n);
	return ret;
}
#else
inline unsigned int trailing_zeroes(uint32_t n)
{
	return __builtin_ctz(n);
}
#endif



extern const uint64_block_t<64 > _lastblockmask64 [64];
extern const uint64_block_t<128> _lastblockmask128[128];
extern const uint64_block_t<256> _lastblockmask256[256];
extern const uint64_block_t<512> _lastblockmask512[512];
template<bool masked> uint64_block_t<64 > lastwordmask(size_t columns, block_tag<64 ,masked>) { return _lastblockmask64 [columns % 64 ]; }
template<bool masked> uint64_block_t<128> lastwordmask(size_t columns, block_tag<128,masked>) { return _lastblockmask128[columns % 128]; }
template<bool masked> uint64_block_t<256> lastwordmask(size_t columns, block_tag<256,masked>) { return _lastblockmask256[columns % 256]; }
template<bool masked> uint64_block_t<512> lastwordmask(size_t columns, block_tag<512,masked>) { return _lastblockmask512[columns % 512]; }

inline uint64_t lastwordmask(size_t cols)
{
	return _lastblockmask64[cols % 64].v[0];
}
inline uint64_t firstwordmask(size_t cols)
{
	return uint64_t(0) - (uint64_t(1)<<(cols%64));
}

// return the smallest 2^t >= n
template<typename Int>
inline Int next_pow2(Int n)
{
	const unsigned bits = sizeof(Int)*8;
	--n;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	if (8 < bits) n |= n >> 8;
	if (16 < bits) n |= n >> 16;
	if (32 < bits) n |= n >> 32;
	return ++n;
}



inline void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols, bool b)
{
	if (b)
		v_setcolumns(v,coloffset,cols);
	else
		v_clearcolumns(v,coloffset,cols);
}

inline void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols)
{
	if (v.columns == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = firstwordmask(coloffset);
	auto lwm = lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = lwm = fwm & lwm;
		firstword2 = firstword;
	}
	*(v.data()+firstword) |= fwm;
	auto first = v.data()+firstword2, last = v.data()+lastword;
	for (; first != last; ++first)
		*first |= ~uint64_t(0);
	*first |= lwm;
}

inline void v_flipcolumns(const v_ptr& v, size_t coloffset, size_t cols)
{
	if (v.columns == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = firstwordmask(coloffset);
	auto lwm = lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = fwm & lwm;
		lwm = 0;
		firstword2 = firstword;
	}
	*(v.data()+firstword) ^= fwm;
	auto first = v.data()+firstword2, last = v.data()+lastword;
	for (; first != last; ++first)
		*first ^= ~uint64_t(0);
	*first ^= lwm;
}

inline void v_clearcolumns(const v_ptr& v, size_t coloffset, size_t cols)
{
	if (v.columns == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = ~firstwordmask(coloffset);
	auto lwm = ~lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = lwm = fwm | lwm;
		firstword2 = firstword;
	}
	*(v.data()+firstword) &= fwm;
	auto first = v.data()+firstword2, last = v.data()+lastword;
	for (; first != last; ++first)
		*first = 0;
	*first &= lwm;
}



inline size_t v_hw(const cv_ptr& v)
{
	if (v.columns == 0)
		return 0;
	size_t words = (v.columns + 63)/64;
	uint64_t lwm = lastwordmask(v.columns);
	size_t hw = 0;
	auto first1 = v.data(), last1 = v.data() + words - 1;
	for (; first1 != last1; ++first1)
		hw += hammingweight(*first1);
	hw += hammingweight((*first1) & lwm);
	return hw;
}

inline size_t v_hw_and(const cv_ptr& v1, const cv_ptr& v2)
{
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
	if (v1.columns == 0)
		return 0;
	size_t words = (v1.columns + 63)/64;
	uint64_t lwm = lastwordmask(v1.columns);
	size_t hw = 0;
	auto first1 = v1.data(), last1 = v1.data() + words - 1;
	auto first2 = v2.data();
	for (; first1 != last1; ++first1,++first2)
		hw += hammingweight(*first1 & *first2);
	hw += hammingweight((*first1 & *first2) & lwm);
	return hw;
}

inline size_t v_hw_or(const cv_ptr& v1, const cv_ptr& v2)
{
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
	if (v1.columns == 0)
		return 0;
	size_t words = (v1.columns + 63)/64;
	uint64_t lwm = lastwordmask(v1.columns);
	size_t hw = 0;
	auto first1 = v1.data(), last1 = v1.data() + words - 1;
	auto first2 = v2.data();
	for (; first1 != last1; ++first1,++first2)
		hw += hammingweight(*first1 | *first2);
	hw += hammingweight((*first1 | *first2) & lwm);
	return hw;
}

inline size_t v_hw_xor(const cv_ptr& v1, const cv_ptr& v2)
{
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
	if (v1.columns == 0)
		return 0;
	size_t words = (v1.columns + 63)/64;
	uint64_t lwm = lastwordmask(v1.columns);
	size_t hw = 0;
	auto first1 = v1.data(), last1 = v1.data() + words - 1;
	auto first2 = v2.data();
	for (; first1 != last1; ++first1,++first2)
		hw += hammingweight(*first1 ^ *first2);
	hw += hammingweight((*first1 ^ *first2) & lwm);
	return hw;
}



template<size_t bits, bool masked>
inline bool v_isequal(const cv_ptr& v1, const cv_ptr& v2, block_tag<bits,masked>)
{
	if (v1.columns != v2.columns)
		return false;
	if (v1.columns == 0)
		return true;
	const size_t words = (v1.columns + bits - 1) / bits - (masked?1:0);
	auto first1 = make_block_ptr(v1.ptr, block_tag<bits,masked>()), last1 = first1 + words;
	auto first2 = make_block_ptr(v2.ptr, block_tag<bits,masked>());
	for (; first1 != last1; ++first1, ++first2)
		if (*first1 != *first2)
			return false;
	if (masked)
	{
		auto lwm = lastwordmask(v1.columns, block_tag<bits,masked>());
		if ((lwm & *first1) != (lwm & *first2))
			return false;
	}
	return true;
}

template<size_t bits, bool masked>
inline void v_swap(const v_ptr& v1, const v_ptr& v2, block_tag<bits,masked>)
{
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
	if (v1.columns == 0)
		return;
	const size_t words = (v1.columns+bits-1) / bits - (masked?1:0);
	auto first1 = make_block_ptr(v1.ptr, block_tag<bits,masked>()), last1 = first1 + words;
	auto first2 = make_block_ptr(v2.ptr, block_tag<bits,masked>());
	for (; first1 != last1; ++first1,++first2)
		std::swap(*first1, *first2);
	if (masked)
	{
		auto lwm = lastwordmask(v1.columns, block_tag<bits,masked>());
		auto diff = (*first1 ^ *first2) & lwm;
		*first1 ^= diff;
		*first2 ^= diff;
	}
}



#define MCCL_VECTOR_BASE_FUNCTION_1OP(func,expr) \
template<size_t bits, bool masked> \
inline void v_ ## func (const v_ptr& dst, block_tag<bits,masked>) \
{ \
	if (dst.columns == 0) \
		return; \
	size_t words = (dst.columns + bits-1)/bits - (masked?1:0); \
	auto firstd = make_block_ptr(dst.ptr, block_tag<bits,masked>()), lastd = firstd + words; \
	for (; firstd != lastd; ++firstd) \
		*firstd = expr ; \
	if (masked) \
	{ \
		auto lwm = lastwordmask(dst.columns, block_tag<bits,masked>()); \
		auto diff = lwm & ((expr) ^ *firstd); \
		*firstd ^= diff; \
	} \
}

MCCL_VECTOR_BASE_FUNCTION_1OP(not,~*firstd)
MCCL_VECTOR_BASE_FUNCTION_1OP(clear,*firstd^*firstd)
MCCL_VECTOR_BASE_FUNCTION_1OP(set,*firstd|~*firstd)

template<size_t bits, bool masked> 
inline void v_set(const v_ptr& v, bool b, block_tag<bits,masked>)
{
	if (b)
		v_set(v, block_tag<bits,masked>());
	else
		v_clear(v, block_tag<bits,masked>());
}



#define MCCL_VECTOR_BASE_FUNCTION_2OP(func,expr) \
template<size_t bits, bool masked> \
inline void v_ ## func (const v_ptr& dst, const cv_ptr& v1, block_tag<bits,masked>) \
{ \
	if (dst.columns == 0) \
		return; \
	if (dst.columns != v1.columns) \
		throw std::out_of_range("vectors do not have equal dimensions"); \
	size_t words = (dst.columns + bits-1)/bits - (masked?1:0); \
	auto firstd = make_block_ptr(dst.ptr, block_tag<bits,masked>()), lastd = firstd + words; \
	auto first1 = make_block_ptr(v1.ptr, block_tag<bits,masked>()); \
	for (; firstd != lastd; ++firstd, ++first1) \
		*firstd = expr ; \
	if (masked) \
	{ \
		auto lwm = lastwordmask(dst.columns, block_tag<bits,masked>()); \
		auto diff = lwm & ((expr) ^ *firstd); \
		*firstd ^= diff; \
	} \
}

MCCL_VECTOR_BASE_FUNCTION_2OP(copy,*first1)
MCCL_VECTOR_BASE_FUNCTION_2OP(copynot,~*first1)
MCCL_VECTOR_BASE_FUNCTION_2OP(and  ,(*firstd) & (*first1))
MCCL_VECTOR_BASE_FUNCTION_2OP(or   ,(*firstd) | (*first1))
MCCL_VECTOR_BASE_FUNCTION_2OP(xor  ,(*firstd) ^ (*first1))
MCCL_VECTOR_BASE_FUNCTION_2OP(nand ,~((*firstd) & (*first1)))
MCCL_VECTOR_BASE_FUNCTION_2OP(nor  ,~((*firstd) | (*first1)))
MCCL_VECTOR_BASE_FUNCTION_2OP(nxor ,~((*firstd) ^ (*first1)))
MCCL_VECTOR_BASE_FUNCTION_2OP(andin,(*firstd)  & (~*first1))
MCCL_VECTOR_BASE_FUNCTION_2OP(andni,(~*firstd) & (*first1))
MCCL_VECTOR_BASE_FUNCTION_2OP(orin ,(*firstd)  | (~*first1))
MCCL_VECTOR_BASE_FUNCTION_2OP(orni ,(~*firstd) | (*first1))



#define MCCL_VECTOR_BASE_FUNCTION_3OP(func,expr) \
template<size_t bits, bool masked> \
inline void v_ ## func (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2, block_tag<bits,masked>) \
{ \
	if (dst.columns == 0) \
		return; \
	if (dst.columns != v1.columns || dst.columns != v2.columns) \
		throw std::out_of_range("vectors do not have equal dimensions"); \
	size_t words = (dst.columns + bits-1)/bits - (masked?1:0); \
	auto firstd = make_block_ptr(dst.ptr, block_tag<bits,masked>()), lastd = firstd + words; \
	auto first1 = make_block_ptr(v1.ptr, block_tag<bits,masked>()); \
	auto first2 = make_block_ptr(v2.ptr, block_tag<bits,masked>()); \
	for (; firstd != lastd; ++firstd, ++first1, ++first2) \
		*firstd = expr ; \
	if (masked) \
	{ \
		auto lwm = lastwordmask(dst.columns, block_tag<bits,masked>()); \
		auto diff = lwm & ((expr) ^ *firstd); \
		*firstd ^= diff; \
	} \
}

MCCL_VECTOR_BASE_FUNCTION_3OP(and  ,(*first1) & (*first2))
MCCL_VECTOR_BASE_FUNCTION_3OP(or   ,(*first1) | (*first2))
MCCL_VECTOR_BASE_FUNCTION_3OP(xor  ,(*first1) ^ (*first2))
MCCL_VECTOR_BASE_FUNCTION_3OP(nand ,~((*first1) & (*first2)))
MCCL_VECTOR_BASE_FUNCTION_3OP(nor  ,~((*first1) | (*first2)))
MCCL_VECTOR_BASE_FUNCTION_3OP(nxor ,~((*first1) ^ (*first2)))
MCCL_VECTOR_BASE_FUNCTION_3OP(andin,(*first1)  & (~*first2))
MCCL_VECTOR_BASE_FUNCTION_3OP(andni,(~*first1) & (*first2))
MCCL_VECTOR_BASE_FUNCTION_3OP(orin ,(*first1)  | (~*first2))
MCCL_VECTOR_BASE_FUNCTION_3OP(orni ,(~*first1) | (*first2))

} // namespace detail

MCCL_END_NAMESPACE
