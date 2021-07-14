#include <mccl/config/config.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <nmmintrin.h>
#include <cassert>

MCCL_BEGIN_NAMESPACE

namespace detail {

inline size_t hammingweight(uint64_t n)
{
	return _mm_popcnt_u64(n);
}
inline size_t hammingweight(uint32_t n)
{
	return _mm_popcnt_u32(n);
}
template<size_t bits>
inline size_t hammingweight(const block_t<bits>& v)
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

const uint64_t _lastwordmask[64] = {
	~uint64_t(0), (uint64_t(1)<<1)-1, (uint64_t(1)<<2)-1, (uint64_t(1)<<3)-1, (uint64_t(1)<<4)-1, (uint64_t(1)<<5)-1, (uint64_t(1)<<6)-1, (uint64_t(1)<<7)-1, (uint64_t(1)<<8)-1, (uint64_t(1)<<9)-1,
	(uint64_t(1)<<10)-1, (uint64_t(1)<<11)-1, (uint64_t(1)<<12)-1, (uint64_t(1)<<13)-1, (uint64_t(1)<<14)-1, (uint64_t(1)<<15)-1, (uint64_t(1)<<16)-1, (uint64_t(1)<<17)-1, (uint64_t(1)<<18)-1, (uint64_t(1)<<19)-1,
	(uint64_t(1)<<20)-1, (uint64_t(1)<<21)-1, (uint64_t(1)<<22)-1, (uint64_t(1)<<23)-1, (uint64_t(1)<<24)-1, (uint64_t(1)<<25)-1, (uint64_t(1)<<26)-1, (uint64_t(1)<<27)-1, (uint64_t(1)<<28)-1, (uint64_t(1)<<29)-1,
	(uint64_t(1)<<30)-1, (uint64_t(1)<<31)-1, (uint64_t(1)<<32)-1, (uint64_t(1)<<33)-1, (uint64_t(1)<<34)-1, (uint64_t(1)<<35)-1, (uint64_t(1)<<36)-1, (uint64_t(1)<<37)-1, (uint64_t(1)<<38)-1, (uint64_t(1)<<39)-1,
	(uint64_t(1)<<40)-1, (uint64_t(1)<<41)-1, (uint64_t(1)<<42)-1, (uint64_t(1)<<43)-1, (uint64_t(1)<<44)-1, (uint64_t(1)<<45)-1, (uint64_t(1)<<46)-1, (uint64_t(1)<<47)-1, (uint64_t(1)<<48)-1, (uint64_t(1)<<49)-1,
	(uint64_t(1)<<50)-1, (uint64_t(1)<<51)-1, (uint64_t(1)<<52)-1, (uint64_t(1)<<53)-1, (uint64_t(1)<<54)-1, (uint64_t(1)<<55)-1, (uint64_t(1)<<56)-1, (uint64_t(1)<<57)-1, (uint64_t(1)<<58)-1, (uint64_t(1)<<59)-1,
	(uint64_t(1)<<60)-1, (uint64_t(1)<<61)-1, (uint64_t(1)<<62)-1, (uint64_t(1)<<63)-1
};
inline uint64_t lastwordmask(size_t cols)
{
//	return (~uint64_t(0)) >> ((64-(cols%64))%64);
	return _lastwordmask[cols%64];
}
inline uint64_t firstwordmask(size_t cols)
{
	return uint64_t(0) - (uint64_t(1)<<(cols%64));
//	return _firstwordmask[cols%64];
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







template<void f(uint64_t*, uint64_t*, uint64_t)>
inline void v_1op_f(const v_ptr& dst)
{
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (dst.columns == 0)
		return;
#endif
	const size_t words = (dst.columns - 1) / 64;
	auto lwm = lastwordmask(dst.columns);
	f(dst.data(), dst.data()+words, lwm);
}

template<void f(uint64_t*, uint64_t*, const uint64_t*, uint64_t)>
inline void v_2op_f(const v_ptr& dst, const cv_ptr& src)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (dst.columns != src.columns)
		throw std::out_of_range("v_2op_f: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (src.columns == 0)
		return;
#endif
	const size_t words = (src.columns - 1) / 64;
	auto lwm = lastwordmask(src.columns);
	f(dst.data(), dst.data()+words, src.data(), lwm);
}

template<void f(uint64_t*, uint64_t*, const uint64_t*, const uint64_t*, uint64_t)>
inline void v_3op_f(const v_ptr& dst, const cv_ptr& src1, const cv_ptr& src2)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (dst.columns != src1.columns || dst.columns != src2.columns)
		throw std::out_of_range("v_3op_f: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (dst.columns == 0)
		return;
#endif
	const size_t words = (dst.columns - 1) / 64;
	auto lwm = lastwordmask(dst.columns);
	f(dst.data(), dst.data()+words, src1.data(), src2.data(), lwm);
}



template<size_t bits, void f(block_t<bits>*, block_t<bits>*)>
inline void v_1op_f(const v_ptr& dst, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (dst.columns == 0)
		return;
#endif
	const size_t words = dst.columns/64;
	f(reinterpret_cast<block_t<bits>*>(dst.data()), reinterpret_cast<block_t<bits>*>(dst.data()+words));
}

template<size_t bits, void f(block_t<bits>*, block_t<bits>*, const block_t<bits>*)>
inline void v_2op_f(const v_ptr& dst, const cv_ptr& src, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (dst.columns != src.columns)
		throw std::out_of_range("v_2op_f: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (src.columns == 0)
		return;
#endif
	const size_t words = src.columns/64;
	f(reinterpret_cast<block_t<bits>*>(dst.data()), reinterpret_cast<block_t<bits>*>(dst.data()+words),
	  reinterpret_cast<const block_t<bits>*>(src.data()));
}

template<size_t bits, void f(block_t<bits>*, block_t<bits>*, const block_t<bits>*, const block_t<bits>*)>
inline void v_3op_f(const v_ptr& dst, const cv_ptr& src1, const cv_ptr& src2, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (dst.columns != src1.columns || dst.columns != src2.columns)
		throw std::out_of_range("v_3op_f: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (dst.columns == 0)
		return;
#endif
	const size_t words = dst.columns / 64;
	f(reinterpret_cast<block_t<bits>*>(dst.data()), reinterpret_cast<block_t<bits>*>(dst.data()+words), 
	  reinterpret_cast<const block_t<bits>*>(src1.data()), reinterpret_cast<const block_t<bits>*>(src2.data()));
}



#define MCCL_MATRIX_BASE_FUNCTION_1OP(func,expr) \
inline void _f1_ ## func (uint64_t* first1, uint64_t* last1, uint64_t lwm) \
{ \
	for (; first1 != last1; ++first1) \
		*first1 = expr ; \
	*first1 = (( expr ) & lwm) | ((*first1) & ~lwm); \
} \
template<size_t bits> \
inline void _f1_ ## func (block_t<bits>* first1, block_t<bits>* last1) \
{ \
	for (; first1 != last1; ++first1) \
		*first1 = expr ; \
} \
inline void v_ ## func (const v_ptr& v) { v_1op_f<_f1_ ## func >(v); } \
template<size_t bits> \
inline void v_ ## func (const v_ptr& v, aligned_tag<bits>) { v_1op_f<bits,_f1_ ## func >(v, aligned_tag<bits>()); }



#define MCCL_MATRIX_BASE_FUNCTION_2OP(func,expr) \
inline void _f2_ ## func (uint64_t* first1, uint64_t* last1, const uint64_t* first2, uint64_t lwm) \
{ \
	for (; first1 != last1; ++first1,++first2) \
		*first1 = expr ; \
	*first1 = (( expr ) & lwm) | ((*first1) & ~lwm); \
} \
template<size_t bits> \
inline void _f2_ ## func (block_t<bits>* first1, block_t<bits>* last1, const block_t<bits>* first2) \
{ \
	for (; first1 != last1; ++first1,++first2) \
		*first1 = expr ; \
} \
inline void v_ ## func (const v_ptr& v1, const cv_ptr& v2) { v_2op_f<_f2_ ## func >(v1, v2); } \
template<size_t bits> \
inline void v_ ## func (const v_ptr& v1, const cv_ptr& v2, aligned_tag<bits>) { v_2op_f<bits,_f2_ ## func >(v1, v2, aligned_tag<bits>()); }



#define MCCL_MATRIX_BASE_FUNCTION_3OP(func,expr) \
inline void _f3_ ## func (uint64_t* dstfirst, uint64_t* dstlast, const uint64_t* first1, const uint64_t* first2, uint64_t lwm) \
{ \
	for (; dstfirst != dstlast; ++dstfirst,++first1,++first2) \
		*dstfirst = expr ; \
	*dstfirst = ((*dstfirst)&~lwm) | ((  expr   )&lwm); \
} \
template<size_t bits> \
inline void _f3_ ## func (block_t<bits>* dstfirst, block_t<bits>* dstlast, const block_t<bits>* first1, const block_t<bits>* first2) \
{ \
	for (; dstfirst != dstlast; ++dstfirst,++first1,++first2) \
		*dstfirst = expr ; \
} \
inline void v_ ## func (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2 ) { v_3op_f<_f3_ ## func >(dst, v1, v2); } \
template<size_t bits> \
inline void v_ ## func (const v_ptr& dst, const cv_ptr& v1, const cv_ptr& v2, aligned_tag<bits>) { v_3op_f<bits,_f3_ ## func >(dst, v1, v2, aligned_tag<bits>()); }



MCCL_MATRIX_BASE_FUNCTION_1OP(not,~*first1)
MCCL_MATRIX_BASE_FUNCTION_1OP(clear,*first1 ^ *first1)
MCCL_MATRIX_BASE_FUNCTION_1OP(set,*first1 | ~*first1)

MCCL_MATRIX_BASE_FUNCTION_2OP(copy,*first2)
MCCL_MATRIX_BASE_FUNCTION_2OP(copynot,~*first2)
MCCL_MATRIX_BASE_FUNCTION_2OP(and  ,(*first1) & (*first2))
MCCL_MATRIX_BASE_FUNCTION_2OP(or   ,(*first1) | (*first2))
MCCL_MATRIX_BASE_FUNCTION_2OP(xor  ,(*first1) ^ (*first2))
MCCL_MATRIX_BASE_FUNCTION_2OP(nand ,~((*first1) & (*first2)))
MCCL_MATRIX_BASE_FUNCTION_2OP(nor  ,~((*first1) | (*first2)))
MCCL_MATRIX_BASE_FUNCTION_2OP(nxor ,~((*first1) ^ (*first2)))
MCCL_MATRIX_BASE_FUNCTION_2OP(andin,(*first1)  & (~*first2))
MCCL_MATRIX_BASE_FUNCTION_2OP(andni,(~*first1) & (*first2))
MCCL_MATRIX_BASE_FUNCTION_2OP(orin ,(*first1)  | (~*first2))
MCCL_MATRIX_BASE_FUNCTION_2OP(orni ,(~*first1) | (*first2))

MCCL_MATRIX_BASE_FUNCTION_3OP(and  ,(*first1) & (*first2))
MCCL_MATRIX_BASE_FUNCTION_3OP(or   ,(*first1) | (*first2))
MCCL_MATRIX_BASE_FUNCTION_3OP(xor  ,(*first1) ^ (*first2))
MCCL_MATRIX_BASE_FUNCTION_3OP(nand ,~((*first1) & (*first2)))
MCCL_MATRIX_BASE_FUNCTION_3OP(nor  ,~((*first1) | (*first2)))
MCCL_MATRIX_BASE_FUNCTION_3OP(nxor ,~((*first1) ^ (*first2)))
MCCL_MATRIX_BASE_FUNCTION_3OP(andin,(*first1)  & (~*first2))
MCCL_MATRIX_BASE_FUNCTION_3OP(andni,(~*first1) & (*first2))
MCCL_MATRIX_BASE_FUNCTION_3OP(orin ,(*first1)  | (~*first2))
MCCL_MATRIX_BASE_FUNCTION_3OP(orni ,(~*first1) | (*first2))



                      inline void v_set  (const v_ptr& v, bool b)                    { if (b) v_set(v); else v_clear(v); }
template<size_t bits> inline void v_set  (const v_ptr& v, bool b, aligned_tag<bits>) { if (b) v_set(v, aligned_tag<bits>()); else v_clear(v, aligned_tag<bits>()); }



inline bool v_isequal(const cv_ptr& v1, const cv_ptr& v2)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		return false;
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return true;
#endif
	const size_t words = (v1.columns + 63) / 64;
	auto first1 = v1.data(), first2 = v2.data(), last1 = v1.data() + words - 1;
	for (; first1 != last1; ++first1, ++first2)
		if (*first1 != *first2)
			return false;
	auto lwm = lastwordmask(v1.columns);
	if ((lwm & *first1) != (lwm & *first2))
		return false;
	return true;
}
template<size_t bits>
inline bool v_isequal(const cv_ptr& v1, const cv_ptr& v2, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		return false;
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return true;
#endif
	const size_t words = v1.columns / 64;
	auto first1 = reinterpret_cast<const block_t<bits>*>(v1.data()), last1 = reinterpret_cast<const block_t<bits>*>(v1.data() + words);
	auto first2 = reinterpret_cast<const block_t<bits>*>(v2.data());
	block_t<bits> zero; zero ^= zero;
	block_t<bits> acc(zero);
	for (; first1 != last1; ++first1, ++first2)
		acc |= *first1 ^ *first2;
	return acc == zero;
}



inline void v_swap(const v_ptr& v1, const v_ptr& v2)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return;
#endif
	const size_t words = (v1.columns - 1) / 64;
	auto lwm = lastwordmask(v1.columns);
	auto first1 = v1.data(), last1 = v1.data() + words;
	auto first2 = v2.data();
	for (; first1 != last1; ++first1,++first2)
		std::swap(*first1, *first2);
	uint64_t tmp = ((*first1) ^ (*first2)) & lwm;
	*first1 ^= tmp;
	*first2 ^= tmp;
}
template<size_t bits>
inline void v_swap(const v_ptr& v1, const v_ptr& v2, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return;
#endif
	const size_t words = v1.columns / 64;
	auto first1 = reinterpret_cast<block_t<bits>*>(v1.data()), last1 = reinterpret_cast<block_t<bits>*>(v1.data() + words);
	auto first2 = reinterpret_cast<block_t<bits>*>(v2.data());
	for (; first1 != last1; ++first1,++first2)
		std::swap(*first1, *first2);
}



inline size_t v_hw(const cv_ptr& v)
{
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v.columns == 0)
		return 0;
#endif
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
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return 0;
#endif
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
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return 0;
#endif
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
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return 0;
#endif
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



template<size_t bits>
inline size_t v_hw(const cv_ptr& v, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v.columns == 0)
		return 0;
#endif
	size_t words = v.columns/64;
	size_t hw = 0;
	auto first1 = reinterpret_cast<const block_t<bits>*>(v.data()), last1 = reinterpret_cast<const block_t<bits>*>(v.data() + words);
	for (; first1 != last1; ++first1)
		hw += hammingweight(*first1);
	return hw;
}
template<size_t bits>
inline size_t v_hw_and(const cv_ptr& v1, const cv_ptr& v2, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return 0;
#endif
	const size_t words = v1.columns / 64;
	auto first1 = reinterpret_cast<const block_t<bits>*>(v1.data()), last1 = reinterpret_cast<const block_t<bits>*>(v1.data() + words);
	auto first2 = reinterpret_cast<const block_t<bits>*>(v2.data());
	size_t hw = 0;
	for (; first1 != last1; ++first1,++first2)
		hw += hammingweight(*first1 & *first2);
	return hw;
}
template<size_t bits>
inline size_t v_hw_or(const cv_ptr& v1, const cv_ptr& v2, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return 0;
#endif
	const size_t words = v1.columns / 64;
	auto first1 = reinterpret_cast<const block_t<bits>*>(v1.data()), last1 = reinterpret_cast<const block_t<bits>*>(v1.data() + words);
	auto first2 = reinterpret_cast<const block_t<bits>*>(v2.data());
	size_t hw = 0;
	for (; first1 != last1; ++first1,++first2)
		hw += hammingweight(*first1 | *first2);
	return hw;
}
template<size_t bits>
inline size_t v_hw_xor(const cv_ptr& v1, const cv_ptr& v2, aligned_tag<bits>)
{
#ifndef MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
#endif
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v1.columns == 0)
		return 0;
#endif
	const size_t words = v1.columns / 64;
	auto first1 = reinterpret_cast<const block_t<bits>*>(v1.data()), last1 = reinterpret_cast<const block_t<bits>*>(v1.data() + words);
	auto first2 = reinterpret_cast<const block_t<bits>*>(v2.data());
	size_t hw = 0;
	for (; first1 != last1; ++first1,++first2)
		hw += hammingweight(*first1 ^ *first2);
	return hw;
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
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v.columns == 0 || cols == 0)
		return;
#endif
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
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v.columns == 0 || cols == 0)
		return;
#endif
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
#ifndef MCCL_VECTOR_ASSUME_NONEMPTY
	if (v.columns == 0 || cols == 0)
		return;
#endif
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



} // namespace detail

MCCL_END_NAMESPACE
