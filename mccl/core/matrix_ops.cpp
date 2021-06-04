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

static const uint64_t _lastwordmask[64] = {
	~uint64_t(0), (uint64_t(1)<<1)-1, (uint64_t(1)<<2)-1, (uint64_t(1)<<3)-1, (uint64_t(1)<<4)-1, (uint64_t(1)<<5)-1, (uint64_t(1)<<6)-1, (uint64_t(1)<<7)-1, (uint64_t(1)<<8)-1, (uint64_t(1)<<9)-1,
	(uint64_t(1)<<10)-1, (uint64_t(1)<<11)-1, (uint64_t(1)<<12)-1, (uint64_t(1)<<13)-1, (uint64_t(1)<<14)-1, (uint64_t(1)<<15)-1, (uint64_t(1)<<16)-1, (uint64_t(1)<<17)-1, (uint64_t(1)<<18)-1, (uint64_t(1)<<19)-1,
	(uint64_t(1)<<20)-1, (uint64_t(1)<<21)-1, (uint64_t(1)<<22)-1, (uint64_t(1)<<23)-1, (uint64_t(1)<<24)-1, (uint64_t(1)<<25)-1, (uint64_t(1)<<26)-1, (uint64_t(1)<<27)-1, (uint64_t(1)<<28)-1, (uint64_t(1)<<29)-1,
	(uint64_t(1)<<30)-1, (uint64_t(1)<<31)-1, (uint64_t(1)<<32)-1, (uint64_t(1)<<33)-1, (uint64_t(1)<<34)-1, (uint64_t(1)<<35)-1, (uint64_t(1)<<36)-1, (uint64_t(1)<<37)-1, (uint64_t(1)<<38)-1, (uint64_t(1)<<39)-1,
	(uint64_t(1)<<40)-1, (uint64_t(1)<<41)-1, (uint64_t(1)<<42)-1, (uint64_t(1)<<43)-1, (uint64_t(1)<<44)-1, (uint64_t(1)<<45)-1, (uint64_t(1)<<46)-1, (uint64_t(1)<<47)-1, (uint64_t(1)<<48)-1, (uint64_t(1)<<49)-1,
	(uint64_t(1)<<50)-1, (uint64_t(1)<<51)-1, (uint64_t(1)<<52)-1, (uint64_t(1)<<53)-1, (uint64_t(1)<<54)-1, (uint64_t(1)<<55)-1, (uint64_t(1)<<56)-1, (uint64_t(1)<<57)-1, (uint64_t(1)<<58)-1, (uint64_t(1)<<59)-1,
	(uint64_t(1)<<60)-1, (uint64_t(1)<<61)-1, (uint64_t(1)<<62)-1, (uint64_t(1)<<63)-1
};
static inline uint64_t lastwordmask(size_t cols)
{
	return (~uint64_t(0)) >> ((64-(cols%64))%64);
//	return _lastwordmask[cols%64];
}
static inline uint64_t firstwordmask(size_t cols)
{
	return uint64_t(0) - (uint64_t(1)<<(cols%64));
//	return _firstwordmask[cols%64];
}

// return the smallest 2^t >= n
template<typename Int>
inline Int next_pow2(Int n)
{
	static const unsigned bits = sizeof(Int)*8;
	--n;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	if (8 < bits) n |= n >> 8;
	if (16 < bits) n |= n >> 16;
	if (32 < bits) n |= n >> 32;
	return ++n;
}


void m_print(std::ostream& o, const cm_ptr& m, bool transpose)
{
	o << "[";
	if (!transpose)
	{
		for (size_t r = 0; r < m.rows; ++r)
		{
			o << (r==0 ? "[" : " [");
			for (size_t c = 0; c < m.columns; ++c)
				o << m_getbit(m,r,c);
			o << "]" << std::endl;
		}
	}
	else
	{
		for (size_t c = 0; c < m.columns; ++c)
		{
			o << "[";
			for (size_t r = 0; r < m.rows; ++r)
				o << m_getbit(m,r,c);
			o << "]" << std::endl;
		}
	}
	o << "]";
}

void v_print(std::ostream& o, const cv_ptr& v)
{
	o << "[";
	for (size_t c = 0; c < v.columns; ++c)
		o << v_getbit(v,c);
	o << "]";
}




bool m_isequal(const cm_ptr& m1, const cm_ptr& m2)
{
	if (m1.rows != m2.rows || m1.columns != m2.columns)
		return false;
	if (m1.rows == 0 || m1.columns == 0)
		return true;
	const size_t words = (m1.columns + 63) / 64;
	auto lwm = lastwordmask(m1.columns);
	for (size_t r = 0; r < m1.rows; ++r)
	{
		auto first1 = m1.data(r), first2 = m2.data(r), last1 = m1.data(r) + words - 1;
		for (; first1 != last1; ++first1, ++first2)
			if (*first1 != *first2)
				return false;
		if ((lwm & *first1) != (lwm & *first2))
			return false;
	}
	return true;
}
bool v_isequal(const cv_ptr& v1, const cv_ptr& v2)
{
	if (v1.columns != v2.columns)
		return false;
	if (v1.columns == 0)
		return true;
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




void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols, bool b)
{
	if (b)
		m_setcolumns(m,coloffset,cols);
	else
		m_clearcolumns(m,coloffset,cols);
}
void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols, bool b)
{
	if (b)
		v_setcolumns(v,coloffset,cols);
	else
		v_clearcolumns(v,coloffset,cols);
}

void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols)
{
	if (m.columns == 0 || m.rows == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = firstwordmask(coloffset);
	auto lwm = lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = lwm = fwm & lwm;
		firstword2 = firstword;
	}
	for (size_t r = 0; r < m.rows; ++r)
	{
		*(m.data(r)+firstword) |= fwm;
		auto first = m.data(r)+firstword2, last = m.data(r)+lastword;
		for (; first != last; ++first)
			*first |= ~uint64_t(0);
		*first |= lwm;
	}
}
void v_setcolumns(const v_ptr& v, size_t coloffset, size_t cols)
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
void m_flipcolumns(const m_ptr& m, size_t coloffset, size_t cols)
{
	if (m.columns == 0 || m.rows == 0 || cols == 0)
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
	for (size_t r = 0; r < m.rows; ++r)
	{
		*(m.data(r)+firstword) ^= fwm;
		auto first = m.data(r)+firstword2, last = m.data(r)+lastword;
		for (; first != last; ++first)
			*first ^= ~uint64_t(0);
		*first ^= lwm;
	}
}
void v_flipcolumns(const v_ptr& v, size_t coloffset, size_t cols)
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
void m_clearcolumns(const m_ptr& m, size_t coloffset, size_t cols)
{
	if (m.columns == 0 || m.rows == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = ~firstwordmask(coloffset);
	auto lwm = ~lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = lwm = fwm | lwm;
		firstword2 = firstword;
	}
	for (size_t r = 0; r < m.rows; ++r)
	{
		*(m.data(r)+firstword) &= fwm;
		auto first = m.data(r)+firstword2, last = m.data(r)+lastword;
		for (; first != last; ++first)
			*first = 0;
		*first &= lwm;
	}
}
void v_clearcolumns(const v_ptr& v, size_t coloffset, size_t cols)
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




template<void f(uint64_t*, uint64_t*, uint64_t)>
void m_1op_f(const m_ptr& dst)
{
	if (dst.rows == 0 || dst.columns == 0)
		return;
	const size_t words = (dst.columns - 1) / 64;
	auto lwm = lastwordmask(dst.columns);
	for (size_t r = 0; r < dst.rows; ++r)
		f(dst.data(r), dst.data(r) + words, lwm);
}
template<void f(uint64_t*, uint64_t*, uint64_t)>
void v_1op_f(const v_ptr& dst)
{
	if (dst.columns == 0)
		return;
	const size_t words = (dst.columns - 1) / 64;
	auto lwm = lastwordmask(dst.columns);
	f(dst.data(), dst.data()+words, lwm);
}

template<void f(uint64_t*, uint64_t*, const uint64_t*, uint64_t)>
void m_2op_f(const m_ptr& dst, const cm_ptr& src)
{
	if (dst.rows != src.rows || dst.columns != src.columns)
		throw std::out_of_range("m_2op_f: matrices do not have equal dimensions");
	if (src.rows == 0 || src.columns == 0)
		return;
	const size_t words = (src.columns - 1) / 64;
	auto lwm = lastwordmask(src.columns);
	for (size_t r = 0; r < src.rows; ++r)
		f(dst.data(r), dst.data(r) + words, src.data(r), lwm);
}
template<void f(uint64_t*, uint64_t*, const uint64_t*, uint64_t)>
void v_2op_f(const v_ptr& dst, const cv_ptr& src)
{
	if (dst.columns != src.columns)
	{
		std::cout << dst.columns << " " << src.columns << std::endl;
		throw std::out_of_range("v_2op_f: vectors do not have equal dimensions");
	}
	if (src.columns == 0)
		return;
	const size_t words = (src.columns - 1) / 64;
	auto lwm = lastwordmask(src.columns);
	f(dst.data(), dst.data()+words, src.data(), lwm);
}

template<void f(uint64_t*, uint64_t*, const uint64_t*, const uint64_t*, uint64_t)>
void m_3op_f(const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2)
{
	if (dst.rows != src1.rows || dst.rows != src2.rows || dst.columns != src1.columns || dst.columns != src2.columns)
		throw std::out_of_range("m_3op_f: matrices do not have equal dimensions");
	if (dst.rows == 0 || dst.columns == 0)
		return;
	const size_t words = (dst.columns - 1) / 64;
	auto lwm = lastwordmask(dst.columns);
	for (size_t r = 0; r < dst.rows; ++r)
		f(dst.data(r), dst.data(r) + words, src1.data(r), src2.data(r), lwm);
}
template<void f(uint64_t*, uint64_t*, const uint64_t*, const uint64_t*, uint64_t)>
void v_3op_f(const v_ptr& dst, const cv_ptr& src1, const cv_ptr& src2)
{
	if (dst.columns != src1.columns || dst.columns != src2.columns)
		throw std::out_of_range("v_3op_f: vectors do not have equal dimensions");
	if (dst.columns == 0)
		return;
	const size_t words = (dst.columns - 1) / 64;
	auto lwm = lastwordmask(dst.columns);
	f(dst.data(), dst.data()+words, src1.data(), src2.data(), lwm);
}




static inline void _f1_not(uint64_t* first, uint64_t* last, uint64_t lwm)
{
	for (; first != last; ++first)
		*first ^= ~uint64_t(0);
	*first ^= lwm; 
}
static inline void _f1_clear(uint64_t* first, uint64_t* last, uint64_t lwm)
{
	for (; first != last; ++first)
		*first = 0;
	*first &= ~lwm;
}
static inline void _f1_set(uint64_t* first, uint64_t* last, uint64_t lwm)
{
	for (; first != last; ++first)
		*first = ~uint64_t(0);
	*first |= lwm;
}

void m_set  (const m_ptr& m) { m_1op_f<_f1_set>(m); }
void m_clear(const m_ptr& m) { m_1op_f<_f1_clear>(m); }
void m_not  (const m_ptr& m) { m_1op_f<_f1_not>(m); }

void v_set  (const v_ptr& v) { v_1op_f<_f1_set>(v); }
void v_clear(const v_ptr& v) { v_1op_f<_f1_clear>(v); }
void v_not  (const v_ptr& v) { v_1op_f<_f1_not>(v); }

void m_set(const m_ptr& m, bool b)
{
	if (b)
		m_set(m);
	else
		m_clear(m);
}
void v_set(const v_ptr& v, bool b)
{
	if (b)
		v_set(v);
	else
		v_clear(v);
}



static inline void _f2_copy(uint64_t* first1, uint64_t* last1, const uint64_t* first2, uint64_t lwm)
{
	for (; first1 != last1; ++first1,++first2)
		*first1 = *first2;
	*first1 = ((*first2)&lwm) | ((*first1)&~lwm);
}
static inline void _f2_copynot(uint64_t* first1, uint64_t* last1, const uint64_t* first2, uint64_t lwm)
{
	for (; first1 != last1; ++first1,++first2)
		*first1 = ~(*first2);
	*first1 = ((~(*first2))&lwm) | ((*first1)&~lwm);
}
static inline void _f2_and(uint64_t* first1, uint64_t* last1, const uint64_t* first2, uint64_t lwm)
{
	for (; first1 != last1; ++first1,++first2)
		*first1 &= *first2;
	*first1 &= (*first2) | lwm;
}
static inline void _f2_or(uint64_t* first1, uint64_t* last1, const uint64_t* first2, uint64_t lwm)
{
	for (; first1 != last1; ++first1,++first2)
		*first1 |= *first2;
	*first1 |= (*first2) & ~lwm;
}
static inline void _f2_xor(uint64_t* first1, uint64_t* last1, const uint64_t* first2, uint64_t lwm)
{
	for (; first1 != last1; ++first1,++first2)
		*first1 ^= *first2;
	*first1 ^= (*first2) & ~lwm;
}

void m_copy   (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_copy>   (dst, src); }
void m_copynot(const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_copynot>(dst, src); }
void m_and    (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_and>    (dst, src); }
void m_or     (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_or>     (dst, src); }
void m_xor    (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_xor>    (dst, src); }

void v_copy   (const v_ptr& dst, const cv_ptr& src) { v_2op_f<_f2_copy>   (dst, src); }
void v_copynot(const v_ptr& dst, const cv_ptr& src) { v_2op_f<_f2_copynot>(dst, src); }
void v_and    (const v_ptr& dst, const cv_ptr& src) { v_2op_f<_f2_and>    (dst, src); }
void v_or     (const v_ptr& dst, const cv_ptr& src) { v_2op_f<_f2_or>     (dst, src); }
void v_xor    (const v_ptr& dst, const cv_ptr& src) { v_2op_f<_f2_xor>    (dst, src); }




static inline void _f3_and(uint64_t* dstfirst, uint64_t* dstlast, const uint64_t* srcfirst1, const uint64_t* srcfirst2, uint64_t lwm)
{
	for (; dstfirst != dstlast; ++dstfirst,++srcfirst1,++srcfirst2)
		*dstfirst = (*srcfirst1) & (*srcfirst2);
	*dstfirst = ((*dstfirst)&~lwm) | ((  (*srcfirst1)&(*srcfirst2)   )&lwm);
}
static inline void _f3_or(uint64_t* dstfirst, uint64_t* dstlast, const uint64_t* srcfirst1, const uint64_t* srcfirst2, uint64_t lwm)
{
	for (; dstfirst != dstlast; ++dstfirst,++srcfirst1,++srcfirst2)
		*dstfirst = (*srcfirst1) | (*srcfirst2);
	*dstfirst = ((*dstfirst)&~lwm) | ((  (*srcfirst1)|(*srcfirst2)   )&lwm);
}
static inline void _f3_xor(uint64_t* dstfirst, uint64_t* dstlast, const uint64_t* srcfirst1, const uint64_t* srcfirst2, uint64_t lwm)
{
	for (; dstfirst != dstlast; ++dstfirst,++srcfirst1,++srcfirst2)
		*dstfirst = (*srcfirst1) ^ (*srcfirst2);
	*dstfirst = ((*dstfirst)&~lwm) | ((  (*srcfirst1)^(*srcfirst2)   )&lwm);
}

void m_and    (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_and>(dst, src1, src2); }
void m_or     (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_or> (dst, src1, src2); }
void m_xor    (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_xor>(dst, src1, src2); }

void v_and    (const v_ptr& dst, const cv_ptr& src1, const cv_ptr& src2) { v_3op_f<_f3_and>(dst, src1, src2); }
void v_or     (const v_ptr& dst, const cv_ptr& src1, const cv_ptr& src2) { v_3op_f<_f3_or> (dst, src1, src2); }
void v_xor    (const v_ptr& dst, const cv_ptr& src1, const cv_ptr& src2) { v_3op_f<_f3_xor>(dst, src1, src2); }

void v_swap(const v_ptr& v1, const v_ptr& v2)
{
	if (v1.columns != v2.columns)
		throw std::out_of_range("v_swap: vectors do not have equal dimensions");
	if (v1.columns == 0)
		return;
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

size_t m_hw(const cm_ptr& m)
{
	if (m.columns == 0 || m.rows == 0)
		return 0;
	size_t words = (m.columns + 63)/64;
	uint64_t lwm = lastwordmask(m.columns);
	size_t hw = 0;
	for (size_t r = 0; r < m.rows; ++r)
	{
		auto first1 = m.data(r), last1 = m.data(r) + words - 1;
		for (; first1 != last1; ++first1)
			hw += hammingweight(*first1);
		hw += hammingweight((*first1) & lwm);
	}
	return hw;
}
size_t v_hw(const cv_ptr& v)
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

template<size_t bits = 64>
void block_transpose(uint64_t* dst, size_t dststride, const uint64_t* src, size_t srcstride)
{
	static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
	static_assert(64 >= bits, "bits must not exceed uint64_t bitsize");

	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[bits];

	// first loop iteration, load src store in tmp
//#pragma unroll
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k, src+=srcstride, src2+=srcstride)
	{
		// j = (bits/2)
		uint64_t a = *src, b = *src2;
		uint64_t t = ((a>>(bits/2)) ^ b) & m;
		tmp[k] = a ^ (t << (bits/2));
		tmp[k+(bits/2)] = b ^ t;
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
//#pragma unroll
		for (unsigned int l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
//#pragma unroll
	for (unsigned int k=0;  k<bits;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		*dst = tmp[k] ^ (t<<1); dst+=dststride;
		*dst = tmp[k+1] ^ t; dst+=dststride;
	}
}

template<size_t bits = 64>
inline void block_transpose2(uint64_t* dst, size_t dststride, const uint64_t* src, size_t srcstride)
{
	static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
	static_assert(64 >= bits, "bits must not exceed uint64_t bitsize");

	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[2*bits];

	// first loop iteration, load src store in tmp
//#pragma unroll
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k, src+=srcstride, src2+=srcstride)
	{
		// j = (bits/2)
		uint64_t a1 = *src, b1 = *src2;
		uint64_t t1 = ((a1>>(bits/2)) ^ b1) & m;
		tmp[k] = a1 ^ (t1 << (bits/2));
		tmp[k+(bits/2)] = b1 ^ t1;
		uint64_t a2 = *(src+1), b2 = *(src2+1);
		uint64_t t2 = ((a2>>(bits/2)) ^ b2) & m;
		tmp[k+bits] = a2 ^ (t2 << (bits/2));
		tmp[k+(bits/2)+bits] = b2 ^ t2;
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
//#pragma unroll
		for (unsigned int l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			uint64_t t2 = ((tmp[k+bits]>>j) ^ tmp[k+j+bits]) & m;
			tmp[k+bits] ^= t2<<j;
			tmp[k+j+bits] ^= t2;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
//#pragma unroll
	for (unsigned int k=0;  k<2*bits;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		*dst = tmp[k] ^ (t<<1); dst+=dststride;
		*dst = tmp[k+1] ^ t; dst+=dststride;
	}
}

template<size_t bits = 64>
inline void block_transpose(uint64_t* dst, size_t dststride, size_t dstrows, const uint64_t* src, size_t srcstride, size_t srcrows)
{
	static_assert(bits >= 4, "bits >= 4");
	static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
	static_assert(sizeof(uint64_t)*8 >= bits, "bits must not exceed uint64_t bitsize");
	assert(dstrows <= bits);
	assert(srcrows <= bits);
	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[bits+2]; // <= add 2 to avoid incorrect out-of-bounds warning
	// first loop iteration, load src store in tmp
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k)
	{
		if (k < srcrows)
		{
			uint64_t a = *src, b = 0;
			src += srcstride;
			if ((k+(bits/2)) < srcrows)
			{
				b = *src2;
				src2 += srcstride;
			}
			uint64_t t = (b ^ (a >> (bits/2))) & m;
			tmp[k] = a ^ (t << (bits/2));
			tmp[k+(bits/2)] = b ^ t;
		}
		else
		{
			tmp[k] = 0;
			tmp[k+(bits/2)] = 0;
		}
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
		for (unsigned int l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
	unsigned int k=0;
	for (;  k+1 < dstrows;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		*dst = tmp[k] ^ (t<<1); dst+=dststride;
		*dst = tmp[k+1] ^ t; dst+=dststride;
	}
	// note both k and bits are even and k < dstrows <= bits
	// so k+1 < bits as well, nevertheless compilers may warn
	if (k < dstrows)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		*dst = tmp[k] ^ (t<<1);
	}
}



template<typename uint64_t>
inline void block_transpose(uint64_t* dst, size_t dststride, size_t dstrows, const uint64_t* src, size_t srcstride, size_t srcrows, size_t bits)
{
	assert(0 == (bits&(bits-1))); // bits must be power of 2
	assert(64 >= bits); // bits must not exceed uint64_t bitsize
	assert(dstrows <= bits);
	assert(srcrows <= bits);
	if (bits < 4)
		bits = 4;
	if (bits > 64) throw std::out_of_range("block_transpose: bits > 64");
	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[8*sizeof(uint64_t)];
	// first loop iteration, load src store in tmp
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k)
	{
		if (k < srcrows)
		{
			uint64_t a = *src, b = 0;
			src += srcstride;
			if ((k+(bits/2)) < srcrows)
			{
				b = *src2;
				src2 += srcstride;
			}
			uint64_t t = (b ^ (a >> (bits/2))) & m;
			tmp[k] = a ^ (t << (bits/2));
			tmp[k+(bits/2)] = b ^ t;
		}
		else
		{
			tmp[k] = 0;
			tmp[k+(bits/2)] = 0;
		}
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
		for (unsigned l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
	unsigned int k=0;
	for (;  k+1 < dstrows;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		*dst = tmp[k] ^ (t<<1); dst+=dststride;
		*dst = tmp[k+1] ^ t; dst+=dststride;
	}
	if (k < dstrows)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		*dst = tmp[k] ^ (t<<1);
	}
}



void m_transpose(const m_ptr& dst, const cm_ptr& src)
{
	static const size_t bits = 64;
	if (dst.columns != src.rows || dst.rows != src.columns)
		throw std::out_of_range("m_transpose: matrix dimensions do not match");
	// process batch of bits rows
	size_t r = 0;
	for (; r+bits <= src.rows; r += bits)
	{
		// process block of bits columns
		size_t c = 0;
		for (; c+2*bits <= src.columns; c += 2*bits)
		{
			block_transpose2(dst.data(c,r), dst.stride, src.data(r,c), src.stride);
		}
		if (c+bits <= src.columns)
		{
			block_transpose(dst.data(c,r), dst.stride, src.data(r,c), src.stride);
			c += bits;
		}
		// process block of partial C columns
		if (c < src.columns)
		{
			block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, bits);
		}
	}
	// process last rows
	if (r < src.rows)
	{
		size_t c = 0;
		for (; c+bits <= src.columns; c += bits)
		{
			block_transpose(dst.data(c,r), dst.stride, bits, src.data(r,c), src.stride, (src.rows % bits));
		}
		// process final bits x C submatrix
		if (c < src.columns)
		{
			size_t partialbits = next_pow2<uint32_t>(std::max(src.columns % bits, src.rows % bits));
			if (partialbits == bits)
				block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, (src.rows % bits));
			else
				block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, (src.rows % bits), partialbits);
		}
	}
}

}

MCCL_END_NAMESPACE
