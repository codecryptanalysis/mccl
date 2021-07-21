#include <mccl/config/config.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <nmmintrin.h>
#include <cassert>

MCCL_BEGIN_NAMESPACE

namespace detail {

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
void m_swapcolumns(const m_ptr& m, size_t c1, size_t c2)
{
	size_t w1 = c1/64, w2 = c2/64, r2 = (c1-c2)%64;
	uint64_t* w1ptr = m.data(0)+w1;
	uint64_t w1mask = uint64_t(1) << (c1%64);
	if (w1 == w2)
	{
		// same word column swap
		for (size_t k = 0; k < m.rows; ++k,w1ptr+=m.stride)
		{
			uint64_t x1 = *w1ptr;
			uint64_t tmp = (x1^rotate_left(x1,r2)) & w1mask;
			*w1ptr = x1 ^ tmp ^ rotate_right(tmp,r2);
		}
	}
	else
	{
		uint64_t* w2ptr = m.data(0)+w2;
		// two word column swap
		for (size_t k = 0; k < m.rows; ++k,w1ptr+=m.stride,w2ptr+=m.stride)
		{
			uint64_t x1 = *w1ptr, x2 = *w2ptr;
			uint64_t tmp = (x1^rotate_left(x2,r2)) & w1mask;
			*w1ptr = x1 ^ tmp;
			*w2ptr = x2 ^ rotate_right(tmp,r2);
		}
	}
}

void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols, bool b)
{
	if (b)
		m_setcolumns(m,coloffset,cols);
	else
		m_clearcolumns(m,coloffset,cols);
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

void m_set  (const m_ptr& m) { m_1op_f<_f1_set>(m); }
void m_clear(const m_ptr& m) { m_1op_f<_f1_clear>(m); }
void m_not  (const m_ptr& m) { m_1op_f<_f1_not>(m); }
void m_set  (const m_ptr& m, bool b) { if (b) m_set(m); else m_clear(m); }

void m_copy   (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_copy>   (dst, src); }
void m_copynot(const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_copynot>(dst, src); }
void m_and    (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_and>    (dst, src); }
void m_or     (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_or>     (dst, src); }
void m_xor    (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_xor>    (dst, src); }
void m_nand   (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_nand>   (dst, src); }
void m_nor    (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_nor>    (dst, src); }
void m_nxor   (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_nxor>   (dst, src); }
void m_andin  (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_andin>  (dst, src); }
void m_andni  (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_andni>  (dst, src); }
void m_orin   (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_orin>   (dst, src); }
void m_orni   (const m_ptr& dst, const cm_ptr& src) { m_2op_f<_f2_orni>   (dst, src); }
void m_and    (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_and>(dst, src1, src2); }
void m_or     (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_or> (dst, src1, src2); }
void m_xor    (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_xor>(dst, src1, src2); }
void m_nand   (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_nand>(dst, src1, src2); }
void m_nor    (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_nor> (dst, src1, src2); }
void m_nxor   (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_nxor>(dst, src1, src2); }
void m_andin  (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_andin>(dst, src1, src2); }
void m_andni  (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_andni>(dst, src1, src2); }
void m_orin   (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_orin> (dst, src1, src2); }
void m_orni   (const m_ptr& dst, const cm_ptr& src1, const cm_ptr& src2) { m_3op_f<_f3_orni> (dst, src1, src2); }

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
	if (dst.columns == 0 || dst.rows == 0)
		return;
	if (dst.columns != src.rows || dst.rows != src.columns)
	{
		std::cout << dst.columns << " " << src.rows << "  " << dst.rows << " " << src.columns << std::endl;
		throw std::out_of_range("m_transpose: matrix dimensions do not match");
	}
	if (dst.ptr == src.ptr)
	{
		std::cout << dst.ptr << " " << src.ptr << std::endl;
		throw std::runtime_error("m_transpose: src and dst are equal! cannot transpose inplace");
	}
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
