#ifndef MCCL_CORE_MATRIX_PERMUTE_HPP
#define MCCL_CORE_MATRIX_PERMUTE_HPP

#include <mccl/core/matrix.hpp>

#include <iostream>
#include <functional>
#include <random>
#include <numeric>
#include <utility>

MCCL_BEGIN_NAMESPACE

/*
   Class to maintain H^T in desired ISD form:
   The performance optimized ISD form for H is:
      H = ( 0  | H1 )
          ( AI | H0 )
      where AI is the antidiagonal identity matrix
      as a result of performing reverse row reduction from bottom to top
   This translates to H^T to
      H^T = ( 0    | AI   )
            ( H1^T | H0^T )
      This form ensures H1^T columns are before H0^T columns
      and thus is flexible with padding H1^T column words (64-bit or s-bit SIMD) with additional H0^T columns
*/
class HT_ISD_form_t
{
public:
    HT_ISD_form_t() {}
    HT_ISD_form_t(const mat_view& _HT) { reset(_HT); }
    void reset(const mat_view& _HT)
    {
    	HT.reset(_HT.ptr);
    	perm.resize(HT.rows());
    	std::iota(perm.begin(), perm.end(), 0);
    	row_permute();
    	echelonize(0, HT.rows(), HT.columns());
    };

    const std::vector<uint32_t>& permutation() const { return perm; }
    
    vec_view_it operator[](size_t r) const { return HT[r]; }
    vec_view_it operator()(size_t r) const { return HT[r]; }

    // permute rows: swap each row in [b1:e1) with a uniformly random chosen row from [b2:e2)
    void row_permute(size_t b1, size_t e1, size_t b2, size_t e2)
    {
    	if (b1 >= e1 || b2 >= e2)
    		return;
	if (e1 > HT.rows())
		e1 = HT.rows();
	if (e2 > HT.rows())
		e2 = HT.rows();
    	size_t n2 = e2 - b2;
    	for (size_t i = b1; i < e1; ++i)
    	{
    		size_t j = b2 + (rndgen() % n2);
    		if (j == i)
    			continue;
    		std::swap(perm[i], perm[j]);
    		HT[i].swap(HT[j]);
    	}
    }
    template<size_t bits>
    void row_permute(size_t b1, size_t e1, size_t b2, size_t e2, aligned_tag<bits>)
    {
    	if (b1 >= e1 || b2 >= e2)
    		return;
	if (e1 > HT.rows())
		e1 = HT.rows();
	if (e2 > HT.rows())
		e2 = HT.rows();
    	size_t n2 = e2 - b2;
    	for (size_t i = b1; i < e1; ++i)
    	{
    		size_t j = b2 + (rndgen() % n2);
    		if (j == i)
    			continue;
    		std::swap(perm[i], perm[j]);
    		HT[i].swap(HT[j], aligned_tag<bits>());
    	}
    }

    void row_permute(size_t b = 0, size_t e = ~uint64_t(0))
    {
	row_permute(b,e,b,e);
    }

    // full *column* reduction of matrix H^T with reverse column ordering
    //    note: column reduction normally should only do full column operations (swap, xor)
    //          in this case row swaps are cheaper and also allowed when recorded in permutation perm
    //          our implementation performs the xor of one column onto many columns at the same time by xoring rows
    //
    // this corresponds to full row reduction of matrix H with reverse row ordering
    //    note: row reduction normally should only do full row operations (swap, xor)
    //          in this case column swaps are allowed when recorded in permutation perm
    //
    // since we do reverse column ordering: pivot_end points to startpivot+1
    size_t echelonize(size_t row_begin, size_t row_end, size_t pivot_end)
    {
    	for (size_t r = row_begin; r < row_end; ++r)
    	{
		--pivot_end;
		// find pivot for row r column pivot_start
		// normally we swap columns, but row swaps are also allowed for ISD
		size_t r2 = r;
		for (; r2 < HT.rows() && HT(r2,pivot_end)==false; ++r2)
			;
		if (r2 == HT.rows())
		{
			++pivot_end;
			continue;
		}
		if (r2 != r)
		{
			std::swap(perm[r], perm[r2]);
			HT[r2].swap(HT[r]);
		}
		
		vec_view pivotrow(HT[r]);
		pivotrow.clearbit(pivot_end);
		auto HTrowit = HT[0];
		for (r2 = 0; r2 < HT.rows(); ++r2,++HTrowit)
			if (HT(r2,pivot_end))
				HTrowit.vxor(pivotrow);
		pivotrow.clear();
		pivotrow.setbit(pivot_end);
    	}
    	return pivot_end;
    }
    template<size_t bits>
    size_t echelonize(size_t row_begin, size_t row_end, size_t pivot_end, aligned_tag<bits>)
    {
    	for (size_t r = row_begin; r < row_end; ++r)
    	{
		--pivot_end;
		// find pivot for row r column pivot_start
		// normally we swap columns, but row swaps are also allowed for ISD
		size_t r2 = r;
		for (; r2 < HT.rows() && HT(r2,pivot_end)==false; ++r2)
			;
		if (r2 == HT.rows())
		{
			++pivot_end;
			continue;
		}
		if (r2 != r)
		{
			std::swap(perm[r], perm[r2]);
			HT[r2].swap(HT[r], aligned_tag<bits>());
		}
		
		vec_view pivotrow(HT[r]);
		pivotrow.clearbit(pivot_end);
		auto HTrowit = HT[0];
		for (r2 = 0; r2 < HT.rows(); ++r2,++HTrowit)
			if (HT(r2,pivot_end))
				HTrowit.vxor(pivotrow, aligned_tag<bits>());
		pivotrow.clear(aligned_tag<bits>());
		pivotrow.setbit(pivot_end);
    	}
    	return pivot_end;
    }
    
    // requires that HT has ISD form up to row_start
    void next_form(size_t AI_rows, size_t row_start = 0)
    {
    	row_permute(row_start, AI_rows, AI_rows, HT.rows());
    	size_t pivot_end = echelonize(row_start, AI_rows, HT.columns() - row_start);
    	if (HT.columns() - AI_rows != pivot_end)
    	{
    		std::cerr << "HT_ISD_form_t::next_form(" << AI_rows << "," << row_start << "): pivot_end=" << pivot_end << " != HT.columns()-AI_rows=" << HT.columns()-AI_rows << std::endl;
    	}
    }
    template<size_t bits>
    void next_form(size_t AI_rows, size_t row_start, aligned_tag<bits>)
    {
    	row_permute(row_start, AI_rows, AI_rows, HT.rows(), aligned_tag<bits>());
    	size_t pivot_end = echelonize(row_start, AI_rows, HT.columns() - row_start, aligned_tag<bits>());
    	if (HT.columns() - AI_rows != pivot_end)
    	{
    		std::cerr << "HT_ISD_form_t::next_form(" << AI_rows << "," << row_start << "): pivot_end=" << pivot_end << " != HT.columns()-AI_rows=" << HT.columns()-AI_rows << std::endl;
    	}
    }
    
private:
    mat_view HT;
    std::vector<uint32_t> perm;
    mccl_base_random_generator rndgen;
};

























class matrix_permute_t
{
public:
    matrix_permute_t() {}
    matrix_permute_t(const mat_view& _m) { reset(_m); }
    void reset(const mat_view& _m)
    {
    	m.reset(_m.ptr);
    	permutation.resize(m.columns());
    	std::iota(permutation.begin(), permutation.end(), 0);
    };

    // permute columns: swap each column in [b1:e1) with a uniformly random chosen column from [b2:e2)
    void random_permute(size_t b1, size_t e1, size_t b2, size_t e2)
    {
    	if (b1 >= e1 || b2 >= e2)
    		return;
	if (e1 > m.columns())
		e1 = m.columns();
	if (e2 > m.columns())
		e2 = m.columns();
    	size_t n2 = e2 - b2;
    	for (size_t i = b1; i < e1; ++i)
    	{
    		size_t j = b2 + (rndgen() % n2);
    		if (j == i)
    			continue;
    		std::swap(permutation[i], permutation[j]);
    		
    		size_t wi = i/64, wj = (j/64) - wi, rj = (i-j)%64;
    		uint64_t maski = uint64_t(1) << (i%64);
    		uint64_t* ptri = m.ptr.data(0) + wi;
    		if (wj != 0)
    		{
	    		for (size_t k = 0; k < m.rows(); ++k,ptri+=m.stride())
	    		{
	    			// constant time swap of two bits in two distinct uint64_t
	    			uint64_t xi = *ptri, xj = rotate_left(*(ptri+wj),rj);
	    			uint64_t tmp = (xi ^ xj) & maski;
	    			*ptri = xi ^ tmp;
	    			*(ptri+wj) = rotate_right(xj ^ tmp, rj);
	    		}
		} else
		{
	    		for (size_t k = 0; k < m.rows(); ++k,ptri+=m.stride())
	    		{
	    			// constant time swap of two bits in one uint64_t
	    			uint64_t xi = *ptri;
	    			uint64_t tmp = (xi ^ rotate_left(xi,rj)) & maski;
	    			*ptri = xi ^ tmp ^ rotate_right(tmp,rj);
	    		}
		}
    	}
    }

    void random_permute(size_t b = 0, size_t e = ~uint64_t(0))
    {
	random_permute(b,e,b,e);
    }
    
    std::vector<uint32_t>& get_permutation()
    {
  	return permutation;
    }
    
private:
    mat_view m;
    std::vector<uint32_t> permutation;
    mccl_base_random_generator rndgen;
};


class matrix_enumeraterows_t
{
public:
	matrix_enumeraterows_t()
	{}
	matrix_enumeraterows_t(const cmat_view& m, unsigned rmax, unsigned rmin=1)
		: _m(m), _partialsums(rmax+1, m.columns()), _rmin(rmin), _rmax(rmax)
	{
		reset(rmax, rmin);
	}
	matrix_enumeraterows_t(const cmat_view& m, const cvec_view& origin, unsigned rmax, unsigned rmin=1)
		: _m(m), _partialsums(rmax+1, m.columns())
	{
		reset(origin, rmax, rmin);
	}
	void reset(const cmat_view& m, unsigned rmax, unsigned rmin=1)
	{
		_m.reset(m);
		_partialsums = mat(rmax+1, m.columns());
		reset(rmax, rmin);
	}
	void reset(const cmat_view& m, const cvec_view& origin, unsigned rmax, unsigned rmin=1)
	{
		_m.reset(m);
		_partialsums = mat(rmax+1, m.columns());
		reset(origin, rmax, rmin);
	}
	// reset: reset enumeration
	// assume matrix_ref doesn't change, its content may be altered
	void reset(unsigned rmax, unsigned rmin=1)
	{
		_rmax = rmax;
		_rmin = rmin;
		assert(_rmax > 0);
		assert(_rmin > 0);
		if (_rmax > _m.rows())
			_rmax = _m.rows();
		if (_rmin > _rmax)
			_rmin = _rmax;

		if (_partialsums.rows() < rmax+1)
			_partialsums = mat(rmax+1, _m.columns());
		_partialsums[0].clear();
		for (unsigned i = 0; i < _rmin-1; ++i)
		{
			_partialsums[i+1] = _partialsums[i] ^ _m[i];
			_selection[i] = i;
		}
		_selection[_rmin-1]=_rmin-1;
		_dst.reset(_partialsums[_rmax].ptr);
	}
	void reset(const cvec_view& origin, unsigned rmax, unsigned rmin=1)
	{
		if (origin.columns() != _m.columns())
			throw std::runtime_error("matrix_enumeraterows_t:: origin does not have the correct number of columns");
		_rmax = rmax;
		_rmin = rmin;
		assert(_rmax > 0);
		assert(_rmin > 0);
		if (_rmax > _m.rows())
			_rmax = _m.rows();
		if (_rmin > _rmax)
			_rmin = _rmax;

		if (_partialsums.rows() < rmax+1)
			_partialsums = mat(rmax+1, _m.columns());
		_partialsums[0] = v_copy(origin);
		for (unsigned i = 0; i < _rmin-1; ++i)
		{
			_partialsums[i+1] = _partialsums[i] ^ _m[i];
			_selection[i] = i;
		}
		_selection[_rmin-1]=_rmin-1;
		_dst.reset(_partialsums[_rmax].ptr);
	}

	// the internal result vector, don't modify the reference! you can modify its contents
	const vec_view& result() { return _dst; }
	unsigned* selection() { return _selection; }
	unsigned selectionsize() const { return _rmin; }
	
	// compute next combination
	const vec_view& compute()
	{
		_dst = _partialsums[_rmin-1] ^ _m[_selection[_rmin-1]];
		return _dst;
	}
	void compute(const vec_view& dst)
	{
		dst = _partialsums[_rmin-1] ^ _m[_selection[_rmin-1]];
	}
	template<size_t bits>
	const vec_view& compute(aligned_tag<bits>)
	{
		_dst = v_xor(_partialsums[_rmin-1], _m[_selection[_rmin-1]], aligned_tag<bits>());
		return _dst;
	}
	template<size_t bits>
	void compute(const vec_view& dst, aligned_tag<bits>)
	{
		dst = v_xor(_partialsums[_rmin-1], _m[_selection[_rmin-1]], aligned_tag<bits>());
	}
	// move to next selection, return false if none
	bool next()
	{
		// determine next sum to generate
		if (++_selection[_rmin-1] < _m.rows()) // TODO : USE LIKELY MACRO
			return true;
		// slow path
		return _nextselection();
	}
	template<size_t bits>
	bool next(aligned_tag<bits>)
	{
		// determine next sum to generate
		if (++_selection[_rmin-1] < _m.rows()) // TODO : USE LIKELY MACRO
			return true;
		// slow path
		return _nextselection(aligned_tag<bits>());
	}
	
private:
	bool _nextselection()
	{
		unsigned i = _rmin-1;
		while (i >= 1)
		{
			// increase prior index, check if it overflows as well
			if (++_selection[i-1] >= _m.rows() - (_rmin-i))
				--i; // yes it overflows, continue to check earlier index
			else
			{
				// no it doesn't overflow: update partial sum
				_partialsums[i] = _partialsums[i-1] ^ _m[_selection[i-1]];
				break;
			}
		}
		if (i == 0)
		{
			// end of enumeration for this value of _rmin
			if (++_rmin > _rmax)
				return false;
			_selection[0] = 0;
			_partialsums[1] = _partialsums[0] ^ _m[0];
			i = 1;
		}
		// i >= 1 && valid _selection[i-1] & _partialsums[i]
		// reset remaining selection & partial sums
		for (; i < _rmin-1; ++i)
		{
			_selection[i] = _selection[i-1] + 1;
			_partialsums[i+1] = _partialsums[i] ^ _m[_selection[i]];
		}
		// correct as _rmin >= 2 when you reach this point
		_selection[_rmin-1] = _selection[_rmin-2] + 1;
		return true;
	}
	template<size_t bits>
	bool _nextselection(aligned_tag<bits>)
	{
		unsigned i = _rmin-1;
		while (i >= 1)
		{
			// increase prior index, check if it overflows as well
			if (++_selection[i-1] >= _m.rows() - (_rmin-i))
				--i; // yes it overflows, continue to check earlier index
			else
			{
				// no it doesn't overflow: update partial sum
				_partialsums[i] = v_xor(_partialsums[i-1], _m[_selection[i-1]], aligned_tag<bits>());
				break;
			}
		}
		if (i == 0)
		{
			// end of enumeration for this value of _rmin
			if (++_rmin > _rmax)
				return false;
			_selection[0] = 0;
			_partialsums[1] = v_xor(_partialsums[0], _m[0], aligned_tag<bits>());
			i = 1;
		}
		// i >= 1 && valid _selection[i-1] & _partialsums[i]
		// reset remaining selection & partial sums
		for (; i < _rmin-1; ++i)
		{
			_selection[i] = _selection[i-1] + 1;
			_partialsums[i+1] = v_xor(_partialsums[i], _m[_selection[i]], aligned_tag<bits>());
		}
		// correct as _rmin >= 2 when you reach this point
		_selection[_rmin-1] = _selection[_rmin-2] + 1;
		return true;
	}
	
	cmat_view _m;
	mat _partialsums;
	vec_view _dst;
	
	unsigned _selection[16]; // <= hard-coded maximum of sum of 16 vectors
	unsigned _rmin, _rmax;

};

MCCL_END_NAMESPACE

#endif
