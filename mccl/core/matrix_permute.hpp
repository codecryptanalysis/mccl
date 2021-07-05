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
    HT_ISD_form_t(const mat_view& _HT, size_t _echelon_rows) { reset(_HT, _echelon_rows); }
    void reset(const mat_view& _HT, size_t _echelon_rows)
    {
    	assert(_echelon_rows < _HT.rows());

    	HT.reset(_HT.ptr);
    	echelon_rows = _echelon_rows;
    	ISD_rows = HT.rows() - echelon_rows;
    	cur_row = 0;

    	perm.resize(HT.rows());
    	std::iota(perm.begin(), perm.end(), 0);

    	for (size_t i = 0; i < echelon_rows; ++i)
    		update1(i);
    };

    const std::vector<uint32_t>& permutation() const { return perm; }
    
    vec_view_it operator[](size_t r) const { return HT[r]; }
    vec_view_it operator()(size_t r) const { return HT[r]; }
    
    // update 1 echelon row: swap with random row outside echelon form and bring it back to echelon form
    void update1(size_t idx)
    {
    	assert(idx < echelon_rows);
    	// echelon row idx must have 1-bit in column pivotcol:
    	size_t pivotcol = HT.columns() - idx - 1;
    	// find random row to swap with
    	//   must have bit set at pivot column
    	//   start at random position and then do linear search
    	//   TODO: avoid ending up at previously swapped out column, maybe reduce search region?
    	size_t r = rndgen() % ISD_rows;
	for (; r < ISD_rows && HT(echelon_rows + r,pivotcol)==false; ++r)
		;
	// wrap around
	if (r == ISD_rows) // unlikely
	{
		r = 0;
		for (; r < ISD_rows && HT(echelon_rows + r,pivotcol)==false; ++r)
			;
	}
	// oh oh if we wrap around twice
	if (r == ISD_rows) // unlikely
		throw std::runtime_error("HT_ISD_form_t::update(): cannot find pivot");
	// swap rows
	std::swap(perm[idx], perm[echelon_rows + r]);
	HT[idx].swap(HT[echelon_rows + r]);

	// bring back in echelon form
	vec_view pivotrow(HT[idx]);
	pivotrow.clearbit(pivotcol);
	auto HTrowit = HT[0];
	for (size_t r2 = 0; r2 < HT.rows(); ++r2,++HTrowit)
		if (HT(r2,pivotcol))
			HTrowit.vxor(pivotrow);
	pivotrow.clear();
	pivotrow.setbit(pivotcol);
    }
    // update 1 echelon row: swap with random row outside echelon form and bring it back to echelon form
    template<size_t bits>
    void update1(size_t idx, aligned_tag<bits>)
    {
    	assert(idx < echelon_rows);
    	// echelon row idx must have 1-bit in column pivotcol:
    	size_t pivotcol = HT.columns() - idx - 1;
    	// find random row to swap with
    	//   must have bit set at pivot column
    	//   start at random position and then do linear search
    	//   TODO: avoid ending up at previously swapped out column, maybe reduce search region?
    	size_t r = rndgen() % ISD_rows;
	for (; r < ISD_rows && HT(r,pivotcol)==false; ++r)
		;
	// wrap around
	if (r == ISD_rows) // unlikely
	{
		r = 0;
		for (; r < ISD_rows && HT(r,idx)==false; ++r)
			;
	}
	// oh oh if we wrap around twice
	if (r == ISD_rows) // unlikely
		throw std::runtime_error("HT_ISD_form_t::update(): cannot find pivot");
	// swap rows
	std::swap(perm[idx], perm[r]);
	HT[idx].swap(HT[r], aligned_tag<bits>());

	// bring back in echelon form
	vec_view pivotrow(HT[idx]);
	pivotrow.clearbit(pivotcol);
	auto HTrowit = HT[0];
	for (size_t r2 = 0; r2 < HT.rows(); ++r2,++HTrowit)
		if (HT(r2,pivotcol))
			HTrowit.vxor(pivotrow, aligned_tag<bits>());
	pivotrow.clear(aligned_tag<bits>());
	pivotrow.setbit(pivotcol);
    }

    void update1()
    {
    	update(cur_row++);
    }
    template<size_t bits>
    void update1(aligned_tag<bits>)
    {
    	update(cur_row++, aligned_tag<bits>());
    }

    void update(size_t rows)
    {
    	for (size_t i = 0; i < rows; ++i)
    		update1();
	// TODO: improved multirow update using method of 4 russians
    }
    template<size_t bits>
    void update(size_t rows, aligned_tag<bits>)
    {
    	for (size_t i = 0; i < rows; ++i)
    		update1(aligned_tag<bits>());
	// TODO: improved multirow update using method of 4 russians
    }

private:
    mat_view HT;
    std::vector<uint32_t> perm;
    size_t echelon_rows, ISD_rows, cur_row;
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
