#ifndef MCCL_CORE_MATRIX_PERMUTE_HPP
#define MCCL_CORE_MATRIX_PERMUTE_HPP

#include <mccl/core/matrix.hpp>

#include <iostream>
#include <functional>
#include <random>
#include <numeric>
#include <utility>

MCCL_BEGIN_NAMESPACE

template<typename data_t>
class matrix_permute_t
{
public:
	typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;

    matrix_permute_t(matrix_ref& m) : mat_ref(m) {
    	permutation.resize(m.columns());
    	std::iota(permutation.begin(), permutation.end(), 0);
    };

    // permute uniformly random columns from [l:r] into [l:m]
    // non-optimized
    void random_permute(size_t l, size_t m, size_t r) {
    	size_t n = r-l;
    	uint64_t rng;
    	for(size_t i = 0; i < m-l; i++) {
    		gen(rng);
    		size_t j = i + (rng%(n-i));
    		std::swap(permutation[l+i], permutation[l+j]);
    		for(size_t k = 0; k < mat_ref.rows(); k++ ) {
    			bool c = mat_ref(k,l+i);
    			mat_ref.bitset(k,l+i,mat_ref(k,l+j));
    			mat_ref.bitset(k,l+j,c);
    		}
    	}
    };

  std::vector<uint32_t> get_permutation() {
    return permutation;
  }
  
private:
	matrix_ref mat_ref;
	std::vector<uint32_t> permutation;
	mccl_base_random_generator<uint64_t> gen;
};

template<typename data_t>
class matrix_enumeraterows_t
{
public:
	typedef matrix_ref_t<data_t> matrix_ref;
	typedef matrix_t<data_t> matrix;
	typedef vector_ref_t<data_t> vector_ref;
	typedef vector_ptr_t<data_t> vector_ptr;
	
	matrix_enumeraterows_t(matrix_ref& m, unsigned rmax, unsigned rmin=1)
		: _m(m), _partialsums(rmax+1, m.columns()), _rmin(rmin), _rmax(rmax)
	{
		reset(rmax, rmin);
	}
	matrix_enumeraterows_t(matrix_ref& m, vector_ref& origin, unsigned rmax, unsigned rmin=1)
		: _m(m), _partialsums(rmax+1, m.columns())
	{
		reset(origin, rmax, rmin);
	}
	
	// reset: reset enumeration
	// assume matrix_ref doesn't change, its content may be altered
	void reset(unsigned rmax, unsigned rmin=1)
	{
		assert(_rmax > 0);
		assert(_rmin > 0);
		if (_rmax > _m.rows())
			_rmax = _m.rows();
		if (_rmin > _rmax)
			_rmin = _rmax;

		_partialsums[0].setzero();
		for (unsigned i = 0; i < _rmin-1; ++i)
		{
			_partialsums[i+1].op_xor(_partialsums[i], _m[i]);
			_selection[i] = i;
		}
		_selection[_rmin-1]=_rmin-1;
		_dst.reset(_partialsums[_rmax].base());
	}
	void reset(vector_ref& origin, unsigned rmax, unsigned rmin=1)
	{
		if (origin.columns() != _m.columns())
			throw std::runtime_error("matrix_enumeraterows_t:: origin does not have the correct number of columns");
		assert(_rmax > 0);
		assert(_rmin > 0);
		if (_rmax > _m.rows())
			_rmax = _m.rows();
		if (_rmin > _rmax)
			_rmin = _rmax;

		_partialsums[0] = origin;
		for (unsigned i = 0; i < _rmin-1; ++i)
		{
			_partialsums[i+1].op_xor(_partialsums[i], _m[i]);
			_selection[i] = i;
		}
		_selection[_rmin-1]=_rmin-1;
		_dst.reset(_partialsums[_rmax].base());
	}

	// the internal result vector, don't modify the reference! you can modify its contents
	vector_ref& result() { return _dst; }
	unsigned* selection() { return _selection; }
	unsigned selectionsize() const { return _rmin; }
	
	// compute next combination
	vector_ref& compute()
	{
		_dst.op_xor(_partialsums[_rmin-1], _m[_selection[_rmin-1]]);
		return _dst;
	}
	void compute(vector_ref& dst)
	{
		dst.op_xor(_partialsums[_rmin-1], _m[_selection[_rmin-1]]);
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
				_partialsums[i].op_xor(_partialsums[i-1], _m[_selection[i-1]]);
				break;
			}
		}
		if (i == 0)
		{
			// end of enumeration for this value of _rmin
			if (++_rmin > _rmax)
				return false;
			_selection[0] = 0;
			_partialsums[1].op_xor(_partialsums[0], _m[0]);
			i = 1;
		}
		// i >= 1 && valid _selection[i-1] & _partialsums[i]
		// reset remaining selection & partial sums
		for (; i < _rmin-1; ++i)
		{
			_selection[i] = _selection[i-1] + 1;
			_partialsums[i+1].op_xor(_partialsums[i], _m[_selection[i]]);
		}
		// correct as _rmin >= 2 when you reach this point
		_selection[_rmin-1] = _selection[_rmin-2] + 1;
		return true;
	}
	
	matrix_ref _m;
	matrix _partialsums;
	vector_ref _dst;
	
	unsigned _selection[16]; // <= hard-coded maximum of sum of 16 vectors
	unsigned _rmin, _rmax;

};

MCCL_END_NAMESPACE

#endif
