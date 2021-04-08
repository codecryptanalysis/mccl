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

MCCL_END_NAMESPACE

#endif
