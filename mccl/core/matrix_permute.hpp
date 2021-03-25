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
class vector_ptr_t;
template<typename data_t>
class vector_ref_t;
template<typename data_t>
class matrix_ptr_t;
template<typename data_t>
class matrix_ref_t;

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

    // permute uniformly random columns into first l columns
    // non-optimized
    void random_permute(size_t l) {
    	size_t n = mat_ref.columns();
    	uint64_t rng;
    	for(size_t i = 0; i < l-1; i++) {
    		gen(rng);
    		size_t j = i + (rng%(n-i));
    		std::swap(permutation[i], permutation[j]);
    		for(size_t k = 0; k < mat_ref.rows(); k++ ) {
    			bool c = mat_ref(k,i);
    			mat_ref.bitset(k,i,mat_ref(k,j));
    			mat_ref.bitset(k,j,c);
    		}
    	}
    };
private:
	matrix_ref mat_ref;
	std::vector<uint32_t> permutation;
	mccl_base_random_generator<uint64_t> gen;
};

MCCL_END_NAMESPACE

#endif
