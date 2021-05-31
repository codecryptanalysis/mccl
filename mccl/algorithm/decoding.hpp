// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_ALGORITHM_DECODING_HPP
#define MCCL_ALGORITHM_DECODING_HPP

#include <mccl/core/matrix.hpp>
#include <mccl/core/matrix_permute.hpp>
#include <functional>
#include <stdlib.h>

MCCL_BEGIN_NAMESPACE

// virtual base class: interface to find a single solution for ISD for a given syndrome target
template<typename data_t>
class ISD_API_single
{
public:
    // pass parameters to actual object
    //virtual void configure(const parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    virtual void initialize(const matrix_ref_t<data_t>& H0, const vector_ref_t<data_t>& s0, unsigned int w) = 0;
    
    // probabilistic preparation of loop invariant
    virtual void prepare_loop() = 0;
    
    // perform one loop iteration, return true if successful and store result in e
    virtual bool loop_next() = 0;
    
    // run loop until a solution is found
    virtual void solve()
    {
        prepare_loop();
        while (!loop_next())
            ;
    }
    
    vector_t<data_t> e;
};

// virtual base class: interface for 'exhaustive' ISD returning as many solutions as efficiently as possible
template<typename data_t, typename callback_t = std::function<bool(vector_ref_t<data_t>&)>>
class ISD_API_exhaustive
{
public:
    // pass parameters to actual object
    //virtual void configure(const parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    virtual void initialize(const matrix_ref_t<data_t>& H0, const vector_ref_t<data_t>& s0, unsigned int w, callback_t& callback) = 0;
    
    // preparation of loop invariant
    virtual void prepare_loop(){
    };
    
    // perform one loop iteration, return true if not finished
    virtual bool loop_next(){
    };
    
    // run loop and pass all solutions through callback
    virtual void solve()
    {
        prepare_loop();
        while (loop_next())
            ;
    }
};

size_t get_scratch(size_t k, size_t sz) {
    return ((k+sz-1)/sz) * sz - k;
}

template<typename data_t, typename callback_t = std::function<bool(vector_ref_t<data_t>&)>>
class LB: public ISD_API_exhaustive<data_t, callback_t>
{
private:
    mccl::matrix_t<data_t>* H_ptr = nullptr;
    mccl::matrix_t<data_t>* H01T = nullptr;
    mccl::vector_t<data_t>* S_ptr = nullptr;
    mccl::matrix_ref_t<data_t>* H01_S_view = nullptr;
    mccl::matrix_ref_t<data_t>* H01T_S_view = nullptr;
    matrix_permute_t<uint64_t>* permutator = nullptr;
    size_t n,k,w,rows,cols0,cols1,scratch0,cnt;

public:    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const matrix_ref_t<data_t>& H_, const vector_ref_t<data_t>& S, unsigned int w_, callback_t& callback) {
        cnt = 0;
        w = w_;
        n = H_.columns();
        k = n-H_.rows();
        rows = n-k;
        cols0 = n-k;
        scratch0 = get_scratch(cols0, 64);
        cols1 = k;
        H_ptr = new mccl::matrix_t<data_t>(n-k, cols0+scratch0+cols1+1); // +1 to store S
        for( size_t i = 0; i < rows; i++ ) {
            for( size_t j = 0; j < cols0; j++ ) {
                H_ptr->bitset(i,j+scratch0,H_(i,j));
            }
            for( size_t j = 0; j < cols1; j++ ) {
                H_ptr->bitset(i, cols0+scratch0+j, H_(i, cols0+j));
            }
            H_ptr->bitset(i, cols0+scratch0+cols1, S[i]);
        }

        size_t scratch1 = get_scratch(cols1+1, 64);
        H01_S_view = new mccl::matrix_ref_t<data_t>(H_ptr->submatrix(0, rows, cols0+scratch0, cols1+1, scratch1));

        H01T = new mccl::matrix_t<data_t>(cols1+1, rows);
        size_t scratch01T = get_scratch(rows, 64);
        H01T_S_view = new mccl::matrix_ref_t<data_t>(H01T->submatrix(0, cols1+1, 0, rows, scratch01T));
    
        permutator = new matrix_permute_t<uint64_t>(*H_ptr);
    }
    
    // // preparation of loop invariant
    // virtual void prepare_loop() = 0;
    
    // perform one loop iteration, return true if not finished
    bool loop_next() {
        cnt++;
        permutator->random_permute(scratch0, scratch0+cols0, scratch0+n);
        auto pivotend = echelonize(*H_ptr, scratch0, scratch0+cols0);
        if(pivotend != cols0)
            return true;

        H01T_S_view->transpose(*H01_S_view);
        auto S0 = (*H01T_S_view)[cols1];
        if(hammingweight(S0) <= w) {
            auto perm = permutator->get_permutation();
            mccl::vector_t<uint64_t> sol(n);
            for( size_t i = 0; i < n-k; i++ ) {
                if (S0[i])
                    sol.bitset(perm[scratch0+i]-scratch0);
            }
            std::cerr << "Found solution after " << cnt << " iterations." << std::endl;
            std::cerr << S0 << std::endl;
            std::cerr << sol << std::endl;
            return false;
        }
        return true;
    }
    
    // // run loop and pass all solutions through callback
    // virtual void solve()
    // {
    //     prepare_loop();
    //     while (loop_next())
    //         ;
    // }

    void free() {
        if( H01_S_view != nullptr ) delete H01_S_view;
        if( H01T_S_view != nullptr ) delete H01T_S_view;
        if( H_ptr != nullptr ) delete H_ptr;
        if( H01T != nullptr ) delete H01T;
        if( S_ptr != nullptr ) delete S_ptr;
        if( permutator != nullptr ) delete permutator;
    };
};


template<typename data_t>
bool check_solution(const mccl::matrix_ref_t<data_t> &H01T_view, const vector_ref_t<data_t>& S0, const std::vector<uint32_t>& perm, size_t w, const std::vector<uint32_t>& E1_sparse, size_t w1) {
    vector_t<data_t> tmp(S0.columns());
    tmp ^= S0;
    for( auto i : E1_sparse ) {
        tmp ^= H01T_view[i];
    }
    if(hammingweight(tmp)< w-w1-E1_sparse.size()) {
        // recover and submit solution?
        return true;
    }
    return false;
}

/*
This is the generic algorithm solving target-ISD using a to-be-specified subISD algorithm
Therefore it is derived from ISD_API_target<data_t>.
The subISD algorithm class is a template parameter defaulted to ISD_API_exhaustive<data_t> for flexibility:
This enables it to work with any ISD_API_exhaustive derived object.
However, for optimized performance it is also possible to pass the exhaustive-ISD algorithm type directly
and remove the overhead of virtualized function calls.
In that case, be sure to make the crucial functions as final.
*/

template<typename data_t, typename subISD_t = ISD_API_exhaustive<data_t> >
class ISD_single_generic: public ISD_API_single<data_t>
{
private:
    mccl::matrix_t<data_t>* H_ptr = nullptr;
    mccl::matrix_t<data_t>* H01T = nullptr;
    mccl::vector_t<data_t>* S_ptr = nullptr;
    mccl::vector_ref_t<data_t>* S0_view = nullptr;
    mccl::matrix_ref_t<data_t>* H01_S_view = nullptr;
    mccl::matrix_ref_t<data_t>* H01T_view = nullptr;
    mccl::matrix_ref_t<data_t>* H01T_S_view = nullptr;
    mccl::matrix_ref_t<data_t>* H11_S_view = nullptr;
    mccl::matrix_ref_t<data_t>* H11T_view = nullptr;
    mccl::matrix_ref_t<data_t>* H11T_S_view = nullptr;
    matrix_permute_t<uint64_t>* permutator = nullptr;
    size_t n,k,w,rows,cols0,cols1,scratch0,cnt;
public:
    ISD_single_generic(subISD_t& sI)
        : subISD(&sI)
    {
    }
    
    // pass parameters to actual object
    //virtual void configure(const parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const matrix_ref_t<data_t>& H_, const vector_ref_t<data_t>& S, unsigned int w_) {
        cnt = 0;
        w = w_;
        n = H_.columns();
        k = n-H_.rows();
        rows = n-k;
        cols0 = n-k;
        scratch0 = get_scratch(cols0, 64);
        cols1 = k;
        H_ptr = new mccl::matrix_t<data_t>(n-k, cols0+scratch0+cols1+1); // +1 to store S
        for( size_t i = 0; i < rows; i++ ) {
            for( size_t j = 0; j < cols0; j++ ) {
                H_ptr->bitset(i,j+scratch0,H_(i,j));
            }
            for( size_t j = 0; j < cols1; j++ ) {
                H_ptr->bitset(i, cols0+scratch0+j, H_(i, cols0+j));
            }
            H_ptr->bitset(i, cols0+scratch0+cols1, S[i]);
        }

        size_t scratch1 = get_scratch(cols1+1, 64);
        H01_S_view = new mccl::matrix_ref_t<data_t>(H_ptr->submatrix(0, rows, cols0+scratch0, cols1+1, scratch1));


        H01T = new mccl::matrix_t<data_t>(cols1+1, rows);
        size_t scratch01T = get_scratch(rows, 64);
        H01T_S_view = new mccl::matrix_ref_t<data_t>(H01T->submatrix(0, cols1+1, 0, rows, scratch01T));
        H01T_view = new mccl::matrix_ref_t<data_t>(H01T->submatrix(0, cols1, 0, rows, scratch01T));
        S0_view = new mccl::vector_ref_t<data_t>(H01T->subvector(cols1, 0, rows, scratch01T));

        permutator = new matrix_permute_t<uint64_t>(*H_ptr);
    }
    
    // probabilistic preparation of loop invariant
    void prepare_loop() final
    {
        std::function<bool(const std::vector<uint32_t>&, size_t)> callback = [this](const std::vector<uint32_t>& E1_sparse, size_t w1){
            if(check_solution(*(this->H01T_view), *(this->S0_view), this->permutator->get_permutation(), this->w, E1_sparse, w1)) {
                std::cerr << "SubISD solution" << std::endl;
                return true;
            }
            return false;
        };

        // still assuming ell=0
        vector_ref_t<data_t> S1; // todo: take from H11T_S_view if ell>0
        size_t w1_max = 0;

        subISD->initialize(*H11T_view, S1, w1_max, callback);
    }
    
    // perform one loop iteration, return true if successful and store result in e
    bool loop_next() {
        cnt++;
        permutator->random_permute(scratch0, scratch0+cols0, scratch0+n);
        auto pivotend = echelonize(*H_ptr, scratch0, scratch0+cols0);
        if(pivotend != cols0)
            return true;

        H01T_S_view->transpose(*H01_S_view);
        subISD->solve();

        if(cnt>10) {
            return false;
        }
        return true;
    }
    
    // run loop until a solution is found
    void solve() final
    {
        prepare_loop();
        while (!loop_next())
            ;
    }
    
    subISD_t* subISD;
};

template<typename data_t, typename callback_t = std::function<bool(const std::vector<uint32_t>&, size_t)>>
class subISD_prange: public ISD_API_exhaustive<data_t, callback_t>
{   
private:
    callback_t callback;
public:
    void initialize(const matrix_ref_t<data_t>& H_, const vector_ref_t<data_t>& S, unsigned int w_, callback_t& _callback) {
        callback = _callback;
    }

    bool loop_next(){
        callback({}, 0);
        return false;
    }
};

MCCL_END_NAMESPACE

#endif
