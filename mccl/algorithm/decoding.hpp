// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_ALGORITHM_DECODING_HPP
#define MCCL_ALGORITHM_DECODING_HPP

#include <mccl/core/matrix.hpp>
#include <mccl/core/matrix_permute.hpp>
#include <functional>
#include <stdlib.h>

MCCL_BEGIN_NAMESPACE

// virtual base class: interface to find a single solution for ISD for a given syndrome target
class ISD_API_single
{
public:
    // pass parameters to actual object
    //virtual void configure(parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H and target syndrome s
    virtual void initialize(const mat_view& H, const vec_view& s, unsigned int w) = 0;
    
    // probabilistic preparation of loop invariant
    virtual void prepare_loop() = 0;
    
    // perform one loop iteration, return true if successful and store result in e
    virtual bool loop_next() = 0;
    
    // run loop until a solution is found
    virtual void solve()
    {
        prepare_loop();
        while (loop_next())
            ;
    }
};

// default callback types for subISD to the main ISD
typedef bool (*ISD_callback_t)(void*, const cvec_view, size_t); 
typedef bool (*ISD_sparse_callback_t)(void*, const std::vector<uint32_t>&, size_t);

// generic callback functions that can be instantiated for any main ISD class
template<typename ISD_t>
bool ISD_callback(void* ptr, const cvec_view vec, size_t w)
{
    return reinterpret_cast<ISD_t*>(ptr)->callback(vec, w);
}
template<typename ISD_t>
bool ISD_sparse_callback(void* ptr, const std::vector<uint32_t>& vec, size_t w)
{
    return reinterpret_cast<ISD_t*>(ptr)->callback(vec, w);
}

// virtual base class: interface for 'exhaustive' ISD returning as many solutions as efficiently as possible
template<typename callback_t = ISD_callback_t>
class ISD_API_exhaustive
{
public:
    // pass parameters to actual object
    //virtual void configure(parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H and target syndrome s
    virtual void initialize(const mat_view& H, const vec_view& s, unsigned int w, callback_t callback, void* ptr = nullptr) = 0;
    
    // preparation of loop invariant
    virtual void prepare_loop()
    {
    };
    
    // perform one loop iteration, return true if not finished
    virtual bool loop_next() = 0;
    
    // run loop and pass all solutions through callback
    virtual void solve()
    {
        prepare_loop();
        while (loop_next())
            ;
    }
};

static inline size_t get_scratch(size_t k, size_t sz) 
{
    return ((k+sz-1)/sz) * sz - k;
}

template<typename callback_t = ISD_callback_t>
class LB: public ISD_API_exhaustive<callback_t>
{
private:
    mat H;
    mat H01T;
    vec S;
    
    mat_view H01_S;
    mat_view H01T_S;
    matrix_permute_t permutator;
    
    size_t n,k,w,rows,cols0,cols1,scratch0,cnt;
    vec sol;

public:    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const mat_view& H_, const vec_view& S, unsigned int w_, callback_t callback, void* ptr) final 
    {
        cnt = 0;
        w = w_;
        n = H_.columns();
        k = n-H_.rows();
        rows = n-k;
        cols0 = n-k;
        scratch0 = get_scratch(cols0, 64);
        cols1 = k;
        H.resize(n-k, scratch0+cols0+cols1+1); // scratchcols || H0 || H1 || S
        for( size_t i = 0; i < rows; ++i ) 
        {
            for( size_t j = 0; j < cols0; ++j ) 
            {
                H.setbit(i, scratch0+j, H_(i,j));
            }
            for( size_t j = 0; j < cols1; ++j ) 
            {
                H.setbit(i, scratch0+cols0+j, H_(i, cols0+j));
            }
            H.setbit(i, cols0+scratch0+cols1, S[i]);
        }

//        size_t scratch1 = get_scratch(cols1+1, 64);
        H01_S.reset(H.submatrix(0, rows, scratch0+cols0, cols1+1));

        H01T.resize(cols1+1, rows);
        H01T_S.reset(H01T.submatrix(0, cols1+1, 0, rows));

        permutator.reset(H);
        sol.resize(n);
    }
    
    // // preparation of loop invariant
    void prepare_loop() final
    {
    }
    
    // perform one loop iteration, return true if not finished
    bool loop_next() final
    {
        ++cnt;
        permutator.random_permute(scratch0, scratch0+cols0, scratch0, scratch0+n);
        auto pivotend = echelonize(H, scratch0, scratch0+cols0);
        if(pivotend != cols0)
            return true;

        H01T_S.transpose(H01_S);
        auto S0 = H01T_S[cols1];
        if (hammingweight(S0) <= w) {
            auto perm = permutator.get_permutation();
            for( size_t i = 0; i < n-k; i++ )
            {
                if (S0[i])
                    sol.setbit(perm[scratch0+i]-scratch0);
            }
            std::cerr << "Found solution after " << cnt << " iterations." << std::endl;
            std::cerr << S0 << std::endl;
            std::cerr << sol << std::endl;
            return false;
        }
        return true;
    }
    
    // // run loop and pass all solutions through callback
    void solve() final
    {
         prepare_loop();
         while (loop_next())
             ;
    }
};

class Solution : public std::exception {
    vec sol;
    
    public:
        Solution(const cvec_view& sol_) 
            : std::exception(), sol(sol_)
        {
        }
        
    cvec_view get_solution() { return sol; }
};

bool check_solution(const mat_view& H01T_view, const vec_view& S0, const std::vector<uint32_t>& perm, size_t w, const std::vector<uint32_t>& E1_sparse, size_t w1) 
{
    vec E0(S0);
    for( auto i : E1_sparse ) {
        E0 ^= H01T_view[i];
    }
    if (hammingweight(E0)< w-w1-E1_sparse.size()) 
    {
        std::cerr << "Found solution" << std::endl;
        // recover and submit solution?
        std::cerr << "found solution " << hammingweight(E0) << " " << w << " " << w1 << " " << E1_sparse.size() << std::endl;
        std::cerr << E0 << " " << std::endl;
        size_t k = H01T_view.rows();
        size_t n = H01T_view.columns()+k;
        size_t scratch0 = get_scratch(n-k, 64);
        
        vec sol(n);
        for( size_t i = 0; i < n-k; i++ ) {
            if (E0[i])
                sol.setbit(perm[scratch0+i]-scratch0);
        }
        for( auto i : E1_sparse ) {
            sol.setbit(perm[n-k+i+scratch0]-scratch0);
        }
        throw Solution(sol);
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

template<typename subISD_t = ISD_API_exhaustive<>>
class ISD_single_generic: public ISD_API_single
{
private:
    mat H;
    mat H01T;
    vec S;

    vec_view S0_view;
    mat_view H01_S_view, H01T_view, H01T_S_view, H11_S_view, H11T_view, H11T_S_view;    
    matrix_permute_t permutator;
    
    size_t n,k,w,rows,cols0,cols1,scratch0,cnt;
public:
    ISD_single_generic(subISD_t& sI)
        : subISD(&sI)
    {
    }
    
    // pass parameters to actual object
    //virtual void configure(parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const mat_view& H_, const vec_view& S, unsigned int w_) final
    {
        cnt = 0;
        w = w_;
        n = H_.columns();
        k = n-H_.rows();
        rows = n-k;
        cols0 = n-k;
        scratch0 = get_scratch(cols0, 64);
        cols1 = k;
        H.resize(n-k, scratch0 + cols0 + cols1 + 1); // scratch0 || cols0 || cols1 || S
        H.clear();
        for( size_t i = 0; i < rows; i++ ) {
            for( size_t j = 0; j < cols0; j++ ) {
                H.setbit(i, scratch0 + j, H_(i,j));
            }
            for( size_t j = 0; j < cols1; j++ ) {
                H.setbit(i, scratch0 + cols0 + j, H_(i, cols0+j));
            }
            H.setbit(i, scratch0 + cols0 + cols1, S[i]);
        }

//        size_t scratch1 = get_scratch(cols1+1, 64);
        H01_S_view.reset(H.submatrix(0, rows, cols0+scratch0, cols1+1));

        H01T.resize(cols1+1, rows);
        H01T.clear();
//        size_t scratch01T = get_scratch(rows, 64);
        H01T_S_view.reset(H01T.submatrix(0, cols1+1, 0, rows));
        H01T_view.reset(H01T.submatrix(0, cols1, 0, rows));
        S0_view.reset(H01T.subvector(cols1, 0, rows));

        // todo: fix for l > 0
        H11T_view.reset(H01T.submatrix(0, cols1, 0, rows));

        permutator.reset(H);
    }
    
    // callback function
    inline bool callback(const std::vector<uint32_t>& E1_sparse, size_t w1)
    {
            if(check_solution(this->H01T_view, this->S0_view, this->permutator.get_permutation(), this->w, E1_sparse, w1)) 
            {
                std::cerr << "SubISD solution" << std::endl;
                return true;
            }
            return false;
    }
    
    // probabilistic preparation of loop invariant
    void prepare_loop() final
    {
/*
        std::function<bool(const std::vector<uint32_t>&, size_t)> callback = [this](const std::vector<uint32_t>& E1_sparse, size_t w1)
        {
            if(check_solution(this->H01T_view, this->S0_view, this->permutator.get_permutation(), this->w, E1_sparse, w1)) 
            {
                std::cerr << "SubISD solution" << std::endl;
                return true;
            }
            return false;
        };
*/
        // still assuming ell=0
        vec_view S1; // todo: take from H11T_S_view if ell>0
        size_t w1_max = 0;

        subISD->initialize(H11T_view, S1, w1_max, ISD_sparse_callback<ISD_single_generic<subISD_t>>, this);
    }
    
    // perform one loop iteration, return true if successful and store result in e
    bool loop_next() final
    {
        ++cnt;
        permutator.random_permute(scratch0, scratch0+cols0, scratch0, scratch0+n);
        auto pivotend = echelonize(H, scratch0, scratch0+cols0);
        if(pivotend != cols0)
            return true;

        H01T_S_view.transpose(H01_S_view);
        try
        {
            subISD->solve();
        } catch(Solution& sol)
        {
            std::cerr << "ISD_single_generic found solution after " << cnt << " iterations" << std::endl;
            std::cerr << sol.get_solution() << std::endl;
            // todo: free memory
            throw sol;
            return false;
        }
        return true;
    }
    
    // run loop until a solution is found
    void solve() final
    {
        prepare_loop();
        while (loop_next())
            ;
    }
    
    subISD_t* subISD;
};

template<typename callback_t = ISD_sparse_callback_t>
class subISD_prange: public ISD_API_exhaustive<callback_t>
{   
private:
    callback_t callback;
    void* ptr;
    std::vector<uint32_t> E1_sparse;
        
public:
    void initialize(const mat_view& H_, const vec_view& S, unsigned int w_, callback_t _callback, void* _ptr)
    {
        callback = _callback;
        ptr = _ptr;
    }

    bool loop_next()
    {
        (*callback)(ptr,E1_sparse, 0);
        return false;
    }
};

template<typename callback_t = ISD_sparse_callback_t>
class subISD_LB: public ISD_API_exhaustive<callback_t>
{   
private:
    callback_t callback;
    void* ptr;
    matrix_enumeraterows_t rowenum;
    size_t p = 3;
    std::vector<uint32_t> E1_sparse;
public:
    void initialize(const mat_view& H_, const vec_view& S, unsigned int w_, callback_t _callback, void* _ptr) final
    {
        callback = _callback;
        ptr = _ptr;
        rowenum.reset(H_, p, 1);
        E1_sparse.resize(1);
    }

    void prepare_loop() final 
    {
        rowenum.reset(p, 1);
    }

    bool loop_next() final
    {
        rowenum.compute();

        // todo: optimize and pass computed error sum
        E1_sparse[0] = *rowenum.selection();
        (*callback)(ptr, E1_sparse, 0);
        return rowenum.next();
    }

    void solve() final
    {
        prepare_loop();
        while (loop_next())
            ;
    }
};

MCCL_END_NAMESPACE

#endif
