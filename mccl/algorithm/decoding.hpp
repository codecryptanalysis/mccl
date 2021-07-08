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
    // if called then it must be called before initialize
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
// takes pointer to object & representation of H2T rows & and w1partial
// returns true if solution is correct and exhaustive enumeration can be stopped
typedef bool (*ISD_callback_t)(void*, const cvec_view, unsigned int);
typedef bool (*ISD_sparse_callback_t)(void*, const std::vector<uint32_t>&, unsigned int);
typedef bool (*ISD_sparserange_callback_t)(void*, const uint32_t*, const uint32_t*, unsigned int);

// generic callback functions that can be instantiated for any main ISD class
template<typename ISD_t>
bool ISD_callback(void* ptr, const cvec_view vec, unsigned int w)
{
    return reinterpret_cast<ISD_t*>(ptr)->callback(vec, w);
}
template<typename ISD_t>
bool ISD_sparse_callback(void* ptr, const std::vector<uint32_t>& vec, unsigned int w)
{
    return reinterpret_cast<ISD_t*>(ptr)->callback(vec, w);
}
template<typename ISD_t>
bool ISD_sparserange_callback(void* ptr, const uint32_t* begin, const uint32_t* end, unsigned int w)
{
    return reinterpret_cast<ISD_t*>(ptr)->callback(begin, end, w);
}

template<typename ISD_t>
ISD_callback_t make_ISD_callback(const ISD_t&, ISD_callback_t)
{
    return ISD_callback<ISD_t>;
}
template<typename ISD_t>
ISD_sparse_callback_t make_ISD_callback(const ISD_t&, ISD_sparse_callback_t)
{
    return ISD_sparse_callback<ISD_t>;
}
template<typename ISD_t>
ISD_sparserange_callback_t make_ISD_callback(const ISD_t&, ISD_sparserange_callback_t)
{
    return ISD_sparserange_callback<ISD_t>;
}


// virtual base class: interface for 'exhaustive' ISD returning as many solutions as efficiently as possible
template<typename _callback_t = ISD_callback_t>
class ISD_API_exhaustive
{
public:
    typedef _callback_t callback_t;
    // pass parameters to actual object
    // if called then it must be called before initialize
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
typedef ISD_API_exhaustive<ISD_callback_t> ISD_API_exhaustive_t;
typedef ISD_API_exhaustive<ISD_sparse_callback_t> ISD_API_exhaustive_sparse_t;
typedef ISD_API_exhaustive<ISD_sparserange_callback_t> ISD_API_exhaustive_sparserange_t;






// virtual base class: interface for 'exhaustive' ISD returning as many solutions as efficiently as possible
// differs from ISD_API_exhaustive in that H is given transposed for better efficiency between main_ISD and sub_ISD
template<typename _callback_t = ISD_sparserange_callback_t>
class ISD_API_exhaustive_transposed
{
public:
    typedef _callback_t callback_t;
    // pass parameters to actual object
    // if called then it must be called before initialize
    //virtual void configure(parameters_t& params) = 0;

    // deterministic initialization for given parity check matrix H and target syndrome s
    virtual void initialize(const cmat_view& Htransposed_padded, size_t Hcolumns, const cvec_view& s, unsigned int w, callback_t callback, void* ptr = nullptr) = 0;

    // preparation of loop invariant
    virtual void prepare_loop() = 0;

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
typedef ISD_API_exhaustive_transposed<ISD_callback_t> ISD_API_exhaustive_transposed_t;
typedef ISD_API_exhaustive_transposed<ISD_sparse_callback_t> ISD_API_exhaustive_transposed_sparse_t;
typedef ISD_API_exhaustive_transposed<ISD_sparserange_callback_t> ISD_API_exhaustive_transposed_sparserange_t;


// implementation of ISD_single_generic that can be instantiated with any subISD
// based on common view on transposed H
// will use reverse column ordering for column reduction on Htransposed (instead of row reduction on H)
//
// HT = ( 0   IR  ) where IR is the reversed identity matrix IR with 1's on the anti-diagonal
//      ( H1T H0T )
//
// this makes it easy to include additional columns of H0T together with H1T to subISD

template<typename subISDT_t = ISD_API_exhaustive_transposed_sparserange_t, size_t _bit_alignment = 64>
class ISD_single_generic_transposed: public ISD_API_single
{
public:
    typedef typename subISDT_t::callback_t callback_t;
    
    static const size_t bit_alignment = _bit_alignment;
    typedef aligned_tag<bit_alignment> this_aligned_tag;

    ISD_single_generic_transposed(subISDT_t& sI)
        : subISDT(&sI)
    {
        n = k = w = 0;
        l = 0; // ISD form parameter
        u = 1; // how many rows to swap each iteration. max n - k - l
    }

    // pass parameters to actual object
    //virtual void configure(parameters_t& params) = 0;
    // if called then it must be called before initialize
    void configure(size_t _l = 0, size_t _u = 1)
    {
        l = _l;
        u = _u;
    }

    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const mat_view& _H, const vec_view& _S, unsigned int _w) final
    {
        n = _H.columns();
        k = n - _H.rows();
        w = _w;
        HST.reset(_H, _S, l);

        C.resize(HST.Spadded().columns());
        C2padded.reset(C.subvector(0, HST.S2padded().columns()));
        C1restpadded.reset(C.subvector(HST.S2padded().columns(), HST.S1restpadded().columns()));
        
        sol.clear();
        cnt = 0;
    }

    // callback function
    inline bool callback(const uint32_t* begin, const uint32_t* end, unsigned int w1partial)
    {
            // weight of solution consists of w2 (=end-begin) + w1partial (given) + w1rest (computed below)
            size_t wsol = w1partial + (end - begin);
            if (wsol > w)
                return true;
            // 1. compute w1rest
            auto p = begin, e = end - 1;
            if (p == end)
            {
                wsol += hammingweight(HST.S1restpadded(), this_aligned_tag());
            }
            else if (p == e)
            {
                wsol += hammingweight_xor(HST.S1restpadded(), HST.H1Trestpadded()[*p], this_aligned_tag());
            }
            else
            {
                C1restpadded.vxor(HST.S1restpadded(), HST.H1Trestpadded()[*p], this_aligned_tag());
                for (++p; p != e; ++p)
                    C1restpadded.vxor(HST.H1Trestpadded()[*p], this_aligned_tag());
                wsol += hammingweight_xor(C1restpadded, HST.H1Trestpadded()[*e], this_aligned_tag());
            }
            // 2. check weight: too large => return
            if (wsol > w)
                return true;
            // 3. construct full solution on E1 and E2 part
            C.copy(HST.Spadded(), this_aligned_tag());
            for (p = begin; p != end; ++p)
                C.vxor(HST.H12Tpadded()[*p], this_aligned_tag());
            if (wsol != (end-begin) + hammingweight(C, this_aligned_tag()))
                throw std::runtime_error("ISD_single_generic_transposed::callback: internal error 1: w1partial is not correct?");
            for (p = begin; p != end; ++p)
                sol.push_back(HST.permutation( HST.echelonrows() + *p ));
            for (size_t c = 0; c < HST.HT().columns(); ++c)
            {
                if (C[c] == false)
                    continue;
                if (c < HST.H2T().columns())
                    throw std::runtime_error("ISD_single_generic_transposed::callback: internal error 2: H2T combination non-zero!");
                sol.push_back(HST.permutation( HST.HT().columns() - 1 - c ));
            }
            return false;
    }
    inline bool callback(const std::vector<uint32_t>& sol, unsigned int w1partial)
    {
        return callback(&sol[0], (&sol[0])+sol.size(), w1partial);
    }

    // probabilistic preparation of loop invariant
    void prepare_loop() final
    {
        subISDT->initialize(HST.H2Tpadded(), HST.H2T().columns(), HST.S2padded(), w, 
        //ISD_sparserange_callback<ISD_single_generic_transposed<subISDT_t>>, 
        make_ISD_callback(*this, callback_t()),
        this);
    }

    // perform one loop iteration, return true if successful and store result in e
    bool loop_next() final
    {
        ++cnt;
        // swap u rows in HST & bring in echelon form
        HST.update(u);
        // find all subISD solutions
        subISDT->solve();
        return !sol.empty();
    }

    // run loop until a solution is found
    void solve() final
    {
        prepare_loop();
        while (!loop_next())
            ;
    }

    vec get_solution() const
    {
        vec ret;
        ret.resize(HST.HT().rows());
        for (unsigned i = 0; i < sol.size(); ++i)
            ret.setbit(sol[i]);
        return ret;
    }
    size_t get_cnt() const { return cnt; }
    
private:
    subISDT_t* subISDT;
    HST_ISD_form_t<_bit_alignment> HST;
    vec C;
    vec_view C2padded, C1restpadded;
    
    std::vector<uint32_t> sol;
    
    size_t n, k, w, l, u;
    
    size_t cnt;
};











static inline size_t get_scratch(size_t k, size_t sz)
{
    return ((k+sz-1)/sz) * sz - k;
}

class Solution : public std::exception {
    vec sol;

    public:
        Solution(const cvec_view& sol_)
            : std::exception(), sol(sol_)
        {
        }

    cvec_view get_solution() { return sol; }
};

inline bool check_solution(const mat_view& H01T_view, const vec_view& S0, const std::vector<uint32_t>& perm, size_t w, const std::vector<uint32_t>& E1_sparse, size_t w1)
{
    vec E0(S0);
    for( auto i : E1_sparse ) {
        E0 ^= H01T_view[i];
    }
    if (hammingweight(E0)<= w-w1-E1_sparse.size())
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
Therefore it is derived from ISD_API_target.
The subISD algorithm class is a template parameter defaulted to ISD_API_exhaustive for flexibility:
This enables it to work with any ISD_API_exhaustive derived object.
However, for optimized performance it is also possible to pass the exhaustive-ISD algorithm type directly
and remove the overhead of virtualized function calls.
In that case, be sure to make the crucial functions as final.
*/

template<typename subISD_t = ISD_API_exhaustive_sparse_t>
class ISD_single_generic: public ISD_API_single
{
private:
    mat H;
    mat H01T;
    vec S;
    vec solution;

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

        solution = vec(n);
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

    cvec_view get_solution() {
        return solution;
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
        {
            --cnt;
            return true;
        }

        H01T_S_view.transpose(H01_S_view);
        try
        {
            subISD->solve();
        } catch(Solution& sol)
        {
            std::cerr << "ISD_single_generic found solution after " << cnt << " iterations" << std::endl;
            std::cerr << sol.get_solution() << std::endl;
            // todo: free memory
            solution.copy(sol.get_solution());
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

    int get_cnt() 
    {
        return cnt;
    }

    subISD_t* subISD;
};

MCCL_END_NAMESPACE

#endif
