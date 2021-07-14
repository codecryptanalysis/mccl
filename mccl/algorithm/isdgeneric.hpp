// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_ALGORITHM_ISDGENERIC_HPP
#define MCCL_ALGORITHM_ISDGENERIC_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/core/matrix_isdform.hpp>

MCCL_BEGIN_NAMESPACE

// implementation of ISD_single_generic that can be instantiated with any subISD
// based on common view on transposed H
// will use reverse column ordering for column reduction on Htransposed (instead of row reduction on H)
//
// HT = ( 0   RI  ) where RI is the reversed identity matrix RI with 1's on the anti-diagonal
//      ( H2T H1T )
//
// this makes it easy to include additional columns of H1T together with H2T to subISD

template<typename subISDT_t = subISDT_API, size_t _bit_alignment = 64>
class ISD_generic
    : public syndrome_decoding_API
{
public:
    typedef typename subISDT_t::callback_t callback_t;
    
    static const size_t bit_alignment = _bit_alignment;
    typedef aligned_tag<bit_alignment> this_aligned_tag;

    ISD_generic(subISDT_t& sI)
        : subISDT(&sI)
    {
        n = k = w = 0;
        l = 0; // ISD form parameter
        u = -1; // how many rows to swap each iteration. max n - k - l
        update_type = 14;
    }
    
    ~ISD_generic() final
    {
    }

    // pass parameters to actual object
    // // virtual void configure(parameters_t& params) = 0;
    // if called then it must be called before initialize
    void configure(size_t _l = 0, int _u = -1, int _update_type = 14)
    {
        l = _l;
        u = _u;
        update_type = _update_type;
    }

    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const cmat_view& _H, const cvec_view& _S, unsigned int _w) final
    {
        n = _H.columns();
        k = n - _H.rows();
        w = _w;
        Horg.reset(_H);
        Sorg.reset(_S);
        HST.reset(_H, _S, l);

        C.resize(HST.Spadded().columns());
        C2padded.reset(C.subvector(0, HST.S2padded().columns()));
        C1restpadded.reset(C.subvector(HST.S2padded().columns(), HST.S1restpadded().columns()));
        
        sol.clear();
        solution = vec();
        cnt = 0;
    }

    // probabilistic preparation of loop invariant
    void prepare_loop() final
    {
        subISDT->initialize(HST.H2Tpadded(), HST.H2T().columns(), HST.S2padded(), w, 
            make_ISD_callback(*this), this);
    }

    // perform one loop iteration, return true if successful and store result in e
    bool loop_next() final
    {
        ++cnt;
        // swap u rows in HST & bring in echelon form
        HST.update(u, update_type);
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

    cvec_view get_solution() const final
    {
        return solution;
    }





    bool check_solution() const
    {
        if (solution.columns() == 0)
            throw std::runtime_error("ISD_generic::check_solution: no solution");
        return check_SD_solution(Horg, Sorg, w, solution);
    }
    
    size_t get_cnt() const { return cnt; }




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
                
            // this should be a correct solution at this point

            // 3. construct full solution on E1 and E2 part
            C.copy(HST.Spadded(), this_aligned_tag());
            for (p = begin; p != end; ++p)
                C.vxor(HST.H12Tpadded()[*p], this_aligned_tag());
            if (wsol != (end-begin) + hammingweight(C, this_aligned_tag()))
                throw std::runtime_error("ISD_generic::callback: internal error 1: w1partial is not correct?");
            sol.clear();
            for (p = begin; p != end; ++p)
                sol.push_back(HST.permutation( HST.echelonrows() + *p ));
            for (size_t c = 0; c < HST.HT().columns(); ++c)
            {
                if (C[c] == false)
                    continue;
                if (c < HST.H2T().columns())
                    throw std::runtime_error("ISD_generic::callback: internal error 2: H2T combination non-zero!");
                sol.push_back(HST.permutation( HST.HT().columns() - 1 - c ));
            }
            solution = vec(HST.HT().rows());
            for (unsigned i = 0; i < sol.size(); ++i)
                solution.setbit(sol[i]);

            // TODO: make option to verify solutions
            if (!check_solution())
                throw std::runtime_error("ISD_generic::callback: internal error 3: solution is incorrect!");
            return false;
    }
    
private:
    subISDT_t* subISDT;

    // original parity check matrix H^T and syndrome S
    cmat_view Horg;
    cvec_view Sorg;
    // solution with respect to original H
    std::vector<uint32_t> sol;
    vec solution;

    // maintains (U(H|S)P)^T in ISD form for random column permutations P
    HST_ISD_form_t<_bit_alignment> HST;

    // temporary vector to compute sum of syndrome and H columns
    vec C;
    vec_view C2padded, C1restpadded;
    
    // parameters
    size_t n, k, w, l;
    int u;
    int update_type;
    
    // iteration count
    size_t cnt;
};

MCCL_END_NAMESPACE

#endif
