// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_ALGORITHM_ISDGENERIC_HPP
#define MCCL_ALGORITHM_ISDGENERIC_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/core/matrix_isdform.hpp>
#include <mccl/tools/statistics.hpp>

MCCL_BEGIN_NAMESPACE

struct ISD_generic_config_t
{
    const std::string modulename = "isd_generic";
    const std::string description = "ISD generic configuration";
    const std::string manualstring =
        "ISD generic:\n"
        "\tInput: (n-k) x n matrix H, (n-k) vector S, max error weight w, subISD\n"
        "\tParameters:\n"
        "\t\tl: determines the number of rows of H2 and S2\n"
        "\t\tu: the number of echelon columns and ISD columns to swap per iteration\n"
        "\t\tupdatetype: the swap strategy:\n"
        "\t\t\t1: u times: swap random echelon & ISD column\n"
        "\t\t\t2: swap u random distinct echelon cols with u random (non-distinct) ISD cols\n"
        "\t\t\t3: swap u random distinct echelon cols with u random distinct ISD cols\n"
        "\t\t\t4: like 3, ensure further distinctness per batch of (n-k)*k/n choices\n"
        "\t\t\t12: like 2, but use round-robin echelon column selection\n"
        "\t\t\t13: like 3, but use round-robin echelon column selection\n"
        "\t\t\t14: like 4, but use round-robin echelon column selection\n"
        "\t\t\t10: use round-robin echelon column selection & round-robin scanning for ISD column with pivot bit set\n"
        "\tAlgorithm:\n"
        "\t\tApply random column permutation of H\n"
        "\t\tPerform echelonization on (H|S) over (n-k-l) rows:\n"
        "\t\t\tH|S = (I H1 S1)\n"
        "\t\t\t      (0 H2 S2)\n"
        "\t\tRepeatedly:\n"
        "\t\t\tCall subISD(H2, S2, w)\n"
        "\t\t\tCheck every output solution and quit when a proper solution is found\n"
        "\t\t\tRandomly swap u echelon columns with u ISD columns\n"
        "\t\t\tPerform echelonization over (n-k-l) rows\n"
        ;

    unsigned int l = 0;
    int u = -1;
    unsigned int updatetype = 14;
    bool verify_solution = true;

    template<typename Container>
    void process(Container& c)
    {
        c(l, "l", 0, "ISD parameter l");
        c(u, "u", -1, "Number of columns to swap per iteration (-1=auto)");
        c(updatetype, "updatetype", 14, "Update strategy type: 1, 2, 3, 4, 12, 13, 14, 10");
        c(verify_solution, "verifysolution", true, "Set verification of solutions");
    }
};

// global default. modifiable.
// at construction of ISD_generic the current global default values will be loaded
extern ISD_generic_config_t ISD_generic_config_default;



// implementation of ISD_single_generic that can be instantiated with any subISD
// based on common view on transposed H
// will use reverse column ordering for column reduction on Htransposed (instead of row reduction on H)
//
// HT = ( 0   RI  ) where RI is the reversed identity matrix RI with 1's on the anti-diagonal
//      ( H2T H1T )
//
// this makes it easy to include additional columns of H1T together with H2T to subISD

template<typename subISDT_t = subISDT_API, size_t _bit_alignment = 256, bool _masked = false>
class ISD_generic
    final : public syndrome_decoding_API
{
public:
    typedef typename subISDT_t::callback_t callback_t;

    static const size_t bit_alignment = _bit_alignment;
    
    typedef uint64_block_t<bit_alignment>  this_block_t;
    typedef block_tag<bit_alignment,_masked> this_block_tag;

    ISD_generic(subISDT_t& sI)
        : subISDT(&sI), config(ISD_generic_config_default), stats("ISD-generic")
    {
        n = k = w = 0;
    }

    ~ISD_generic()
    {
    }

    void load_config(const configmap_t& configmap)
    {
        mccl::load_config(config, configmap);
    }
    void save_config(configmap_t& configmap)
    {
        mccl::save_config(config, configmap);
    }

    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const cmat_view& _H, const cvec_view& _S, unsigned int _w)
    {
        stats.cnt_initialize.inc();
        // set parameters according to current config
        l = config.l;
        u = config.u;
        update_type = config.updatetype;

        n = _H.columns();
        k = n - _H.rows();
        w = _w;
        Horg.reset(_H);
        Sorg.reset(_S);
        HST.reset(_H, _S, l);

        C.resize(HST.S().columns());
        
        blocks_per_row = HST.H12T().row_blocks();
        block_stride = HST.H12T().block_stride();
        H12T_blockptr = HST.H12T().block_ptr();
        S_blockptr = HST.S().block_ptr();
        C_blockptr = C.block_ptr();
        
        sol.clear();
        solution = vec();
    }

    // probabilistic preparation of loop invariant
    void prepare_loop(bool _benchmark = false)
    {
        stats.cnt_prepare_loop.inc();
        benchmark = _benchmark;
        subISDT->initialize(HST.H12T(), HST.H2T().columns(), HST.S2(), w, make_ISD_callback(*this), this);
    }

    // perform one loop iteration, return true if successful and store result in e
    bool loop_next()
    {
        stats.cnt_loop_next.inc();
        // swap u rows in HST & bring in echelon form
        HST.update(u, update_type);
        // find all subISD solutions
        subISDT->solve();
        return !sol.empty();
    }

    // run loop until a solution is found
    void solve()
    {
        stats.cnt_solve.inc();
        prepare_loop();
        while (!loop_next())
            ;
        stats.refresh();
    }

    cvec_view get_solution() const
    {
        return cvec_view(solution);
    }

    // retrieve statistics
    decoding_statistics get_stats() const
    {
        return stats;
    };



    bool check_solution()
    {
        stats.cnt_check_solution.inc();
        if (solution.columns() == 0)
            throw std::runtime_error("ISD_generic::check_solution: no solution");
        return check_SD_solution(Horg, Sorg, w, solution);
    }
    

    // callback function
    inline bool callback(const uint32_t* begin, const uint32_t* end, unsigned int w1partial)
    {
            stats.cnt_callback.inc();
            // weight of solution consists of w2 (=end-begin) + w1partial (given) + w1rest (computed below)
            size_t wsol = w1partial + (end - begin);
            if (wsol > w)
                return true;

            wsol = end - begin;
            if (begin == end)
            {
                // case selection size 0
                auto Sptr = S_blockptr;
                auto Cptr = C_blockptr;
                for (unsigned i = 0; i < blocks_per_row; ++i,++Sptr,++Cptr)
                {
                    wsol += hammingweight( *Cptr = *Sptr );
                    if (wsol > w)
                        return true;
                }
            } else if (begin == end-1)
            {
                // case selection size 1
                auto Sptr = S_blockptr;
                auto Cptr = C_blockptr;
                auto HTrowptr = H12T_blockptr + block_stride*(*begin);
                for (unsigned i = 0; i < blocks_per_row; ++i,++Cptr,++Sptr,++HTrowptr)
                {
                    wsol += hammingweight( *Cptr = *Sptr ^ *HTrowptr );
                    if (wsol > w)
                        return true;
                }
            } else {
                // case selection size >= 2
                auto Cptr = C_blockptr;
                auto Sptr = S_blockptr;
                for (unsigned i = 0; i < blocks_per_row; ++i,++Cptr,++Sptr)
                {
                    const uint32_t* p = begin;
                    *Cptr = *Sptr ^ *(H12T_blockptr + block_stride*(*p) + i);
                    for (++p; p != end-1; ++p)
                    {
                        *Cptr = *Cptr ^ *(H12T_blockptr + block_stride*(*p) + i);
                    }
                    wsol += hammingweight( *Cptr = *Cptr ^ *(H12T_blockptr + block_stride*(*p) + i) );
                    if (wsol > w)
                        return true;
                }
            }

            // this should be a correct solution at this point
            if (benchmark)
                return true;

            // 3. construct full solution on echelon and ISD part
            if (wsol != (end-begin) + hammingweight(C))
                throw std::runtime_error("ISD_generic::callback: internal error 1: w1partial is not correct?");
            sol.clear();
            for (auto p = begin; p != end; ++p)
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
            if (config.verify_solution && !check_solution())
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
    HST_ISD_form_t<_bit_alignment,_masked> HST;

    // temporary vector to compute sum of syndrome and H columns
    vec_t<this_block_tag> C;
    
    // block pointers to H12T, S and C
    size_t block_stride, blocks_per_row;
    const this_block_t* H12T_blockptr;
    const this_block_t* S_blockptr;
    this_block_t* C_blockptr;
    
    
    // parameters
    ISD_generic_config_t config;

    size_t n, k, w;
    unsigned int l;
    int u;
    unsigned int update_type;
    bool benchmark;
    
    // iteration count
    decoding_statistics stats;
};

MCCL_END_NAMESPACE

#endif
