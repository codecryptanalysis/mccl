// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_ALGORITHM_DECODING_HPP
#define MCCL_ALGORITHM_DECODING_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix.hpp>

MCCL_BEGIN_NAMESPACE

class syndrome_decoding_API;

struct syndrome_decoding_problem
{
    mat H;
    vec S;
    unsigned int w;
    
    bool check_solution(const cvec_view& E) const;
    
    template<typename ISD_t>
    vec solve(ISD_t& ISD);
};

bool check_SD_solution(const cmat_view& H, const cvec_view& S, unsigned int w, const cvec_view& E);


// virtual base class: interface to find a single solution for syndrome decoding
class syndrome_decoding_API
{
public:
    // virtual destructor, so we can properly delete a derived class through its base class pointer
    virtual ~syndrome_decoding_API() {}
    
    // pass parameters to actual object
    // if called then it must be called before initialize
    //virtual void configure(parameters_t& params) = 0;

    // deterministic initialization for given parity check matrix H and target syndrome S
    virtual void initialize(const cmat_view& H, const cvec_view& S, unsigned int w) = 0;
    virtual void initialize(const syndrome_decoding_problem& SD)
    {
        initialize(SD.H, SD.S, SD.w);
    }

    // probabilistic preparation of loop invariant
    virtual void prepare_loop() = 0;

    // perform one loop iteration
    // return true to continue loop (no solution has been found yet)
    virtual bool loop_next() = 0;

    // run loop until a solution is found
    virtual void solve()
    {
        prepare_loop();
        while (loop_next())
            ;
    }
    
    // retrieve solution if any
    virtual cvec_view get_solution() const = 0;
};

template<typename ISD_t = syndrome_decoding_API>
vec solve_syndrome_decoding(ISD_t& ISD, const cmat_view& H, const cvec_view& S, unsigned int w)
{
    ISD.initialize(H, S, w);
    ISD.solve();
    return ISD.get_solution();
}

template<typename ISD_t>
vec syndrome_decoding_problem::solve(ISD_t& ISD)
{
    ISD.initialize(H, S, w);
    ISD.solve();
    return ISD.get_solution();
}




// default callback types for subISD to the main ISD
// takes pointer to object & representation of H12T rows & and w1partial
// returns true while enumeration should be continued, false when it should be stopped
typedef bool (*ISD_callback_t)(void*, const uint32_t*, const uint32_t*, unsigned int);

// generic callback functions that can be instantiated for any subISD class
template<typename subISD_t>
bool ISD_callback(void* ptr, const uint32_t* begin, const uint32_t* end, unsigned int w)
{
    return reinterpret_cast<subISD_t*>(ptr)->callback(begin, end, w);
}

template<typename subISD_t>
ISD_callback_t make_ISD_callback(const subISD_t&)
{
    return ISD_callback<subISD_t>;
}

// virtual base class: interface for 'exhaustive' subISD returning as many solutions as efficiently as possible
// note that H is given transposed for better efficiency between main_ISD and sub_ISD
class subISDT_API
{
public:
    // virtual destructor, so we can properly delete a derived class through its base class pointer
    virtual ~subISDT_API() {}
    
    typedef ISD_callback_t callback_t;
    // pass parameters to actual object
    // if called then it must be called before initialize
    //virtual void configure(parameters_t& params) = 0;

    // deterministic initialization for given parity check matrix H and target syndrome s
    virtual void initialize(const cmat_view& H12T_padded, size_t H2T_columns, const cvec_view& S, unsigned int w, callback_t callback, void* ptr = nullptr) = 0;

    // preparation of loop invariant
    virtual void prepare_loop() = 0;

    // perform one loop iteration
    // pass all solution through callback
    // return true to continue loop (no main ISD solution has been found & subISD search space not exhausted yet)
    virtual bool loop_next() = 0;

    // run loop and pass all solutions through callback
    virtual void solve()
    {
        prepare_loop();
        while (loop_next())
            ;
    }
};

MCCL_END_NAMESPACE

#endif
