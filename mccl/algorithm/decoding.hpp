// generic decoding API, virtual class from which decoding algorithms can be derived

#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/core/matrix.hpp>
#include <functional>
#include <stdlib.h>

MCCL_BEGIN_NAMESPACE

// virtual base class: interface for ISD for a given syndrome target
template<typename data_t>
class ISD_API_target
{
public:
    // pass parameters to actual object
    //virtual void configure(const parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    virtual void initialize(const matrix_ref_t<data_t>& H0, const vector_ref_t<data_t>& s0) = 0;
    
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

// virtual base class: interface for exhaustive ISD returning all solutions
template<typename data_t>
class ISD_API_exhaustive
{
public:
    // pass parameters to actual object
    //virtual void configure(const parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    virtual void initialize(const matrix_ref_t<data_t>& H0, const vector_ref_t<data_t>& s0, std::function<void,vector_ref_t<data_t>&>& callback) = 0;
    
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
class ISD_target_generic: public ISD_API_target<data_t>
{
public:
    ISD_target_generic(subISD_t& sI)
        : subISD(&sI)
    {
    }
    
    // pass parameters to actual object
    //virtual void configure(const parameters_t& params) = 0;
    
    // deterministic initialization for given parity check matrix H0 and target syndrome s0
    void initialize(const matrix_ref_t<data_t>& H0, const vector_ref_t<data_t>& s0) final
    {
        H = H0;
        s = s0;
    }
    
    // probabilistic preparation of loop invariant
    void prepare_loop() final
    {
    }
    
    // perform one loop iteration, return true if successful and store result in e
    bool loop_next() final
    {
    }
    
    // run loop until a solution is found
    /*void solve()
    {
        prepare_loop();
        while (!loop_next())
            ;
    }*/
    
    subISD_t* subISD;
    
    vector_t<data_t> e;
    matrix_t<data_t> H;
    vector_t<data_t> s;
};

MCCL_END_NAMESPACE
