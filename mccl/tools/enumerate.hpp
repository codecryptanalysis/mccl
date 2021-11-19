#ifndef MCCL_TOOLS_ENUMERATE_HPP
#define MCCL_TOOLS_ENUMERATE_HPP

#include <mccl/config/config.hpp>

MCCL_BEGIN_NAMESPACE

template<typename Idx = uint16_t>
class enumerate_t
{
public:
    typedef Idx index_type;

    // if return type of f is void always return true (continue enumeration)
    template<typename F, typename ... Args>
    inline auto call_function(F&& f, Args&& ... args)
        -> typename std::enable_if<std::is_same<void,decltype(f(std::forward<Args>(args)...))>::value, bool>::type
    {
        f(std::forward<Args>(args)...);
        return true;
    }
    
    // if return type of f is bool return output of f (true to continue enumeration, false to stop)
    template<typename F, typename ... Args>
    inline auto call_function(F&& f, Args&& ... args)
        -> typename std::enable_if<std::is_same<bool,decltype(f(std::forward<Args>(args)...))>::value, bool>::type
    {
        return f(std::forward<Args>(args)...);
    }


    template<typename T, typename F>
    void enumerate1_val(const T* begin, const T* end, F&& f)
    {
        for (; begin != end; ++begin)
            if (!call_function(f,*begin))
                return;
    }

    template<typename T, typename F>
    void enumerate12_val(const T* begin, const T* end, F&& f)
    {
        if (end-begin < 2)
            return;
        for (; begin != end; )
        {
            auto val = *begin;
            if (!call_function(f,val))
                return;
            for (auto it = ++begin; it != end; ++it)
                if (!call_function(f,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate2_val(const T* begin, const T* end, F&& f)
    {
        if (end-begin < 2)
            return;
        for (; begin != end; )
        {
            auto val = *begin;
            for (auto it = ++begin; it != end; ++it)
                if (!call_function(f,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate3_val(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        if (count < 3)
            return;
        auto mid = begin + (count/2);
        // try to have as large as possible inner loop
        // first half loop on 2nd value: use 3rd value in innerloop
        for (auto it2 = begin+1; it2 != mid; ++it2)
        {
            for (auto it1 = begin; it1 != it2; ++it1)
            {
                auto val = *it2 ^ *it1;
                for (auto it3 = it2+1; it3 != end; ++it3)
                    if (!call_function(f,val ^ *it3))
                        return;
            }
        }
        // second half loop on 2nd value: use 1st value in innerloop
        for (auto it2 = mid; it2 != end-1; ++it2)
        {
            for (auto it3 = it2+1; it3 != end; ++it3)
            {
                auto val = *it2 ^ *it3;
                for (auto it1 = begin; it1 != it2; ++it1)
                    if (!call_function(f,val ^ *it1))
                        return;
            }
        }
    }
    
    template<typename T, typename F>
    void enumerate4_val(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        if (count < 4)
            return;
        auto mid = begin + std::min<size_t>(32, count/3);
        // try to have as large as possible inner loop
        // first half iteration: loop 2nd value until middle:
        // iterate on 3rd+4th value in inner loop, 1st in outer loop
        for (auto it2 = begin+1; it2 != mid; ++it2)
        {
            for (auto it1 = begin; it1 != it2; ++it1)
            {
                for (auto it3 = it2+1; it3 != end-1; ++it3)
                {
                    auto val = *it1 ^ *it2 ^ *it3;
                    for (auto it4 = it3+1; it4 != end; ++it4)
                    {
                        if (!call_function(f, val ^ *it3))
                            return;
                    }
                }
            }
        }
        // second half iteration: loop 2nd value from middle to end
        // iterate on 1st in inner loop, 3rd+4th value in outer loop
        for (auto it2 = mid; it2 != end-2; ++it2)
        {
            for (auto it3 = it2+1; it3 != end-1; ++it3)
            {
                for (auto it4 = it3+1; it4 != end; ++it4)
                {
                    auto val = *it4 ^ *it2 ^ *it3;
                    for (auto it1 = begin; it1 != it2; ++it1)
                    {
                        if (!call_function(f, val ^ *it3))
                            return;
                    }
                }
            }
        }
    }
    
    template<typename T, typename F>
    void enumerate_val(const T* begin, const T* end, size_t p, F&& f)
    {
        switch (p)
        {
            default: throw std::runtime_error("enumerate::enumerate_val: only 1 <= p <= 4 supported");
            case 4:
                enumerate4_val(begin,end,f);
                __attribute__ ((fallthrough));
            case 3:
                enumerate3_val(begin,end,f);
                __attribute__ ((fallthrough));
            case 2:
                enumerate12_val(begin,end,f);
                return;
            case 1:
                enumerate1_val(begin,end,f);
        }
    }


    template<typename T, typename F>
    void enumerate1(const T* begin, const T* end, F&& f)
    {
        idx[0] = 0;
        for (; begin != end; ++begin,++idx[0])
            if (!call_function(f,idx+0,idx+1,*begin))
                return;
    }

    template<typename T, typename F>
    void enumerate12(const T* begin, const T* end, F&& f)
    {
        idx[0] = 0;
        for (; begin != end; ++idx[0])
        {
            auto val = *begin;
            if (!call_function(f,idx+0,idx+1,val))
                return;
            idx[1] = idx[0]+1;
            for (auto it = ++begin; it != end; ++it,++idx[1])
                if (!call_function(f,idx+0,idx+2,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate2(const T* begin, const T* end, F&& f)
    {
        idx[0] = 0;
        for (; begin != end; ++idx[0])
        {
            auto val = *begin;
            idx[1] = idx[0]+1;
            for (auto it = ++begin; it != end; ++it,++idx[1])
                if (!call_function(f,idx+0,idx+2,val ^ *it))
                    return;
        }
    }

    template<typename T, typename F>
    void enumerate3(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        auto mid = begin + (count/2);
        idx[1] = 1;
        // try to have as large as possible inner loop
        // first half loop on 2nd value: use 3rd value in innerloop
        for (auto it2 = begin+1; it2 != mid; ++it2,++idx[1])
        {
            idx[0] = 0;
            for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
            {
                auto val = *it2 ^ *it1;
                idx[2] = idx[1]+1;
                for (auto it3 = it2+1; it3 != end; ++it3,++idx[2])
                    if (!call_function(f,idx+0, idx+3, val ^ *it3))
                        return;
            }
        }
        // second half loop on 2nd value: use 1st value in innerloop
        for (auto it2 = mid; it2 != end-1; ++it2,++idx[1])
        {
            idx[2] = idx[1]+1;
            for (auto it3 = it2+1; it3 != end; ++it3,++idx[2])
            {
                auto val = *it2 ^ *it3;
                idx[0] = 0;
                for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
                    if (!call_function(f,idx+0, idx+3, val ^ *it1))
                        return;
            }
        }
    }

    template<typename T, typename F>
    void enumerate4(const T* begin, const T* end, F&& f)
    {
        size_t count = end-begin;
        if (count < 4)
            return;
        auto mid = begin + std::min<size_t>(32, count/3);
        idx[1] = 1;
        // try to have as large as possible inner loop
        // first half iteration: loop 2nd value until middle:
        // iterate on 3rd+4th value in inner loop, 1st in outer loop
        for (auto it2 = begin+1; it2 != mid; ++it2,++idx[1])
        {
            idx[0] = 0;
            for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
            {
                idx[2] = idx[1]+1;
                for (auto it3 = it2+1; it3 != end-1; ++it3,++idx[2])
                {
                    auto val = *it1 ^ *it2 ^ *it3;
                    idx[3] = idx[2]+1;
                    for (auto it4 = it3+1; it4 != end; ++it4,++idx[3])
                    {
                        if (!call_function(f, idx+0, idx+4, val ^ *it3))
                            return;
                    }
                }
            }
        }
        // second half iteration: loop 2nd value from middle to end
        // iterate on 1st in inner loop, 3rd+4th value in outer loop
        for (auto it2 = mid; it2 != end-2; ++it2)
        {
            idx[2] = idx[1]+1;
            for (auto it3 = it2+1; it3 != end-1; ++it3,++idx[2])
            {
                idx[3] = idx[2]+1;
                for (auto it4 = it3+1; it4 != end; ++it4,++idx[3])
                {
                    auto val = *it4 ^ *it2 ^ *it3;
                    idx[0] = 0;
                    for (auto it1 = begin; it1 != it2; ++it1,++idx[0])
                    {
                        if (!call_function(f, idx+0, idx+4, val ^ *it3))
                            return;
                    }
                }
            }
        }
    }
    
    template<typename T, typename F>
    void enumerate(const T* begin, const T* end, size_t p, F&& f)
    {
        switch (p)
        {
            default: throw std::runtime_error("enumerate::enumerate: only 1 <= p <= 4 supported");
            case 4:
                enumerate4(begin,end,f);
                __attribute__ ((fallthrough));
            case 3:
                enumerate3(begin,end,f);
                __attribute__ ((fallthrough));
            case 2:
                enumerate12(begin,end,f);
                return;
            case 1:
                enumerate1(begin,end,f);
                return;
        }
    }

    index_type idx[16];
};

MCCL_END_NAMESPACE

#endif
