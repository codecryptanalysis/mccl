#ifndef MCCL_CONFIG_CONFIG_HPP
#define MCCL_CONFIG_CONFIG_HPP

#include <mccl/config/config.h>

#include <sstream>
#include <string>
#include <map>
#include <cstdint>
#include <gmpxx.h>

#ifndef MCCL_NAMESPACE
#define MCCL_NAMESPACE mccl
#endif

#ifndef MCCL_BEGIN_NAMESPACE
#define MCCL_BEGIN_NAMESPACE namespace MCCL_NAMESPACE {
#endif

#ifndef MCCL_END_NAMESPACE
#define MCCL_END_NAMESPACE }
#endif

#define MCCL_VECTOR_NO_SANITY_CHECKS 1
#define MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS 1
#define MCCL_VECTOR_ASSUME_NONEMPTY 1

// assumption. TODO: detect in configure
//#define MCCL_HAVE_CPU_COUNTERS

MCCL_BEGIN_NAMESPACE

using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;
using std::int_least8_t;
using std::uint_least8_t;
using std::int_least16_t;
using std::uint_least16_t;
using std::int_least32_t;
using std::uint_least32_t;
using std::int_least64_t;
using std::uint_least64_t;
using std::int_fast8_t;
using std::uint_fast8_t;
using std::int_fast16_t;
using std::uint_fast16_t;
using std::int_fast32_t;
using std::uint_fast32_t;
using std::int_fast64_t;
using std::uint_fast64_t;
using std::intptr_t;
using std::uintptr_t;
using std::size_t;

using bigint_t = ::mpz_class;
using bigfloat_t = ::mpf_class;

typedef std::map<std::string,std::string> configmap_t;

namespace detail
{

    // convert any type to/from std::string
    template<typename Type>
    std::string to_string(const Type& val)
    {
        std::stringstream strstr;
        strstr << val;
        return strstr.str();
    }
    template<typename Type>
    void from_string(const std::string& str, Type& ret)
    {
        std::stringstream strstr(str);
        strstr >> ret;
        if (!strstr)
            throw std::runtime_error("from_string(): could not parse string: " + str);
        // retrieving one more char should lead to EOF and failbit set
        strstr.get();
        if (!!strstr)
            throw std::runtime_error("from_string(): could not fully parse string: " + str);
    }

    // optimizations for std::string itself
    template<>
    inline std::string to_string(const std::string& str)
    {
        return str;
    }
    template<>
    inline void from_string(const std::string& str, std::string& ret)
    {
        ret = str;
    }

    struct load_configmap_helper
    {
        const configmap_t& configmap;
        load_configmap_helper(const configmap_t& _configmap)
            : configmap(_configmap)
        {}
        
        template<typename T, typename T2>
        void operator()(T& val, const std::string& valname, const T2& defaultval, const std::string&) const
        {
            auto it = configmap.find(valname);
            if (it == configmap.end())
                val = defaultval;
            else
                from_string(it->second, val);
        }
        // special case for bool
        void operator()(bool& val, const std::string& valname, bool defaultval, const std::string&) const
        {
            val = defaultval;
            auto it = configmap.find("no-"+valname);
            if (it != configmap.end())
            {
                if (it->second.empty())
                    val = false;
                else
                {
                    from_string(it->second, val);
                    val = !val;
                }
            }
            it = configmap.find(valname);
            if (it != configmap.end())
            {
                if (it->second.empty())
                    val = true;
                else
                    from_string(it->second, val);
            }
        }
    };
    
    struct save_configmap_helper
    {
        configmap_t& configmap;
        save_configmap_helper(configmap_t& _configmap)
            : configmap(_configmap)
        {}
        
        template<typename T, typename T2>
        void operator()(const T& val, const std::string& valname, const T2&, const std::string&)
        {
            configmap[valname] = to_string(val);
        }
    };
}

template<typename ConfigT>
void load_config(ConfigT& config, const configmap_t& configmap)
{
    detail::load_configmap_helper helper(configmap);
    config.process( helper );
}

template<typename ConfigT>
void save_config(ConfigT& config, configmap_t& configmap)
{
    detail::save_configmap_helper helper(configmap);
    config.process( helper );
}

MCCL_END_NAMESPACE

#endif
