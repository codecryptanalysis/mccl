#ifndef MCCL_CONFIG_UTILS_HPP
#define MCCL_CONFIG_UTILS_HPP

#include <mccl/config/config.hpp>

#include <mccl/contrib/program_options.hpp>
#include <mccl/contrib/string_algo.hpp>

#include <string>
#include <iostream>


MCCL_BEGIN_NAMESPACE

namespace po = program_options;
namespace sa = string_algo;

/* Helper structs and functions for configuration of submodules */

/* Helpers to collect submodule parameters into program options to parse */
struct options_description_insert_helper
{
    po::options_description* opts;
    template<typename T, typename T2>
    void operator()(T&, const std::string& valname, const T2&, const std::string& description)
    {
      // first check if option already exists (some subISDs have a common parameter name like 'p')
      for (auto& option: opts->_options)
      {
        if (valname.size() == 1 && option->shortopt == valname)
          return;
        if (option->longopt == valname)
          return;
      }
      // otherwise add option
      opts->add_options()
        (valname, po::value<T>(), description)
        ;
    }
    // special case for boolean
    void operator()(bool&, const std::string& valname, bool, const std::string& description)
    {
      std::string optname = valname;
      // first check if option already exists (some subISDs have a common parameter name like 'p')
      for (auto& option: opts->_options)
      {
        if (optname.size() == 1 && option->shortopt == optname)
          return;
        if (option->longopt == optname)
          return;
      }
      // otherwise add option
      opts->add_options()
        (optname, po::bool_switch(), description)
        ;
      opts->add_options()
        ("no-"+optname, po::bool_switch(), description)
        ;
    }
};
template<typename Configuration>
void options_description_insert(po::options_description& opts, Configuration& conf)
{
  options_description_insert_helper helper;
  helper.opts = &opts;
  conf.process(helper);
}



/* Helpers to print submodule parameters as program options */
struct get_options_description_helper
{
    po::options_description* opts;
    template<typename T, typename T2>
    void operator()(T&, const std::string& valname, const T2& defaultval, const std::string& description)
    {
      opts->add_options()
        (valname, po::value<T>()->default_value(defaultval), description)
        ;
    }
    // special case for boolean
    void operator()(bool&, const std::string& valname, bool defaultval, const std::string& description)
    {
      std::string optname = valname;
      // if defaultval = true then first show the "no-<valname>" option
      if (defaultval)
      {
        opts->add_options()
          ("no-"+optname, po::bool_switch(), description)
          ;
        opts->add_options()
          (optname, po::bool_switch(), "   (default)")
          ;
      }
      else
      {
        opts->add_options()
          (optname, po::bool_switch(), description)
          ;
        opts->add_options()
          ("no-"+optname, po::bool_switch(), "   (default)")
          ;
      }
    }
};
template<typename Configuration>
po::options_description get_options_description(Configuration& conf, unsigned line_length)
{
  po::options_description opts(conf.description, line_length, line_length/2);
  get_options_description_helper helper;
  helper.opts = &opts;
  conf.process(helper);
  return opts;
}



/* Helpers to print manual of submodules */
template<typename Configuration>
void print_manual(Configuration& conf)
{
  std::string manualstr = conf.manualstring;
  sa::replace_all(manualstr, std::string("\t"), std::string("  "));
  std::cout << "\n" << manualstr << "\n\n";
}

template<typename Module>
std::string get_configuration_str(Module& m)
{
  configmap_t configmap;
  m.save_config(configmap);
  std::string ret;
  for (auto& pv : configmap)
  {
    if (!ret.empty())
      ret.push_back(' ');
    ret.append(pv.first).append("=").append(pv.second);
  }
  return ret;
}

/* Helper API that does it all */
struct module_configuration_API
{
  virtual ~module_configuration_API() {}
  
  virtual void options_description_insert(po::options_description&) {}
  
  virtual void load_config(const configmap_t&) {}
  
  virtual po::options_description get_options_description(size_t line_length) { return po::options_description("", line_length, line_length/2); }
  
  virtual void print_manual() {}
};

template<typename config_t>
struct module_configuration_t final
  : public module_configuration_API
{
  config_t& config;
  
  module_configuration_t(config_t& _config)
    : config(_config)
  {
  }
  
  ~module_configuration_t()
  {
  }
  
  void options_description_insert(po::options_description& opts) final
  {
    mccl::options_description_insert(opts, config);
  }
  
  void load_config(const configmap_t& configmap) final
  {
    mccl::load_config(config, configmap);
  }
  
  po::options_description get_options_description(size_t line_length) final
  {
    return mccl::get_options_description(config, line_length);
  }
  
  void print_manual() final
  {
    mccl::print_manual(config);
  }
};

template<typename config_t>
module_configuration_t<config_t>* make_module_configuration(config_t& _config)
{
  return new module_configuration_t<config_t>(_config);
}


MCCL_END_NAMESPACE

#endif
