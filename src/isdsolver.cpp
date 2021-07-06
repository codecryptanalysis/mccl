#include <iostream>
#include <unistd.h>

#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/LB.hpp>

#include <mccl/contrib/program_options.hpp>

namespace po = program_options;

using namespace mccl;

// we should probably move this function in the tools
std::size_t binomial(std::size_t k, std::size_t N)
{
    if (k > N) return 0;
    std::size_t r = 1;
    if (k > N-k)
        k = N-k;
    for (unsigned i = 0; i < k; ++i)
    {
        r *= (N-i);
        r /= (i+1);
    }
    return r;
}

template<typename subISD_t = ISD_API_exhaustive_sparse_t>
int run_subISD(mat_view &H, vec_view &S, size_t w) {
  subISD_t subISD;
  ISD_single_generic<subISD_t> ISD_single(subISD);
  ISD_single.initialize(H, S, w);
  ISD_single.solve();
  std::cout << "Solution found:" << '\n';
  std::cout << ISD_single.get_solution() << '\n';
  return ISD_single.get_cnt();
}

template<typename subISDT_t = ISD_API_exhaustive_transposed_sparserange_t>
int run_subISDT(mat_view& H, vec_view& S, size_t w, size_t l, size_t p, size_t u)
{
  subISDT_t subISDT;
  subISDT.configure(p);
  ISD_single_generic_transposed<subISDT_t> ISD_single(subISDT);
  ISD_single.configure(l, u);
  ISD_single.initialize(H, S, w);
  ISD_single.solve();
  std::cout << "Solution found:\n";
  std::cout << ISD_single.get_solution() << std::endl;
  return ISD_single.get_cnt();
}

int main(int argc, char *argv[])
{
try
{
    std::string filepath, algo;
    size_t trials = 1;
    size_t n = 0, k = 0, w = 0, l = 0, u = 1, p = 3;
    
    po::options_description allopts, cmdopts("Command options"), opts("Extra options");
    // These are the core commands, you need at least one of these
    cmdopts.add_options()
      ("help,h", "Show options")
      ("file,f", po::value<std::string>(&filepath), "Specify input file")
      ("gen,g", "Generate random ISD instances")
      ;
    // these are other configuration options
    opts.add_options()
      ("algo,a", po::value<std::string>(&algo)->default_value("P"), "Specify algorithm: P, LB, TP, TLB")
      ("trials,t", po::value<size_t>(&trials)->default_value(1), "Number of ISD trials")
      ("genunique", "Generate unique decoding instance")
      ("genrandom", "Generate random decoding instance")
      ("n", po::value<size_t>(&n), "Code length")
      ("k", po::value<size_t>(&k), "Code dimension")
      ("w", po::value<size_t>(&w), "Error weight")
      ("l", po::value<size_t>(&l)->default_value(0), "H2 rows")
      ("p", po::value<size_t>(&p)->default_value(3), "subISD parameter p")
      ("u", po::value<size_t>(&u)->default_value(1), "I column swaps per iteration")
      ;
    allopts.add(cmdopts).add(opts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allopts, false, true), vm);
    po::notify(vm);
    
    if (vm.count("help") || vm.count("file")+vm.count("gen")==0)
    {
      std::cout << cmdopts << opts;
      return 0;
    }
    if (algo != "P" && algo != "LB" && algo != "TP" && algo != "TLB")
    {
      std::cout << "Unknown algorithm: " << algo << std::endl;
      return 1;
    }
    if (vm.positional.size() > 0)
      n = vm.positional[0].as<size_t>();
    if (vm.positional.size() > 1)
      k = vm.positional[1].as<size_t>();
    if (vm.positional.size() > 2)
      w = vm.positional[2].as<size_t>();
    if (vm.positional.size() > 3)
      l = vm.positional[3].as<size_t>();
    if (vm.positional.size() > 4)
      u = vm.positional[4].as<size_t>();
    if (vm.positional.size() > 5)
    {
      std::cout << "Unknown option: " << vm.positional[5].as<std::string>() << std::endl;
      return 1;
    }

    Parser parse;
    if (filepath != "")
    {
      std::cout << "Parsing instance " << filepath << '\n';
      bool b = parse.load_file(filepath);
      if (!b) {
        std::cout << "Parsing instance failed" << '\n';
        return 1;
      }
      n = parse.get_n();
      k = parse.get_k();
      w = parse.get_w();
    } else
    {
      if (n == 0 || k >= n || w >= n)
      {
        std::cout << "Bad input parameters: n=" << n << ", k=" << k << ", w=" << w << std::endl;
        return 1;
      }
      parse.random_SD(n, k, w);
    }
    if (l > n-k)
      l = n-k;
    if (u < 1)
      u = 1;

    std::cout << "n=" << n << ", k=" << k << ", w=" << w << " | algo=" << algo << ", l=" << l << ", p=" << p << ", u=" << u << " | trials=" << trials << std::endl;

    mat_view H = parse.get_H();
    vec_view S = parse.get_S();
    int c = 0;
    for (size_t i = 0; i < trials; ++i)
    {
      if(i > 0 && vm.count("gen")) {
        parse.regenerate();
        H.reset(parse.get_H());
        S.reset(parse.get_S());
      }
      if (algo=="P")
        c += run_subISD<subISD_prange>(H,S,w);
      else if (algo=="LB")
        c += run_subISD<subISD_LB>(H,S,w);
      else if (algo=="TP")
        c += run_subISDT<subISDT_prange>(H,S,w,l,p,u);
      else if (algo=="TLB")
        c += run_subISDT<subISDT_LB>(H,S,w,l,p,u);
    }

    float avg_cnt = float(c) / float(trials);
    float inv_avg_cnt = float(1) / avg_cnt;
    std::cout << "Average number of iterations: " << avg_cnt << '\n';
    std::cout << "Inverse of average number of iterations: " << inv_avg_cnt << '\n';
    return 0;
} catch (std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
} catch (...) {
    std::cerr << "Caught unknown exception!" << std::endl;
    return 1;
}
}
