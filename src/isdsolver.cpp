#include<iostream>
#include <unistd.h>

#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/LB.hpp>


using namespace mccl;

void usage() {
  std::cout << "Usage: isdsolver expects the following arguments " << '\n';
  std::cout << "Necessary arguments:" << '\n';
  std::cout << "\t -f\t to specify the path of the challenge instance file" << '\n';
  std::cout << "\t\t test files are available in 'tests/data/'" << '\n';
  std::cout << "Optional arguments:" << '\n';
  std::cout << "\t -a\t to specify the desired algorithm" << '\n';
  std::cout << "\t\t 'P' for PRANGE algorithm (default)" << '\n';
  std::cout << "\t\t 'LB' for LEE BRICKELL algorithm" << '\n';
  std::cout << "\t -n\t to repeat the algorithm several time" << '\n';
  std::cout << "\t\t expects an integer (default = 1)" << '\n';
}

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

int main(int argc, char *argv[])
{
    std::string path = "";
    std::string algo = "P";
    std::string N_str = "";
    int N = 1;

    int option;
    while((option = getopt(argc, argv, ":f:a:n:")) != -1){
      switch(option){
        case 'f':
          path = optarg;
          break;
        case 'a':
          algo = optarg;
          break;
        case 'n':
          N_str = optarg;
          break;
        case '?':
          printf("Unknown option: %c\n", optopt);
          break;
       }
    }

    if (algo=="P") {
      algo = "PRANGE";
      std::cout << "Using PRANGE algorithm" << '\n';
    }
    else if (algo=="LB") {
      algo = "LEE BRICKELL";
      std::cout << "Using LEE BRICKELL algorithm" << '\n';
    }
    else {
      std::cout << "Unknown algorithm " << algo << '\n';
      usage();
      return 1;
    }

    if(path==""){
      std::cout << "Missing challenge instance" << '\n';
      usage();
      return 1;
    }

    if(N_str!=""){
      try{
        N = std::stoi(N_str);
      }
      catch(...) {
        std::cout << "Invalid argument for -n, expected a positive integer" << '\n';
        usage();
        return 1;
      }
      if(N<1){
        std::cout << "Invalid argument for -n, expected a positive integer" << '\n';
        usage();
        return 1;
      }
      else if(N>1){
        std::cout << "Repeating the algorithm " << N << " times" << '\n';
      }
    }

    Parser parse;
    std::cout << "Parsing instance " << path << '\n';
    bool b = parse.load_file(path);

    if (!b) {
        std::cout << "Parsing instance failed" << '\n';
        return 1;
    }

    mat_view H = parse.get_H();
    vec_view S = parse.get_S();
    size_t n = parse.get_n();
    size_t k = parse.get_k();
    size_t w = parse.get_w();

    std::cout << "n = " << n << '\n';
    std::cout << "k = " << k << '\n';
    std::cout << "w = " << w << '\n';


    int c = 0;
    for (int i=0; i<N; i++){
      if (algo=="PRANGE")
        c += run_subISD<subISD_prange>(H,S,w);
      else if (algo=="LEE BRICKELL")
        c += run_subISD<subISD_LB>(H,S,w);
      else {
        std::cout << "Unknown algorithm" << '\n';
        return 1;
      }
    }

    float avg_cnt = (float) c / N;
    float inv_avg_cnt = (float) 1 / avg_cnt;
    std::cout << "Average number of iterations: " << avg_cnt << '\n';
    std::cout << "Inverse of average number of iterations: " << inv_avg_cnt << '\n';
    return 0;
  }
