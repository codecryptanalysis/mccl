#ifndef MCCL_TOOLS_PARSER_H
#define MCCL_TOOLS_PARSER_H

#include <mccl/core/matrix.hpp>

#include <mccl/contrib/string_algo.hpp>
namespace sa = string_algo;

#include <string>
#include <fstream>
#include <iostream>

MCCL_BEGIN_NAMESPACE

enum Marker {
	MARK_NONE, MARK_N, MARK_K, MARK_W, MARK_SEED, MARK_HT, MARK_ST
};

std::vector<bool> string_to_booleans(std::string& str) {
	std::vector<bool> V;
	for( char &c : str) {
		if(c=='0') V.push_back(false);
		if(c=='1') V.push_back(true);
	}
	return V;
}

class Parser {
	public:
		bool load_file(std::string filename);
		bool random_SD(int n, int k, int w);
		bool regenerate();
		const mat_view& get_H() { return H; }
		const vec_view& get_S() { return S; }
		size_t get_n() { return n; };
		size_t get_k() { return k; };
		size_t get_w() { return w; };
		size_t get_seed() { return seed; };
		bool check_solution(const cvec_view& E);
	private:
		bool Nset=false, Kset=false, Wset=false, Seedset=false;
		int64_t n, k, w, seed;
		
		std::vector<bool> ST;
		std::vector<std::vector<bool>> HT;
		mat H;
		vec S;
};


bool Parser::load_file(std::string filename) 
{
	std::ifstream f(filename);
	if(!f.good()) {
		std::cerr << "File " << filename << " does not exist" << std::endl;
		return false;
	}

	Nset=false, Kset=false, Wset=false, Seedset=false;
	std::string line;
	std::vector<std::string> HT_string;
	std::string ST_string="";
	Marker active = MARK_NONE;
	size_t linenr=0;
	while(std::getline(f, line)) {
		if(line.size()==0) continue;
		if( line[0]=='#' ) {
			// comment or marker
			sa::trim_left(line, "# ");
			if(line.size()==0) continue;
			if(line[0]=='n' or line[0]=='N') active=MARK_N;
			if(line[0]=='k' or line[0]=='K') active=MARK_K;
			if(line[0]=='w' or line[0]=='W') active=MARK_W;
			if(line[0]=='h' or line[0]=='H') active=MARK_HT;
			if(line[0]=='s' or line[0]=='S') active=MARK_ST;
			if(sa::starts_with(line, "seed")) active=MARK_SEED;
		} else if( line[0] >= '0' and line[0] <= '9' ) {
			switch(active) {
				case MARK_NONE: std::cerr << "Reading numbers while no marker active" << std::endl; return false;
				case MARK_N: n = std::stoi(line); Nset=true; break;
				case MARK_K: k = std::stoi(line); Kset=true; break;
				case MARK_W: w = std::stoi(line); Wset=true; break;
				case MARK_SEED: seed = std::stoi(line); Seedset=true; break;
				case MARK_HT: HT_string.push_back(line); break;
				case MARK_ST: ST_string=line; break;
			}
		} else {
			std::cerr << "Ignoring line: " << linenr << std::endl;
		}
		linenr++;
	}

	// postprocessing
	if(!Nset){ std::cerr << "No length set" << std::endl; return false; }
	if(n<=0) { std::cerr << "Length not positive" << std::endl; return false; } 

	if(!Kset){ std::cerr << "No dim set, defaulting to rate 1/2" << std::endl; k = n/2; }
	if(k<=0 or k >= n) { std::cerr << "Dimension not in correct range" << std::endl; return false; } 

	if(Wset and (w<0 or w>n)) { std::cerr << "Weight not in correct range" << std::endl; return false; }

	if(ST_string.size()>0) {
		ST = string_to_booleans(ST_string);
		if(int64_t(ST.size()) != n-k) { std::cerr << "s doesn't have the correct length" << std::endl; return false; }
	}

	if(int64_t(HT_string.size()) != k) { std::cerr << "H^transpose is missing some rows" << std::endl; return false; }
	HT.clear();
	for(auto &str : HT_string) {
		auto row = string_to_booleans(str);
		if(int64_t(row.size())!=n-k) { std::cerr << "Row has wrong length." << std::endl; return false; };
		HT.push_back(row);
	}

	H = mat(n-k, n);
	H.setidentity();
	for( int64_t r = 0; r < n-k; r++) {
		for( int64_t c = 0; c < k; c++ ) {
			if(HT[c][r])
				H.setbit(r, n-k+c);
		}
	}

	S = vec(n-k);
	for(int64_t r = 0; r < n-k; r++ ) {
		if(ST[r])
			S.setbit(r);
	}
	return true;
}

// generate random testcase
bool Parser::random_SD(int n_, int k_, int w_) {
	n=n_;
	k=k_;
	w=w_;
	seed=0;

	if(n<=0) { std::cerr << "Length not positive" << std::endl; return false; } 
	if(k<=0 or k >= n) { std::cerr << "Dimension not in correct range" << std::endl; return false; } 
	if(w<0 or w>n-k) { std::cerr << "Weight not in correct range [0,n-k]" << std::endl; return false; }
	H = mat(n-k,n);
	fillrandom(H);
	// force identity part for comptability
	for(size_t i = 0; i < size_t(n-k); i++) {
		for(size_t j=0; j < size_t(n-k); j++ ) {
			H.clearbit(i,j);
		}
		H.setbit(i,i);
	}

	// pick random weight w vec by picking random indices 
	// until we have sampled w distinct ones
	// works efficiently if w<<n.
	// mccl_base_random_generator gen;
	// vec E(n);
	// int cnt = 0;
	// while(cnt < w){
	// 	size_t ind = gen()%n;
	// 	if(!E[ind]) {
	// 		E.setbit(ind);
	// 		cnt++;
	// 	}
	// }
	//std::cout << "Generated error:" << E << std::endl;

	// compute syndrome
	S = vec(n-k);
	fillrandom(S);
	// for(size_t i =0; i < size_t(n-k); i++) {
	// 	S.setbit(i,hammingweight_xor(E, H[i])%2);
	// }
	//std::cout << "S: " << S << std::endl;
	return true;
}

bool Parser::regenerate() {
	return random_SD(n,k,w);
}

bool Parser::check_solution(const cvec_view& E)
{
	if (E.columns() != H.columns())
		throw std::runtime_error("Parser::check_solution(): incorrect dimension of E");
	mat HT = m_transpose(H);
	vec C = v_copy(S);
	for (size_t i = 0; i < E.columns(); ++i)
		if (E[i])
			C.vxor(HT[i]);
	return 0 == hammingweight(C);
}

MCCL_END_NAMESPACE

#endif
