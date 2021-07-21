#ifndef MCCL_TOOLS_GENERATOR_HPP
#define MCCL_TOOLS_GENERATOR_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix.hpp>

MCCL_BEGIN_NAMESPACE

class SDP_generator
{
public:
	SDP_generator()
		: _n(-1), _k(-1), _w(-1)
	{
	}

	SDP_generator(int n, int k = -1, int w = -1)
	{
		generate(n, k, w);
	}

	void generate(int n, int k = -1, int w = -1);

	void regenerate() { generate(_n, _k, _w); }

	void seed(uint64_t s) { rndgen.seed(s); }
	uint64_t get_seed() { return rndgen.get_seed(); };

	const mat_view& H() { return _H; }
	const vec_view& S() { return _S; }
	int n() { return _n; };
	int k() { return _k; };
	int w() { return _w; };
		
private:
	int _n, _k, _w;
	mat _H;
	vec _S;
	mat tmpH, tmpI;
	vec tmpS;
		
	mccl_base_random_generator rndgen;
};

MCCL_END_NAMESPACE

#endif
