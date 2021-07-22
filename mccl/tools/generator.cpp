#include <mccl/tools/generator.hpp>
#include <mccl/core/matrix_algorithms.hpp>
#include <mccl/tools/utils.hpp>

MCCL_BEGIN_NAMESPACE

// generate random testcase
void SDP_generator::generate(int n, int k, int w)
{
	_n=n;
	_k=k;
	_w=w;
	
	// automatic parameters
	if (_k < 0)
		_k = _n>>1;
	if (_w < 0)
		_w = get_cryptographic_w(_n, _k);
		
	// sanity checks
	if (_n <= 0)
		throw std::runtime_error("SDP_generator::generate(): n <= 0");
	if (k <= 0 || k >= n)
		throw std::runtime_error("SDP_generator::generate(): k <= 0 || k >= n");
	if (w <= 0 || w > n-k)
		throw std::runtime_error("SDP_generator::generate(): w <= 0 || w > n-k");
		
	// first completely fill with randomly generated bits
	tmpH.resize(n-k,n);
	fillgenerator(tmpH, rndgen);
	_H = m_copy(tmpH);
	
	// now set left n-k by n-k submatrix to the identity
	tmpI.resize(n-k,n-k);
	tmpI.setidentity();
	_H.submatrix(0,n-k,0,n-k) = m_copy(tmpI);

	// generate syndrome
	tmpS.resize(n-k);
	fillgenerator(tmpS, rndgen);
	_S = v_copy(tmpS);
}

MCCL_END_NAMESPACE
