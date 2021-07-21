#ifndef MCCL_TOOLS_PARSER_H
#define MCCL_TOOLS_PARSER_H

#include <mccl/core/matrix.hpp>

MCCL_BEGIN_NAMESPACE

enum Marker {
	MARK_NONE, MARK_N, MARK_K, MARK_W, MARK_SEED, MARK_G, MARK_H, MARK_S, MARK_GT, MARK_HT, MARK_ST
};

typedef std::pair<std::vector<uint64_t>, size_t> parser_binary_vector_t;

class file_parser
{
public:
	bool parse_file(const std::string& filename, std::string fileformat = "");

	const cmat_view& G() const { return _G; }
	const cmat_view& H() const { return _H; }
	const cvec_view& S() const { return _S; }
	size_t n()           const { return _H.columns(); }
	size_t k()           const { return _H.columns() - _H.rows(); }
	int w()              const { return _w; }
	int64_t fileseed()   const { return _fileseed; }

	void reset();

private:
	mat _H, _G;
	vec _S;
	int64_t _n, _k, _w, _fileseed;
	bool quasi_cyclic_H;
	bool omitted_identity_H;
	bool omitted_identity_HT;
	bool omitted_identity_G;
	bool omitted_identity_GT;

	std::vector< parser_binary_vector_t > _Sparsed, _STparsed;
	std::vector< parser_binary_vector_t > _Hparsed, _HTparsed;
	std::vector< parser_binary_vector_t > _Gparsed, _GTparsed;
	std::vector< parser_binary_vector_t > _Munknown;

	// file parsing functions
	bool _parse_file_auto(const std::string& filename);

	// helper functions
	void _parse_integer(const std::string& line, int64_t& n, bool throw_on_duplicate = false) const;
	parser_binary_vector_t _parse_vector(const std::string& line) const;

	mat _parse_matrix(const std::vector<parser_binary_vector_t>& Mparsed) const;
	mat _dual_matrix(const cmat_view& m) const;
	mat _prepend_identity(const cmat_view& m) const;

	void _postprocess_matrices();
};

MCCL_END_NAMESPACE

#endif
