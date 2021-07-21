#include <mccl/tools/parser.hpp>

#include <mccl/contrib/string_algo.hpp>
namespace sa = string_algo;

#include <string>
#include <fstream>
#include <iostream>

MCCL_BEGIN_NAMESPACE

void file_parser::reset()
{
	_n = _k = _w = _fileseed = -1;
	_Sparsed.clear(); _STparsed.clear();
	_Hparsed.clear(); _HTparsed.clear();
	_Gparsed.clear(); _GTparsed.clear();
	_Munknown.clear();
	_G.resize(0,0);
	_H.resize(0,0);
	_S.resize(0);
	quasi_cyclic_H = omitted_identity_H = omitted_identity_HT = false;
	omitted_identity_G = omitted_identity_GT = false;
}

// parse a single integer to a int64_t passed by reference
// if throw_on_duplicate is set then it will throw if variable does not have value -1 (as set by reset())
void file_parser::_parse_integer(const std::string& line, int64_t& n, bool throw_on_duplicate) const
{
	if (throw_on_duplicate && n != -1)
		throw std::runtime_error("Parser::_parse_integer(): integer parameter occured twice!");
	std::stringstream strstr(line);
	strstr >> n;
	if (!strstr)
		throw std::runtime_error("Parser::_parse_integer(): could not parse integer: " + line);
	strstr.get();
	if (!!strstr)
		throw std::runtime_error("Parser::_parse_integer(): could not *fully* parse integer: " + line);
}

// parse and return a binary vector
parser_binary_vector_t file_parser::_parse_vector(const std::string& line) const
{
	parser_binary_vector_t ret;
	uint64_t lastword = 0;
	size_t nextbit = 0;
	for (size_t pos = 0; pos < line.size(); ++pos)
	{
		// skip whitespace & vector braces
		if (line[pos] == ' ' || line[pos] == '\t' || line[pos] == '(' || line[pos] == ')' || line[pos] == '[' || line[pos] == ']')
			continue;
		// process '0' and '1'
		if (line[pos] == '1')
			lastword |= uint64_t(1) << nextbit;
		if (line[pos] == '0' || line[pos] == '1')
		{
			++ret.second;
			if (++nextbit == 64)
			{
				ret.first.emplace_back(lastword);
				lastword = 0;
				nextbit = 0;
			}
			continue;
		}
		// stop at comment character '#', or ending vector brace
		if (line[pos] == '#')
			break;
		// throw on other unexpected character
		throw std::runtime_error("Parse::_parse_vector(): found unexpected character in vector: " + line);
	}
	if (nextbit > 0)
		ret.first.emplace_back(lastword);
	return ret;
}

mat file_parser::_parse_matrix(const std::vector<parser_binary_vector_t>& Mparsed) const
{
	mat ret;
	if (Mparsed.empty())
		return ret;
	size_t columns = Mparsed.front().second;
	for (auto& r : Mparsed)
		if (r.second != columns)
			throw std::runtime_error("Parser::_parse_matrix(): unequal row lengths!");
	ret.resize(Mparsed.size(), columns);
	for (size_t r = 0; r < Mparsed.size(); ++r)
		std::copy(Mparsed[r].first.begin(), Mparsed[r].first.end(), ret.data(r));
	return ret;
}

// generalized dual matrix computation, where the nxn left submatrix may be non-invertible
mat file_parser::_dual_matrix(const cmat_view& m) const
{
	mat msf(m);
	// echelonize msf
	size_t pr = echelonize(msf);
	// remove zero rows
	msf.resize( pr, msf.columns() );
	// compute transpose over which we'll compute
	mat msfT = m_transpose(msf);

	size_t rows = msf.rows(), columns = msf.columns();

	// swap columns such that msf = ( I_n | P ) / rows such that msfT = (I_n | P)^T
	std::vector< std::pair<size_t,size_t> > columnswaps;
	vec tmp;
	for (size_t p = 0; p < rows; ++p)
	{
		// find msf column = msfT row with single bit set at position p
		size_t c = p;
		for (; c < columns; ++c)
			if (hammingweight(msfT[c]) == 1 || msfT(c,p) == true)
				break;
		if (c == p)
			continue;
		if (c == columns)
			throw std::runtime_error("Parser::_dual_matrix(): internal error 1");
		// swap columns
		columnswaps.emplace_back(p, c);
		tmp = msfT[p] ^ msfT[c];
		msfT[p] ^= tmp;
		msfT[c] ^= tmp;
	}
	// we should now have a identity matrix as left submatrix
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < rows; ++c)
			if (msfT(c,r) != (r == c))
				throw std::runtime_error("Parser::_dual_matrix(): internal error 2");
				
	// msf = (I_n | P), so msfdual = ( P^T | I_(n-k) )
	mat dual(columns - rows, columns);
	// write P^T, m_transpose doesn't work
	dual.submatrix(0, dual.rows(), 0, msf.rows()) = m_copy( msfT.submatrix(msf.rows(), dual.rows(), 0, msf.rows()));
	// write I_(n-k)
	for (size_t r = 0; r < dual.rows(); ++r)
		dual.setbit(r, columns-1 - r, true);
	// undo column swaps
	while (!columnswaps.empty())
	{
		auto pc = columnswaps.back();
		columnswaps.pop_back();
		dual.swapcolumns(pc.first, pc.second);
	}
	return dual;
}

mat file_parser::_prepend_identity(const cmat_view& m) const
{
	mat ret(m.rows(), m.rows() + m.columns());
	ret.submatrix(0, m.rows(), 0, m.rows()).setidentity();
	for (size_t r = 0; r < m.rows(); ++r)
		for (size_t c = 0; c < m.columns(); ++c)
			ret.setbit(r, m.rows() + c, m(r,c));
	return ret;
}

void file_parser::_postprocess_matrices()
{
	int matrix_count = (int(!_Gparsed.empty()) + int(!_GTparsed.empty()) + int(!_Hparsed.empty()) + int(!_HTparsed.empty()));
	if (matrix_count == 0)
		throw std::runtime_error("Parser::_postprocess_matrices(): no input generator or parity-check matrix found");
	if (matrix_count > 1)
		throw std::runtime_error("Parser::_postprocess_matrices(): multiple input generator or parity-check matrices found!");
	_G.resize(0,0);
	_H.resize(0,0);
	if (!_Gparsed.empty())
		_G = _parse_matrix(_Gparsed);
	if (!_GTparsed.empty())
		_G = m_transpose(_parse_matrix(_GTparsed));
	// optionally prepend identity
	if ((!_Gparsed.empty()  && omitted_identity_G ) || (!_GTparsed.empty() && omitted_identity_GT))
		_G = _prepend_identity(_G);
	// if G is non-empty then generate H
	if (_G.rows() != 0 || _G.columns() != 0)
	{
		// bring in echelon form
		_G.resize( echelonize(_G), _G.columns() );
		// generate H
		_H = _dual_matrix(_G);
		return;
	}
	if (!_Hparsed.empty())
		_H = _parse_matrix(_Hparsed);
	if (!_HTparsed.empty())
		_H = m_transpose(_parse_matrix(_HTparsed));
	// optionally generate quasi-cyclic H
	if (!_Hparsed.empty() && quasi_cyclic_H)
	{
		// generate M by rotating h
		_H.resize(_H.columns(), _H.columns());
		for (size_t r = 1; r < _H.rows(); ++r)
			for (size_t c = 0; c < _H.columns(); ++c)
				_H.setbit(r, c, _H(0, (c+r)%_H.columns() ));
		// transpose M
		mat _Htmp = m_transpose(_H);
		_H.swap(_Htmp);
		// ensure identity will be prepended
		omitted_identity_H = true;
	}
	// optionally prepend identity
	if ((!_Hparsed.empty()  && omitted_identity_H ) || (!_HTparsed.empty() && omitted_identity_HT))
		_H = _prepend_identity(_H);
	// generate G
	_H.resize( echelonize(_H), _H.columns() );
	_G = _dual_matrix(_H);
	return;	
}


bool file_parser::parse_file(const std::string& filename, std::string fileformat)
{
	sa::to_upper(fileformat);
	/* ========== ADD SPECIAL FORMAT PARSERS HERE ============== */
//	if (fileformat == "SPECIAL_FORMAT")
//		return _parse_file_SPECIAL_FORMAT(filename);

	// default & fallback parser: automatic
	return _parse_file_auto(filename);
}

// automatic parser that can parse all https://decodingchallenge.org/ decoding challenges
bool file_parser::_parse_file_auto(const std::string& filename)
{
	reset();
	
	std::ifstream ifs(filename);
	if (!ifs)
		throw std::runtime_error("Parser::_load_file_auto(): could not open file: " + filename);

	// keep history of markers to help in format detection
	std::vector<Marker> markers({MARK_NONE});
	
	std::string line;
	size_t linenr = 1;
	for (; std::getline(ifs, line); ++linenr)
	{
		// remove whitespace
		sa::trim(line);
		// if empty then ignore
		if (line.empty())
			continue;
		// look for markers
		if (line[0] == '#')
		{
			line.erase(0, 1);
			sa::trim(line);
			sa::to_lower(line);
			if (line == "n")
				markers.emplace_back(MARK_N);
			else if (line == "k")
				markers.emplace_back(MARK_K);
			else if (line == "w")
				markers.emplace_back(MARK_W);
			else if (line == "seed")
				markers.emplace_back(MARK_SEED);
			else if (sa::starts_with(line, "g ") || line == "g")
			{
				markers.emplace_back(MARK_G);
				if (sa::contains(line, "identity part is omitted"))
					omitted_identity_G = true;
			}
			else if (sa::starts_with(line, "g^t"))
			{
				markers.emplace_back(MARK_GT);
				if (sa::contains(line, "identity part is omitted"))
					omitted_identity_GT = true;
			}
			else if (sa::starts_with(line, "h ") || line == "h")
			{
				markers.emplace_back(MARK_H);
				if (sa::contains(line, "identity part is omitted"))
					omitted_identity_H = true;
			}
			else if (sa::starts_with(line, "a vector h of length (0.5 n) which describes the parity-check matrix"))
			{
				markers.emplace_back(MARK_H);
				quasi_cyclic_H = true;
			}
			else if (sa::starts_with(line, "h^t"))
			{
				markers.emplace_back(MARK_HT);
				if (sa::contains(line, "identity part is omitted"))
					omitted_identity_HT = true;
			}
			else if (sa::starts_with(line, "s ") || line == "s")
			{
				markers.emplace_back(MARK_S);
			}
			else if (sa::starts_with(line, "s^t"))
				markers.emplace_back(MARK_ST);
			else
				if (markers.back() != MARK_NONE)
					markers.emplace_back(MARK_NONE);
			continue;
		}
		switch (markers.back())
		{
			case MARK_N:
				_parse_integer(line, _n, true);
				break;
			case MARK_K:
				_parse_integer(line, _k, true);
				break;
			case MARK_W:
				_parse_integer(line, _w, true);
				break;
			case MARK_SEED:
				_parse_integer(line, _fileseed, true);
				break;
			case MARK_G:
				_Gparsed.emplace_back( _parse_vector(line) );
				if (_Gparsed.back().second == 0)
					_Gparsed.pop_back();
				break;
			case MARK_GT:
				_Gparsed.emplace_back( _parse_vector(line) );
				if (_GTparsed.back().second == 0)
					_GTparsed.pop_back();
				break;
			case MARK_H:
				_Hparsed.emplace_back( _parse_vector(line) );
				if (_Hparsed.back().second == 0)
					_Hparsed.pop_back();
				break;
			case MARK_HT:
				_HTparsed.emplace_back( _parse_vector(line) );
				if (_HTparsed.back().second == 0)
					_HTparsed.pop_back();
				break;
			case MARK_S:
				_Sparsed.emplace_back( _parse_vector(line) );
				if (_Sparsed.back().second == 0)
					_Sparsed.pop_back();
				break;
			case MARK_ST:
				_STparsed.emplace_back( _parse_vector(line) );
				if (_STparsed.back().second == 0)
					_STparsed.pop_back();
				break;
			case MARK_NONE:
				_Munknown.emplace_back( _parse_vector(line) );
				if (_Munknown.back().second == 0)
					_Munknown.pop_back();
				break;
		}
	}
	int matrix_count = (int(!_Gparsed.empty()) + int(!_GTparsed.empty()) + int(!_Hparsed.empty()) + int(!_HTparsed.empty()) + int(!_Munknown.empty()));
	if (matrix_count == 0)
		throw std::runtime_error("Parser::_load_file_auto(): no input generator or parity-check matrix found");
	if (matrix_count > 1)
		throw std::runtime_error("Parser::_load_file_auto(): multiple input generator or parity-check matrices found!");
	
	/* detect setting and potentially set post-processing flags */
	// decodingchallenge formats already have proper flags set:
	// - standard syndrome decoding: n, seed, w, HT (omitted identity), ST
	// - low weight decoding: n, seed, HT (omitted identity)
	// - large weight decoding: n, seed, k, w, HT (omitted identity)
	// - Goppa syndrome decoding: n, k, w, HT (omitted identity), ST
	// - Quasi-cyclic syndrome decoding: n, w, HT (single vector), ST
	
	// unknown setting
	if (!_Munknown.empty())
	{
		// assume M = (HT | ST)
		_HTparsed.swap(_Munknown);
		_STparsed.emplace_back(_HTparsed.back());
		_HTparsed.pop_back();
	}
	
	/* post-processing */
	
	// from single input matrix generate all other matrices
	_postprocess_matrices();
	
	// process syndrome
	if (!_STparsed.empty() && !_Sparsed.empty())
		throw std::runtime_error("Parser::_load_file_auto(): multiple syndrome formats found");
	if (!_STparsed.empty())
		_Sparsed.swap(_STparsed);
	mat Stmp = _parse_matrix(_Sparsed);
	if (Stmp.columns() == 1 && Stmp.rows() != 1)
	{
		// transpose Stmp
		mat ST = m_transpose(Stmp);
		Stmp.swap(ST);
	}
	if (Stmp.rows() > 1)
		throw std::runtime_error("Parser::_load_file_auto(): multiple syndromes found");
	if (Stmp.rows() == 1)
		_S = v_copy(Stmp[0]);
	
	/* sanity checks */
	if (_n < 0)
		_n = int64_t(_G.columns());
	if (_k < 0)
		_k = int64_t(_G.rows());
	if (_n != int64_t(_G.columns()) || _k != int64_t(_G.rows()))
		throw std::runtime_error("Parser::_load_file_auto(): G doesn't have the right dimensions");
	if (_n != int64_t(_H.columns()) || (_n-_k) != int64_t(_H.rows()))
		throw std::runtime_error("Parser::_load_file_auto(): H doesn't have the right dimensions");
	if (!(_S.columns() == 0 || _S.columns() == _H.rows()))
		throw std::runtime_error("Parser::_load_file_auto(): S doesn't have the right dimensions");
	return true;
}

MCCL_END_NAMESPACE
