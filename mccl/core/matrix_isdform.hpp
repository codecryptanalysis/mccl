#ifndef MCCL_CORE_MATRIX_ISDFORM_HPP
#define MCCL_CORE_MATRIX_ISDFORM_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix.hpp>
#include <mccl/core/random.hpp>

#include <algorithm>
#include <numeric>

MCCL_BEGIN_NAMESPACE

/*
   Class to bring (H|S)^T in desired ISD form after a random column permutation of H:
   The performance optimized ISD form for (H|S) is:
      HS = ( 0  | H2 | s2 ) = U x ((Horg x P) | Sorg)
           ( AI | H1 | s1 )
      HS: (n-k) x (n+1)
      AI: (n-k-l) x (n-k-l)
      H1: (n-k-l) x (k+l)
      H2:      l  x (k+l)
      where AI is the antidiagonal identity matrix
      as a result of performing reverse row reduction from bottom to top
   This translates to (H|S)^T to
      (H|S)^T = ( 0    | AI   ) 
                ( H2^T | H1^T )
                ( s2^T | s1^T )
      HST: (n+1)   x (n-k)
      AI : (n-k-l) x (n-k-l)
      H1T: (k+l)   x (n-k-l)
      H2T: (k+l)   x l
      This form ensures H2^T columns are before H1^T columns
      and thus is flexible with padding H2^T column words (64-bit or s-bit SIMD) with additional H1^T columns
*/
template<size_t _bit_alignment = 64>
class HST_ISD_form_t
{
public:
    static const size_t bit_alignment = _bit_alignment;
    typedef aligned_tag<bit_alignment> this_aligned_tag;
    
    HST_ISD_form_t() {}
    HST_ISD_form_t(const cmat_view& H_, const cvec_view& S_, size_t l_) { reset(H_, S_, l_); }
    void reset(const cmat_view& H_, const cvec_view& S_, size_t l_)
    {
    	assert( l_ < H_.rows() );
    	assert( S_.columns() == H_.rows() );
    	
    	// setup HST
    	size_t HTrows = H_.columns(), HTcols = H_.rows();
    	size_t HTcolspadded = (HTcols + bit_alignment - 1) & ~(bit_alignment-1);
    	HT_columns = HTcols;
    	H2T_columns = l_;
    	H2T_columns_padded = (H2T_columns + bit_alignment - 1) & ~(bit_alignment - 1);
    	H1T_columns = HTcols - l_;
    	echelon_rows = HTcols - l_;
    	ISD_rows = HTrows - echelon_rows;
    	max_update_rows = size_t( float(echelon_rows) * float(ISD_rows) / float(echelon_rows+ISD_rows) );

    	HST.resize(HTrows + 1, HTcolspadded);
    	HST.clear();

    	// create views
    	_HT        .reset(HST.submatrix(0, HTrows, 0, HTcols));
    	_HTpadded  .reset(HST.submatrix(0, HTrows, 0, HTcolspadded));
    	_H12T      .reset(HST.submatrix(echelon_rows, ISD_rows, 0, HTcols));
    	_H12Tpadded.reset(HST.submatrix(echelon_rows, ISD_rows, 0, HTcolspadded));
    	_S         .reset(HST[HTrows].subvector(0, HTcols));
    	_Spadded   .reset(HST[HTrows].subvector(0, HTcolspadded));

	_H2T      .reset(HST.submatrix(echelon_rows, ISD_rows, 0, H2T_columns));
	_H2Tpadded.reset(HST.submatrix(echelon_rows, ISD_rows, 0, H2T_columns_padded));
	_S2       .reset(_Spadded.subvector(0, H2T_columns));
	_S2padded .reset(_Spadded.subvector(0, H2T_columns_padded));
	
	_H1Trest      .reset(HST.submatrix(echelon_rows, ISD_rows, H2T_columns_padded, HTcols));
	_H1Trestpadded.reset(HST.submatrix(echelon_rows, ISD_rows, H2T_columns_padded, HTcolspadded));
	_S1rest       .reset(_Spadded.subvector(H2T_columns_padded, HTcols));
	_S1restpadded .reset(_Spadded.subvector(H2T_columns_padded, HTcolspadded));

    	// copy H and S into HST
    	_HT.transpose(H_);
    	_S.copy(S_);

    	// setup HT row perm
    	perm.resize(HTrows);
    	std::iota(perm.begin(), perm.end(), 0);

    	// setup HT echelon row perm: used to pick u random echelon rows
    	echelon_perm.resize(echelon_rows);
    	std::iota(echelon_perm.begin(), echelon_perm.end(), 0);
    	cur_echelon_row = 0;

    	// setup HT ISD row perm: used to pick u random ISD rows
    	ISD_perm.resize(ISD_rows);
    	std::iota(ISD_perm.begin(), ISD_perm.end(), 0);
    	cur_ISD_row = 0; rnd_ISD_row = 0;

    	// randomize & bring into ISD form
    	for (echelon_start = 0; echelon_start < echelon_rows; ++echelon_start)
    	{
    		size_t pivotcol = HT_columns - echelon_start - 1;
    		// start at a random remaining row and scan rows for bit set at column pivotcol
    		size_t r = echelon_start + (rndgen() % (_HT.rows() - echelon_start));
    		for (; r < _HT.rows() && _HT(r,pivotcol)==false; ++r)
    			;
		// wrap around once
		if (r == _HT.rows())
		{
			r = echelon_start;
	    		for (; r < _HT.rows() && _HT(r,pivotcol)==false; ++r)
    				;
		}
		// check for failure
		if (r == _HT.rows())
			std::runtime_error("HST_ISD_form_t::reset(): cannot bring HT in ISD form");
		// perform swap and reduction
		swap_echelon(echelon_start, r - echelon_rows);
	}
    }

    const std::vector<uint32_t>& permutation() const { return perm; }
    uint32_t permutation(uint32_t x) const { return perm[x]; }
    
    cvec_view_it operator[](size_t r) const { return HST[r]; }
    cvec_view_it operator()(size_t r) const { return HST[r]; }

    const cmat_view& HSTpadded() const { return HST; }
    size_t echelonrows() const { return echelon_rows; }
    size_t ISDrows() const { return ISD_rows; }

    const cmat_view& HT()            const { return _HT; }
    const cmat_view& HTpadded()      const { return _HTpadded; }
    const cmat_view& H12T()          const { return _H12T; }
    const cmat_view& H12Tpadded()    const { return _H12Tpadded; }
    const cmat_view& H2T()           const { return _H2T; }
    const cmat_view& H2Tpadded()     const { return _H2Tpadded; }
    const cmat_view& H1Trest()       const { return _H1Trest; }
    const cmat_view& H1Trestpadded() const { return _H1Trestpadded; }

    const cvec_view& S()            const { return _S; }
    const cvec_view& Spadded()      const { return _Spadded; }
    const cvec_view& S2()           const { return _S2; }
    const cvec_view& S2padded()     const { return _S2padded; }
    const cvec_view& S1rest()       const { return _S1rest; }
    const cvec_view& S1restpadded() const { return _S1restpadded; }
    
    
    // swap with random row outside echelon form and bring it back to echelon form
    void swap_echelon(size_t echelon_idx, size_t ISD_idx)
    {
    	if (echelon_idx >= echelon_rows || echelon_rows + ISD_idx >= perm.size())
    		throw std::runtime_error("HST_ISD_form_t::swap_echelon(): bad input index");
	// swap rows
	std::swap(perm[echelon_idx], perm[echelon_rows + ISD_idx]);
	HST[echelon_idx].swap(HST[echelon_rows + ISD_idx], this_aligned_tag());

	// bring HST back in echelon form
	size_t pivotcol = HT_columns - echelon_idx - 1;
	vec_view pivotrow(HST[echelon_idx]);
	pivotrow.clearbit(pivotcol);
	auto HSTrowit = HST[echelon_start];
	for (size_t r2 = echelon_start; r2 < HST.rows(); ++r2,++HSTrowit)
		if (HST(r2,pivotcol))
			HSTrowit.vxor(pivotrow, this_aligned_tag());
	pivotrow.clear(this_aligned_tag());
	pivotrow.setbit(pivotcol);
    }
    // update 1 echelon row
    void update1(size_t echelon_idx)
    {
    	if (echelon_idx >= echelon_rows)
    		throw std::runtime_error("HST_ISD_form_t::update(): bad input index");
    	// ISD row must have 1-bit in column pivotcol:
    	size_t pivotcol = HT_columns - echelon_idx - 1;
    	// find random row to swap with
    	//   must have bit set at pivot column
    	//   start at random position and then do linear search
    	size_t ISD_idx = rndgen() % ISD_rows;
	for (; ISD_idx < ISD_rows && HST(echelon_rows + ISD_idx,pivotcol)==false; ++ISD_idx)
		;
	// wrap around
	if (ISD_idx >= ISD_rows) // unlikely
	{
		ISD_idx = 0;
		for (; ISD_idx < ISD_rows && HST(echelon_rows + ISD_idx,pivotcol)==false; ++ISD_idx)
			;
	}
	// oh oh if we wrap around twice
	if (ISD_idx >= ISD_rows) // unlikely
		throw std::runtime_error("HST_ISD_form_t::update1(): cannot find pivot");
	swap_echelon(echelon_idx, ISD_idx);
    }
    // update 1 echelon row
    void update1_ISDseq(size_t echelon_idx)
    {
    	if (echelon_idx >= echelon_rows)
    		throw std::runtime_error("HST_ISD_form_t::update(): bad input index");
    	// ISD row must have 1-bit in column pivotcol:
    	size_t pivotcol = HT_columns - echelon_idx - 1;
    	while (true)
    	{
		cur_ISD_row = (cur_ISD_row + 1) % ISD_rows;
		if (HST(echelon_rows + cur_ISD_row,pivotcol))
			break;
    	}
	swap_echelon(echelon_idx, cur_ISD_row);
    }

    // update 1 echelon row, choose ISD row from next one in a maintained random permutation
    void update1_ISDperm(size_t echelon_idx)
    {
    	if (echelon_idx >= echelon_rows)
    		throw std::runtime_error("HST_ISD_form_t::update1_ISDsubset(): bad input index");
    	size_t pivotcol = HT_columns - echelon_idx - 1;
	size_t ISD_idx = 0;
	while (true)
	{
		// if we have consumed max_update_rows from our permutation then we (lazily) refresh it
		if (cur_ISD_row >= max_update_rows)
		{
			cur_ISD_row = 0;
			rnd_ISD_row = 0;
		}
		for (ISD_idx = cur_ISD_row; ISD_idx < ISD_perm.size(); ++ISD_idx)
		{
			// create random permutation just in time
			if (ISD_idx == rnd_ISD_row)
			{
				std::swap(ISD_perm[ISD_idx], ISD_perm[ ISD_idx + (rndgen() % (ISD_rows - ISD_idx))]);
				++rnd_ISD_row;
			}
			if (HST(echelon_rows + ISD_perm[ISD_idx], pivotcol)==true)
				break;
		}
		if (ISD_idx < ISD_perm.size())
		{
			break;
		}
		// force new permutation
		cur_ISD_row = ISD_rows;
	}
	// move chosen index to cur_ISD_row position and do update
	std::swap(ISD_perm[cur_ISD_row], ISD_perm[ISD_idx]);
	ISD_idx = ISD_perm[cur_ISD_row];
	++cur_ISD_row;
	swap_echelon(echelon_idx, ISD_idx);
    }



    // Type 1: u times: pick a random echelon row & random ISD row to swap
    void update_type1(size_t rows)
    {
    	for (size_t i = 0; i < rows; ++i)
    		update1(rndgen() % echelon_rows);
    }
    // Type 2: pick u random distinct echelon rows & u random (non-distinct) ISD rows to swap
    void update_type2(size_t rows)
    {
    	for (size_t i = 0; i < rows; ++i)
    		std::swap(echelon_perm[i], echelon_perm[ rndgen() % echelon_rows ]);
	for (size_t i = 0; i < rows; ++i)
		update1(echelon_perm[i]);
    }
    // Type 3: pick u random distinct echelon rows & ISD rows to swap
    void update_type3(size_t rows)
    {
    	// trigger refresh of ISD_perm
    	cur_ISD_row = ISD_rows;
    	// refresh echelon_perm
    	for (size_t i = 0; i < rows; ++i)
    		std::swap(echelon_perm[i], echelon_perm[ rndgen() % echelon_rows ]);
	for (size_t i = 0; i < rows; ++i)
		update1_ISDperm(echelon_perm[i]);
    }
    // Type 4: pick max_update_rows = k*(n-k)/n random distinct echelon rows & ISD rows to swap
    //         process u of them this round, keep the rest for next rounds until empty, repeat
    void update_type4(size_t rows)
    {
	for (size_t i = 0; i < rows; ++i)
	{
		// refresh echelon_perm when max_update_rows have been consumed
		if (cur_echelon_row >= max_update_rows)
		{
	    		for (size_t i = 0; i < max_update_rows; ++i)
    				std::swap(echelon_perm[i], echelon_perm[ rndgen() % echelon_rows ]);
			cur_echelon_row = 0;
		}
		update1_ISDperm(echelon_perm[cur_echelon_row]);
		++cur_echelon_row;
	}
    }

    // Type 10: pick u round-robin echelon rows & round-robin scan of ISD rows
    void update_type10(size_t rows)
    {
    	for (size_t i = 0; i < rows; ++i)
	{
		update1_ISDseq(cur_echelon_row);
		cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
	}
    }
    // Type 12: pick u round-robin echelon rows & u random (non-distinct) ISD rows to swap
    void update_type12(size_t rows)
    {
	for (size_t i = 0; i < rows; ++i)
	{
		update1(cur_echelon_row);
		cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
	}
    }
    // Type 13: pick u round-robin echelon rows & u random distinct ISD rows to swap
    void update_type13(size_t rows)
    {
    	// trigger refresh of ISD_perm
    	cur_ISD_row = ISD_rows;
	for (size_t i = 0; i < rows; ++i)
	{
		update1_ISDperm(cur_echelon_row);
		cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
	}
    }
    // Type 14: pick max_update_rows = k*(n-k)/n random distinct ISD rows to swap
    //         process u of them this round with u round-robin echelon rows
    //         keep the rest for next rounds until empty, repeat
    void update_type14(size_t rows)
    {
	for (size_t i = 0; i < rows; ++i)
	{
		update1_ISDperm(cur_echelon_row);
		cur_echelon_row = (cur_echelon_row + 1) % echelon_rows;
	}
    }

    // default choice is to use type 14
    void update(int r = -1, int updatetype = 14)
    {
    	size_t rows = r > 0 ? std::min<size_t>(r, max_update_rows) : max_update_rows;
    	switch (updatetype)
    	{
    		case 1:
    			update_type1(rows);
			break;
		case 2:
			update_type2(rows);
			break;
		case 3:
			update_type3(rows);
			break;
		case 4:
			update_type4(rows);
			break;
		case 10:
			update_type10(rows);
			break;
		case 12:
			update_type12(rows);
			break;
		case 13:
			update_type13(rows);
			break;
    		case 14:
    			update_type14(rows);
    			break;
		default:
			throw std::runtime_error("HST_ISD_form_t::update(): unknown update type");
    	}
    }

private:
    mat HST;

    mat_view _HT, _HTpadded, _H12T, _H12Tpadded;
    vec_view _S, _Spadded;

    mat_view _H2T, _H2Tpadded;
    vec_view _S2, _S2padded;

    mat_view _H1Trest, _H1Trestpadded;
    vec_view _S1rest, _S1restpadded;
    
    std::vector<uint32_t> perm;
    size_t HT_columns, H1T_columns, H2T_columns, H2T_columns_padded;
    size_t echelon_rows, ISD_rows, max_update_rows, echelon_start, cur_echelon_row, cur_ISD_row, rnd_ISD_row;
    std::vector<uint32_t> echelon_perm, ISD_perm;

    mccl_base_random_generator rndgen;
};

MCCL_END_NAMESPACE

#endif
