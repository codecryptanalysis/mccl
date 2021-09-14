# Low-level framework design

## Matrices & Vectors design

This library is designed for matrices and vectors over F_2, i.e., bit matrices and bit vectors.
All the bits in a row or vector are stored contiguously in memory packed in unsigned 64-bit integers `uint64_t` named words.
Matrices and vectors may have arbitrary number of columns (and rows), however the first column always corresponds to the least significant bit of the first word of each row.
Hence, while submatrices and -vectors are supported, they must start on a column position that is a multiple of 64.
To ensure that modifications of a submatrix do not affect other unintended columns in the larger matrix, the library uses a bitmask for the last word of each row.

### Matrix and vector types

Matrices are quite large and copying submatrices and vectors should be avoided if possible.
Hence, we mainly work through classes that capture the specification of a (sub)matrix/vector located in memory called matrix/vector views.
Moreover, the libraries function will therefore take any destination as input parameter, instead of returning a new matrix/vector.

The following types are the default matrix & vector types:
- `mat`: a matrix object that allocates and maintains memory
- `mat_view`: a view to a non-const (sub)matrix
- `cmat_view`: a view to a const (sub)matrix
- `vec_view_it`: a view to a non-const (sub)vector that can iterate over a (sub)matrix rows
- `cvec_view_it`: a view to a const (sub)vector that can iterate over a (sub)matrix rows
- `vec`: a vector object that allocates and maintains memory
- `vec_view`: a view to a non-const (sub)vector
- `cvec_view`: a view to a const (sub)vector

### View objects

View objects behave like pointers:
- (1) copy/move constructor copies the view, not the contents. Therefore copy/move assignment from matrix/vector objects and views has been deleted, to avoid confusion. 
- (2) const-ness of the view only dictates whether the view can be modified or not, and not that of the underlying contents. Thus we have non-const & const matrix/vector view versions: `mat_view` vs `cmat_view`, `vec_view_it` vs `cvec_view_it` and `vec_view` vs `cvec_view`.
It follows that a const view to a non-const matrix/vector may still be used to modify its contents, 
while a non-const view to a const matrix/vector cannot be used to modify its contents.

Otherwise, view objects behave mostly like a matrix/vector object when it comes to matrix/vector operations.
They can be passed to all functions that take a matrix/vector and have all basic matrix/vector member functions.

### Result objects

Matrices and vectors have member functions to modify their contents, e.g. `m.mand(m1,m2);` assigns the result of the bit-wise and of `m1` and `m2` to the matrix (view) `m`.
For convenience, this can also be written as `m = m_and(m1,m2);`. 

Internally, the expression `m_and(m1,m2)` returns a `matrix_result` object that captures `m1`, `m2` and the intended member function call `m.mand` (for a yet unknown matrix (view) `m`). When that `matrix_result` is assigned to the matrix (view) `m`, `m` passes itself to the `matrix_result` which then finally calls the intended function.

### Matrix member functions

Common const matrix member functions
- `columns()`: nr of columns
- `rows()`: nr of rows
- `hw()`: returns the hammingweight
- `wordptr(r = 0)`: returns a `[const] uint64_t*` to the first word of row `r`
- `rowwords()`: nr of words in a row
- `stride()`: nr of words to jump to the next row
- `operator()(r,c)`: returns the bit (as `bool`) on row `r` column `c` of `mat`
- `isequal(m)`: returns a `bool` whether the two matrices have equal contents
- `subvector(r, coloff, cols)`: returns a vector (const) view to row `r`, columns `coloff`,...,`coloff+cols-1`
- `submatrix(rowoff, rows, coloff, cols)`: returns a matrix (const) view to rows `rowoff`,...,`rowoff+rows-1` and columns `coloff`,...,`coloff+cols-1`
- `operator[](r)`, `operator()(r)`: returns a vector iterator (const) view to row `r`
- `begin()`, `end()`: returns a vector iterator (const) view to row `0` and `rows()`
Note that actual const-ness of the return type may depend on the const-ness of the underlying matrix contents.
I.e., const for `const mat`, `cmat_view`, and non-const for `mat` and `mat_view`.

Common matrix member functions:
- `clear()`, `set(b = true)`: clear matrix to 0, set all bits matrix to `b`
- `setidentity()`: clear matrix and set the diagonal bits
- `mnot()`: invert all bits
- `clearbit(r, c)`, `flipbit(r,c)`, `setbit(r,c)`, `setbit(r,c,b)`: modify bit a row `r` column `c` (clear to 0, flip, set to 1, set to `b`)
- `clearcolumns(c_off, c_cnt)`, `flipcolumns(c_off,c_cnt)`, `setcolumns(c_off,c_cnt)`, `setcolumns(c_off,c_cnt,b)`: modify columns `c_off`,..,`c_off+c_cnt-1`: clear to 0, flip, set to 1, set to `b`
- `swapcolumns(c1, c2)`: swap columns `c1` and `c2`
- `copy(m)`: assign contents of `m`
- `mnot(m)`: assign contents of `m` and invert all bits
- `transpose(m)`: assign the transposed contents of `m`
- `mxor(m)`, `mand(m)`, `mor(m)`: assign the xor/and/or of this matrix with `m`
- `mxor(m1,m2)`, `mand(m1,m2)`, `mor(m1,m2)`: assign the xor/and/or of `m1` and `m2`
- `operator^=(m)`, `operator&=(m)`, `operator|=(m)`: assign the xor/and/or of this matrix with `m`

Additional `mat_view`, `cmat_view` member functions:
- `operator=(matrix_result)`: assign the matrix result
- `reset(mv)`: assign the view `mv`

Additional `mat` member functions:
- `resize(rows,cols,b=false)`: resizes matrix, set new bits to `b`
- `operator=(m)`: resize, and assign the contents of m
- `operator=(matrix_result)`: resize, and assign the matrix result
- `operator=(mat&& m)`, `swap(mat& m)`: swap underlying contents with another `mat` object

Global matrix functions:
- `operator==/!=(const mat_view/cmat_view& m1, const mat_view/cmat_view& m2)`: compares *view* of m1 & m2
- `operator==/!=(const mat& m1, const mat/mat_view/cmat_view& m2)`: compares *contents* of m1 & m2
- `operator==/!=(const mat/mat_view/cmat_view& m1, const mat& m2)`: compares *contents* of m1 & m2
- `operator<<(std::ostream& o, m)`: prints `m` to ostream `o`
- `m_transpose(m)`: returns `matrix_result` object to assign the transposed contents of `m`
- `m_copy(m), m_copynot(m)`: returns `matrix_result` object to assign the contents of `m` (as is / inverting all bits)
- `m_and(m1,m2), m_or(m1,m2), m_xor(m1,m2)`: returns a `matrix_result` to assign the contents of the and/or/xor of `m1` and `m2`
- `operator & (m1,m2), operator | (m1,m2), operator ^ (m1,m2)`: returns a `matrix_result` to assign the contents of the and/or/xor of `m1` and `m2`
- `hammingweight(m)`: returns the hammingweight of `m`
