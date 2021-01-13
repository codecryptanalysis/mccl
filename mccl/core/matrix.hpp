#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/config/config.h>

MCCL_BEGIN_NAMESPACE

// matrix_storage allocates & manages memory for matrices
class matrix_storage;
// storage_traits helps distinguishing between non-const/const matrix_storage
template<typename Storage> struct storage_traits;

// matrix_view defines a submatrix of matrix_storage
template<typename Storage = matrix_storage> class matrix_view;
// matrix_view_content is a short-lived wrapper around matrix_view with intent to modify content
template<typename Storage = matrix_storage> class matrix_view_content;

// vector_view defines a row vector of matrix_storage
template<typename Storage = matrix_storage> class vector_view;
// vector_view_content is a short-lived wrapper around vector_view with intent to modify content
template<typename Storage = matrix_storage> class vector_view_content;



/* matrix storage */

// matrix storage automatically rounds up columns
class matrix_storage {
public:
    matrix_storage(size_t rows, size_t columns)
    {
        this->_alloc(rows, columns);
    }
    ~matrix_storage()
    {
        this->_free();
    }
    
    size_t rows() const { return _rows; }
    size_t columns() const { return _columns; }
    size_t scratchcolumns() const { return _scratchcolumns; }

    uint8_t* data(size_t r, size_t c);
    const uint8_t* data(size_t r, size_t c) const;
    const uint8_t* cdata(size_t r, size_t c) const;
    size_t stride() const { return _stride; }
    
    matrix_view<matrix_storage> matrix_view() { return matrix_view<matrix_storage>(*this, 0, rows(), 0, columns(), scratchcolumns()); }
    matrix_view<matrix_storage> submatrix_view(size_t row_offset, size_t rows, size_t column_offset, size_t columns);
        
private:
    void _alloc(size_t rows);
    void _free();
    void* _alloc; // raw memory allocation

    uint8_t* _data; // manually aligned memory
    size_t _stride; // _stride * 8 = _columns + _scratchcolumns

    size_t _rows, _columns, _scratchcolumns;
};

template<typename Storage>
struct storage_traits;

template<>
struct storage_traits<matrix_storage>
{
    typedef uint8_t data_t;
    typedef const uint8_t const_data_t;
    typedef data_t* pointer_t;
    typedef const_data_t* const_pointer_t;

    static const bool modifiable = true;
    static const size_t byte_alignment = 1;
};
template<>
struct storage_traits<const matrix_storage>
{
    typedef const uint8_t data_t;
    typedef const uint8_t const_data_t;
    typedef data_t* pointer_t;
    typedef const_data_t* const_pointer_t;

    static const bool modifiable = false;
    static const size_t byte_alignment = 1;
};




/* matrix & vector views */

// matrix_view_content: wrapper around matrix_view with modify & compare operators
template<typename Storage>
class matrix_view_content {
    matrix_view<Storage>& _matrix_view;
public:
    matrix_view_content(matrix_view<Storage>& matrix_view): _matrix_view(matrix_view) {}
    
    template<typename MatrixView>
    matrix_view& operator=(const MatrixView& matrix) { return _matrix_view.assign(matrix); }
    template<typename MatrixView>
    matrix_view& operator^=(const MatrixView& matrix) { return _matrix_view.add(matrix); }
    
    template<typename MatrixView>
    bool operator==(const MatrixView& matrix) { return _matrix_view.compare(matrix); }
    template<typename MatrixView>
    bool operator!=(const MatrixView& matrix) { return !_matrix_view.compare(matrix); }
};

// matrix_view
template<typename Storage>
class matrix_view {
public:
    using typename storage_traits<Storage>::pointer_t;
    using typename storage_traits<Storage>::const_pointer_t;
    
    /* CONSTRUCT MATRIX VIEW */
    // relative to full storage
    matrix_view(Storage& matrix);
    matrix_view(Storage& matrix, size_t row_offset, size_t rows, size_t column_offset, size_t columns, size_t scratchcolumns = 0);
    // relative to matrix_view
    template<typename Storage2 = Storage>
    matrix_view(matrix_view<Storage2>& matrix);
    template<typename Storage2 = Storage>
    matrix_view(matrix_view<Storage2>& matrix, size_t row_offset, size_t rows, size_t column_offset, size_t columns, size_t scratchcolumns = 0);

    /* MATRIX VIEW OPERATIONS */
    //
    // notes:
    //
    // these do not change content!, only the view
    //
    
    // assign matrix_view (duplicates view, does not copy content!)
    matrix_view& operator=(const matrix_view&) = default;
    matrix_view& operator=(matrix_view&&) = default;
    template<typename Storage2>
    matrix_view& operator=(const matrix_view<Storage2>& matrix); // assign non-const matrix_view to const matrix_view

    size_t rows() const { return _rows; }
    size_t columns() const { return _columns; }
    size_t scratchcolumns() const { return _scratchcolumns; }
    //size_t rows_storage_offset() const { return _row_offset; }
    //size_t columns_storage_offset() const { return _column_offset; }

    template<typename Storage2>
    bool operator==(const matrix_view<Storage2>&);
    template<typename Storage2>
    bool operator!=(const matrix_view<Storage2>&);

    // create submatrix matrix_view, automatically passes on scratchcolumns to the extent possible
    matrix_view submatrix_view(size_t row_offset, size_t rows, size_t column_offset, size_t columns);

    // create vector_view
    vector_view<Storage> operator[](size_t r);
    const vector_view<Storage> operator[](size_t r) const;
    vector_view<Storage> row_view(size_t r, size_t column_offset = 0, size_t columns = ~size_t(0));
    const vector_view<Storage> row_view(size_t r, size_t column_offset = 0, size_t columns = ~size_t(0)) const;

    /* MATRIX OPERATIONS */
    //
    // notes:
    //
    // 1. as-is all operations compile, but non-const operations must throw at runtime on const storage
    // we can later modify function definitions or class definitions to disallow non-const operations on const storage at compile time
    //
    // 2. all are defined as member function
    // where possible also operators using matrix_view_content are defined
    //
    // 3. note that effect on scratch columns is *undefined*: operations may modify scratch columns in any way (eg just partially, or not at all)
    //

    // create matrix_view_content from this matrix_view
    // this supports convenient operators to modify & compare matrix contents
    matrix_view_content<matrix_view> operator*() { return matrix_view_content<matrix_view>(*this); }

    // matrix operations
    void assign(bool b);
    
    template<typename MatrixView>
    void assign(const MatrixView&);
    
    template<typename MatrixView>
    void add(const MatrixView&);
    
    template<typename MatrixView>
    bool compare(const MatrixView&);
    template<typename MatrixView>
    bool operator==(const matrix_view_content<MatrixView>&);
    
    template<typename MatrixView>
    bool operator!=(const matrix_view_content<MatrixView>&);

    template<typename MatrixView>
    void assign_transposed(const MatrixView&);
    
    template<typename MatrixView>
    void add_transposed(const MatrixView&);

    
    // row/column operations
    void assign_row(size_t dst, bool b);
    
    void assign_row(size_t dst, size_t src);
    template<typename MatrixView>
    void assign_row(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void assign_row(size_t dst, const VectorView& vector);

    void add_row(size_t dst, size_t src);
    template<typename MatrixView>
    void add_row(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void add_row(size_t dst, const VectorView& vector);
    
    void and_row(size_t dst, size_t src);
    template<typename MatrixView>
    void and_row(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void and_row(size_t dst, const VectorView& vector);

    void or_row(size_t dst, size_t src);
    template<typename MatrixView>
    void or_row(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void or_row(size_t dst, const VectorView& vector);

    void not_row(size_t dst);

    size_t row_weight(size_t r) const;


    void assign_column(size_t dst, bool b);

    void assign_column(size_t dst, size_t src);
    template<typename MatrixView>
    void assign_column(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void assign_column(size_t dst, const VectorView& vectorTransposed);

    void add_column(size_t dst, size_t src);
    template<typename MatrixView>
    void add_column(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void add_column(size_t dst, const VectorView& vectorTransposed);

    void and_column(size_t dst, size_t src);
    template<typename MatrixView>
    void and_column(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void and_column(size_t dst, const VectorView& vectorTransposed);

    void or_column(size_t dst, size_t src);
    template<typename MatrixView>
    void or_column(size_t dst, const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void or_column(size_t dst, const VectorView& vectorTransposed);

    void not_column(size_t dst);

    size_t column_weight(size_t c) const;

    // bits operations
    bool getbit(size_t r, size_t c) const;
    void setbit(size_t r, size_t c, bool b);
    template<bool b> void setbit(size_t r, size_t c);
    
    // manipulate raw bytes: return pointer to byte containing bit at row r column c
    // WARNING: must ensure alignment to full bytes yourself
    pointer_t data(size_t r, size_t c);
    const_pointer_t data(size_t r, size_t c) const;
    const_pointer_t cdata(size_t r, size_t c) const;
    size_t stride() const;
    
    // find bits
    size_t find_in_row(size_t row, bool b) const;
    size_t find_in_row(size_t row, bool b, size_t column_offset) const;
    size_t find_in_column(size_t column, bool b) const;
    size_t find_in_column(size_t column, bool b, size_t row_offset) const;


    
    // extract rows / columns: operates on minimum dimension (where offset reduces the respective dimension of the matrix_view)
    template<typename MatrixView>
    void extract_matrix(MatrixView&, size_t row_offset = 0, size_t column_offset = 0);
    template<typename VectorView>
    void extract_row(VectorView&, size_t row, size_t column_offset = 0);
    template<typename VectorView>
    void extract_column(VectorView&, size_t column, size_t row_offset = 0);
    
private:
    Storage* _matrix;
    size_t _row_offset, _rows;
    size_t _column_offset, _columns, _scratchcolumns;
};





// vector_view_content: wrapper around vector_view with modify operators
template<typename Storage>
class vector_view_content {
    vector_view<Storage>& _vector_view;
public:
    vector_view_content(vector_view<Storage>& vector_view): _vector_view(vector_view) {}
    
    template<typename VectorView>
    vector_view& operator=(const VectorView& vector) { return _vector_view.assign(vector); }
    template<typename VectorView>
    vector_view& operator^=(const VectorView& vector) { return _vector_view.add(vector); }
    template<typename VectorView>
    vector_view& operator&=(const VectorView& vector) { return _vector_view.and(vector); }
    template<typename VectorView>
    vector_view& operator|=(const VectorView& vector) { return _vector_view.or(vector); }

    template<typename VectorView>
    bool operator==(const VectorView& vector) { return _vector_view.compare(vector); }
    template<typename VectorView>
    bool operator!=(const VectorView& vector) { return !_vector_view.compare(vector); }
};

// vector_view<matrix_storage>, vector_view<const matrix_storage>
template<typename Storage = matrix_storage>
class vector_view {

    /* CONSTRUCT VECTOR VIEW */

    // relative to storage
    vector_view(Storage& matrix, size_t column_offset, size_t columns, size_t row, scratch_columns = 0);
    // relative to matrix_view
    template<typename MatrixView>
    vector_view(MatrixView& matrix, size_t column_offset, size_t columns, size_t row, size_t scratch_columns = 0);
    // relative to vector_view
    template<typename VectorView>
    vector_view(VectorView& vector);
    template<typename VectorView>
    vector_view(VectorView& vector, size_t column_offset, size_t columns, size_t scratch_columns = 0);



    /* VECTOR VIEW OPERATIONS */

    // assign vector_view (duplicates view, does not copy content!)
    vector_view& operator=(const vector_view&) = default;
    vector_view& operator=(vector_view&&) = default;
    template<typename Storage2>
    vector_view& operator=(const vector_view<Storage2>&);

    size_t columns() const { return _columns; }
    size_t scratchcolumns() const { return _scratchcolumns; }
    //size_t rows_storage_offset() const { return _row_offset; }
    //size_t columns_storage_offset() const { return _column_offset; }

    template<typename Storage2>
    bool operator==(const vector_view<Storage2>&);
    template<typename Storage2>
    bool operator!=(const vector_view<Storage2>&);


    
    /* VECTOR CONTENT OPERATIONS */

    // create vector_view_content from this vector_view
    // this supports convenient operators to modify & compare vector contents
    vector_view_content<vector_view> operator*() { return vector_view_content<vector_view>(*this); }

    // vector operations
    void assign(bool b);
    
    template<typename MatrixView>
    void assign(const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void assign(const VectorView& vector);

    template<typename MatrixView>
    void add(const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void add(const VectorView& vector);
    
    template<typename MatrixView>
    void and(const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void and(const VectorView& vector);

    template<typename MatrixView>
    void or(const MatrixView& matrix, size_t src);
    template<typename VectorView>
    void or(const VectorView& vector);

    void not();

    size_t row_weight() const;
    
    // manipulate bits
    bool getbit(size_t c) const;
    void setbit(size_t c, bool b);
    template<bool b> void setbit(size_t c);
    
    // manipulate raw bytes: return pointer to byte containing bit at column c
    // WARNING: must ensure alignment to full bytes yourself
    pointer_t data(size_t c);
    const_pointer_t data(size_t c) const;
    const_pointer_t cdata(size_t c) const;

    // find bits
    size_t find(bool b) const;
    size_t find(bool b, size_t column_offset) const;

    // extract vector
    template<typename VectorView>
    void extract(VectorView&, size_t column_offset = 0);


private:
    Storage* _matrix;
    size_t _column_offset, _columns, _scratchcolumns;
};









void example_code {
    matrix_storage matrix(512, 1024);
    matrix_storage helper_vectors(512, 1024);
    
    matrix_view matview(matrix);
    
    // <load content>
    
    // copy 16x16 submatrix to another submatrix
    * matview.submatrix(0,16,0,16) = matview.submatrix(16,16,0,16);
    // add 16x16 submatrix to another submatrix
    * matview.submatrix(0,16,0,16) ^= matview.submatrix(32,16,0,16);
    
    // one step of gaussian elimination
    size_t pivot = matview.find_in_column(0, 1);
    if (pivot >= matview.rows())
        throw;
    for (size_t i = 0; i < matview.rows(); ++i)
        if (i != pivot && matview(i,0))
            *matview[i] ^= matview[pivot];
    // note above that we use *matview[i] because:
    // matview[i] returns a vector view
    // so *matview[i] gives a vector_view_content using which we can modify content with operators
}

MCCL_END_NAMESPACE

#endif
