#ifndef MCCL_CORE_MATRIX_STORAGE_HPP
#define MCCL_CORE_MATRIX_STORAGE_HPP

#include <mccl/config/config.hpp>
#include <algorithm>
#include <cstring>
#include <cassert>

MCCL_BEGIN_NAMESPACE

// matrix_storage allocates & manages memory for matrices
class matrix_storage;
// storage traits: allocation configuration & helps distinguishing between non-const/const matrix_storage
template<typename Storage> struct storage_traits;

class matrix_storage {
public:
    matrix_storage(size_t rows, size_t columns) { this->_alloc(rows, columns); }
    ~matrix_storage() { this->_free(); }
    matrix_storage(const matrix_storage& ms)
    {
        this->_alloc(ms.rows(), ms.allcolumns());
        memcpy(data(), ms.data(), ms.rows() * stride());
    }
    matrix_storage(matrix_storage&& ms)
    {
        this->_alloc(0,0);
        this->swap(ms);
    }
    
    size_t rows()           const { return _rows; }
    size_t columns()        const { return _columns; }
    size_t scratchcolumns() const { return _scratchcolumns; }
    size_t allcolumns()     const { return _columns + _scratchcolumns; }
    size_t stride()         const { return _stride; }

          uint8_t* data()                         { return _data; }
    const uint8_t* data()                   const { return _data; }
          uint8_t* data(size_t r)                 { return _data + r*stride(); }
    const uint8_t* data(size_t r)           const { return _data + r*stride(); }
          uint8_t* data(size_t r, size_t c)       { return _data + r*stride() + (c/8); }
    const uint8_t* data(size_t r, size_t c) const { return _data + r*stride() + (c/8); }
    
    
    void swap(matrix_storage& m)
    {
        std::swap(_ptr, m._ptr);
        std::swap(_data, m._data);
        std::swap(_stride, m._stride);
        std::swap(_rows, m._rows);
        std::swap(_columns, m._columns);
        std::swap(_scratchcolumns, m._scratchcolumns);
    }
    
private:
    void _alloc(size_t rows, size_t columns);
    void _free();
    
    // pointer to raw allocated memory
    uint8_t* _ptr;

    // pointer to start of aligned data
    uint8_t* _data;
    // _stride * 8 = _columns + _scratchcolumns
    size_t _stride;

    // dimensions: _rows x (_columns + _scratchcolumns)
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
    static const size_t alignment = 6; // align on 1<<6 bytes
    static const size_t column_multiple = 512;
};
template<>
struct storage_traits<const matrix_storage>
{
    typedef const uint8_t data_t;
    typedef const uint8_t const_data_t;
    typedef data_t* pointer_t;
    typedef const_data_t* const_pointer_t;

    static const bool modifiable = false;
    static const size_t alignment = 6; // align on 1<<6 bytes
    static const size_t column_multiple = 512;
};



void matrix_storage::_alloc(size_t rows, size_t columns)
{
    const size_t column_multiple = storage_traits<matrix_storage>::column_multiple;
    const size_t alignment = storage_traits<matrix_storage>::alignment;
    assert(column_multiple % 8 == 0);
    const uintptr_t alignmask = (uintptr_t(1)<<alignment) - 1;
    
    _rows = rows;
    _columns = _columns;

    columns = ((columns + column_multiple - 1) / column_multiple) * column_multiple;
    _scratchcolumns = columns - _columns;

    _stride = columns / 8;
    size_t bytes = rows * _stride;
    if (bytes == 0)
    {
        _ptr = _data = nullptr;
    } else
    {
        bytes += alignmask;
        _ptr = new uint8_t[bytes];
        _data = (uint8_t*)( (uintptr_t(_ptr)+alignmask)&~alignmask );
        // clear all bits in storage
        memset(_data, 0, _rows*_stride);
#ifndef NDEBUG
        // sets bits outside storage to detect memory corruption
        for (auto p = _ptr; p != _data; ++p)
          *p = 0xFF;
        for (auto p = _data + _rows*_stride; p != _ptr + bytes; ++p)
          *p = 0xFF;
#endif
    }
}

void matrix_storage::_free()
{
    if (_ptr != nullptr)
    {
#ifndef NDEBUG
        const size_t alignment = storage_traits<matrix_storage>::alignment;
        const uintptr_t alignmask = (uintptr_t(1)<<alignment) - 1;
        size_t bytes = _rows * _stride + alignmask;
        // check for memory corruption
        for (auto p = _ptr; p != _data; ++p)
          assert(*p == 0xFF);
        for (auto p = _data + _rows*_stride; p != _ptr + bytes; ++p)
          assert(*p == 0xFF);
#endif
        delete[] _ptr;
    }
}

MCCL_END_NAMESPACE

#endif
