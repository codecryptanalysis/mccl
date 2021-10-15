#ifndef MCCL_ALIGNED_VECTOR_HPP
#define MCCL_ALIGNED_VECTOR_HPP

#include <mccl/config/config.hpp>

#include <cstdlib>
#include <vector>

MCCL_BEGIN_NAMESPACE

namespace detail
{

    /*
        allocates sufficient memory for size bytes aligned on alignment 
        plus enough bytes to store a pointer just before
        note: alignment must be power of 2
    */
    inline void* mccl_aligned_malloc(size_t size, size_t alignment)
    {
        void* res = nullptr;
        // allocate sufficient number of bytes
        void* ptr = std::malloc(size + alignment + sizeof(void*));
        if (ptr != nullptr)
        {
            // first skip enough bytes to store a pointer
            uintptr_t tmp = reinterpret_cast<uintptr_t>(ptr) + sizeof(void*);
            // then round up to alignment
            tmp = (tmp | (alignment-1)) + 1;
            res = reinterpret_cast<void*>(tmp);
            // store original malloc ptr just before the memory address pointed by res
            * (reinterpret_cast<void**>(res)-1) = ptr;
        }
        return res;
    }
    
    /* frees memory allocated by mccl_alligned_malloc */
    inline void mccl_aligned_free(void* ptr)
    {
        if (ptr != nullptr)
        {
            // obtain original malloc ptr just before just memory address pointer by ptr
            ptr = * (reinterpret_cast<void**>(ptr)-1);
            // free memory
            std::free( ptr );
        }
    }

    /* 
        A C++11 compatible allocated that does aligned allocation
        note: alignof(T) must be power of 2
    */
    template<typename T>
    struct aligned_allocator
    {
        typedef T  value_type;
        typedef T& reference;
        typedef T* pointer;
        typedef const T&  const_reference;
        typedef const T*  const_pointer;
        typedef size_t    size_type;
        typedef ptrdiff_t difference_type;
        
        template<typename U>
        struct rebind
        {
            typedef aligned_allocator<U> other;
        };

        ~aligned_allocator() noexcept {}
        
        aligned_allocator() noexcept {}
        aligned_allocator(const aligned_allocator&) noexcept {}
        template<typename U>
        aligned_allocator(const aligned_allocator<U>&) noexcept {}
        
              pointer address(      reference r) { return &r; }
        const_pointer address(const_reference r) { return &r; }
        
        template<typename ...Args>
        void construct(pointer p, Args&&... args) { new (p) value_type(std::forward<Args>(args)...); }
        void destroy(pointer p) { p->~value_type(); }
        
        size_type max_size() const noexcept { return size_type(-1) / sizeof(T); }
        
        inline bool operator==(const aligned_allocator&) const { return true; }
        inline bool operator!=(const aligned_allocator&) const { return false; }

        pointer allocate(size_type n)
        {
            pointer ret = reinterpret_cast<pointer>( mccl_aligned_malloc(sizeof(T)*n,alignof(T)) );
            if (ret == nullptr)
                throw std::bad_alloc();
            return ret;
        }
        
        void deallocate(pointer p, size_type)
        {
            mccl_aligned_free(p);
        }

    };
    
} // namespace detail

template<typename T>
using aligned_vector = std::vector<T, detail::aligned_allocator<T>>;

MCCL_END_NAMESPACE

#endif
