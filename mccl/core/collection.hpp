#ifndef MCCL_CORE_COLLECTION_HPP
#define MCCL_CORE_COLLECTION_HPP

#include <mccl/config/config.hpp>

#include <stdlib.h>

#include <new>
#include <cstdint>
#include <atomic>
#include <iterator>
#include <stdexcept>
#include <array>
#include <algorithm>

MCCL_BEGIN_NAMESPACE

namespace page_allocator 
{

	// FIFO concurrent stack implemented as circular buffer
	// stores at most N void* pointers
	// - N must be power of 2
	// - never store more than N elements => throws
	// uses:
	// - _size: amount of elements ready to be popped
	// - _first % N: index of first element to be popped
	// - _last  % N: index beyond the last element ready to be popped
	// - _write % N: index to push the next element 
	template<std::size_t N>
	class concurrent_stack {
	public:
		static_assert( (((N-1) & N) == 0) && (N != 0), "concurrent_stack: N must be power of 2");
		
		typedef void* value_type;
		
		concurrent_stack()
			: _first(0), _last(0), _write(0), _size(0)
		{
			for (std::size_t i = 0; i < N; ++i)
				_buffer[i] = nullptr;
		}
		~concurrent_stack()
		{
			std::size_t size = _size.load();
			std::size_t first = _first.load();
			std::size_t write = _write.load();
			std::size_t last =_last.load();
			// check consistency
			if (last-first != size || last != write)
				throw std::runtime_error("~concurrent_stack(): inconsistency detected");
			// free all elements in storage
			for (; first != last; ++first)
			{
				if (_at(first) == nullptr)
					throw std::runtime_error("~concurrent_stack(): nullptr found");
				this->aligned_free(_at(first));
			}
		}
		
		value_type pop_front()
		{
			// reserve an element in buffer if possible
			// original implementation used a load followed by a while loop with atomic compare_exchange_weak
			// this simply uses an atomic decrease, and in the bad case an atomic increase
			// this might sporadically cause additional allocations under high pop & push contention and near empty stack of free pages,
			//   which is deemed acceptable
			auto size = _size.fetch_sub(1, std::memory_order_acquire);
			if (size <= 0)
			{
				// oops size was 0 or less, so must undo decrease
				_size.fetch_add(1, std::memory_order_release);
				return nullptr;
			}
			// now it is safe to increment _first and use its original value
			std::size_t idx = _first.fetch_add(1, std::memory_order_relaxed);
			value_type p = _at(idx);
			if (p == nullptr)
				throw std::runtime_error("concurrent_stack::pop_front(): unexpected error");
			_at(idx) = nullptr;
			return p;
		}

		void push_back(value_type p)
		{
			if (p == nullptr)
				return;
			std::size_t write = _write.fetch_add(1, std::memory_order_acquire);
			_at(write) = p;
			// there is only contention on _last:
			// we have to wait until previous writes have finished
			while (_last.load(std::memory_order_relaxed) != write)
				;
			_last.fetch_add(1, std::memory_order_release);
			auto size = _size.fetch_add(1, std::memory_order_release);
			if (size >= N-1)
				throw std::runtime_error("concurrent_stack::push_back(): stack overflow");
		}

		// pointers should be allocated through aligned_alloc:
		value_type aligned_alloc(std::size_t size, std::size_t alignment)
		{
			return std::aligned_alloc(size, alignment);
		}
		// pointers should be freed through aligned_free:
		void aligned_free(value_type p)
		{
			std::free( p );
		}

	private:
		value_type& _at(std::size_t idx)
		{
			// perform fast permutation on buffer such that consecutive indices aren't in the same cacheline
			idx *= 17;
			idx %= N;
			return _buffer[idx];
		}
		
		std::array<value_type,N> _buffer;

		std::atomic_size_t _first;
		std::array<char,64-sizeof(std::size_t)> _cachelinepadding1;
		std::atomic_size_t _last;
		std::array<char,64-sizeof(std::size_t)> _cachelinepadding2;
		std::atomic_size_t _write;
		std::array<char,64-sizeof(std::size_t)> _cachelinepadding3;
		std::atomic_int64_t _size;
	};
	
	
	// memory allocator pool for fixed size pages
	class page_allocator_pool {
	public:
		static const std::size_t page_size = 4<<20;   // 4MB pages
		static const std::size_t page_alignment = 64; // cacheline alignment
		static const std::size_t max_pages = 1<<20;   // max 4TB, must be power-of-2

		// obtain page from free page pool, otherwise allocate a new one
		static void* alloc_page()
		{
			void* p = _stack.pop_front();
			return (p != nullptr) 
				? p 
				: _stack.aligned_alloc(page_size, page_alignment);
		}
		// free pages go into pool, not actually freed
		static void free_page(void* p)
		{
			_stack.push_back(p);
		}
		
	private:
		static concurrent_stack<max_pages> _stack;
	};
	
}

template<typename T>
class page_vector
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	typedef page_allocator::page_allocater_pool page_allocator_type;
	static const std::size_t page_size = page_allocator_type::page_size;
	static const std::size_t page_alignment = page_allocator_type::page_alignment;
	static const std::size_t alignment_cost = ((page_alignment % alignof(value_type)) == 0) ? 0 : alignof(value_type);
	static const std::size_t page_capacity = (page_size - alignment_cost) / sizeof(value_type);

	~page_vector()
	{
		_shrink(0);
		_free_page();
	}

	page_vector() noexcept
		: _page(nullptr), _data(nullptr), _size(0)
	{
	}
	page_vector(size_type count) : page_vector()
	{
		if (count > 0)
			_grow(count);
	}
	page_vector(size_type count, const value_type& value) : page_vector()
	{
		if (count > 0)
			_grow(count, value);
	}
	template<typename InputIt>
	page_vector(InputIt first, InputIt last) : page_vector()
	{
		_assign(std::distance(first,last), first, last);
	}
	page_vector(const page_vector& v) : page_vector()
	{
		_assign(v.size(), v.begin(), v.end());
	}
	page_vector(page_vector&& v) noexcept
		: _page(v._page), _data(v._data), _size(v._size)
	{
		v._page = nullptr;
		v._data = nullptr;
		v._size = 0;
	}
	page_vector(std::initializer_list<value_type> init) : page_vector()
	{
		_assign(init.size(), init.begin(), init.end());
	}
	

	page_vector& operator=(const page_vector& v)
	{
		_assign(v.size(), v.begin(), v.end());
		return *this;
	}
	page_vector& operator=(page_vector&& v) noexcept
	{
		std::swap(_data, v._data);
		std::swap(_size, v._size);
		return *this;
	}
	page_vector& operator=(std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
		return *this;
	}
		
	void assign(size_type count, const value_type& value)
	{
		if (count < _size)
			_shrink(count);
		for (auto it = begin(); it != end(); ++it)
			*it = value;
		if (count > _size)
			_grow(count, value);
	}
	
	template<typename InputIt>
	void assign(InputIt first, InputIt last)
	{
		_assign(std::distance(first,last), first, last);
	}
	
	void assign(std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
	}

	reference at(size_type pos)
	{
		if (pos >= _size)
			throw std::out_of_range("page_vector::at(): pos >= size()");
		return *(_data+pos);
	}
	const_reference at(size_type pos) const
	{
		if (pos >= _size)
			throw std::out_of_range("page_vector::at(): pos >= size()");
		return *(_data+pos);
	}
	reference operator[](size_type pos)
	{
		return *(_data+pos);
	}
	const_reference operator[](size_type pos) const
	{
		return *(_data+pos);
	}
	
	      reference front()       noexcept { return *_data; }
	const_reference front() const noexcept { return *_data; }
	      reference back()       noexcept { return *(_data+_size-1); }
	const_reference back() const noexcept { return *(_data+_size-1); }
	      pointer data()       noexcept { return _data; }
	const_pointer data() const noexcept { return _data; }
	
	      iterator  begin()       noexcept { return _data; }
	const_iterator  begin() const noexcept { return _data; }
	const_iterator cbegin() const noexcept { return _data; }
	      iterator  end()       noexcept { return _data+_size; }
	const_iterator  end() const noexcept { return _data+_size; }
	const_iterator cend() const noexcept { return _data+_size; }

	      reverse_iterator  rbegin()       noexcept { return reverse_iterator(end()); }
	const_reverse_iterator  rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	      reverse_iterator  rend()       noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator  rend() const noexcept { return const_reverse_iterator(begin()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	bool empty() const { return _size == 0; }
	size_type size() const { return _size; }
	size_type max_size() const { return page_capacity; }
	size_type capacity() const { return _page == nullptr ? 0 : page_capacity; }
	
	void reserve(size_type new_cap)
	{
		if (new_cap > 0 && _page == nullptr)
			_alloc_page();
	}
	
	void shrink_to_fit()
	{
		if (_size == 0 && _page != nullptr)
			_free_page();
	}
	
	void clear()
	{
		_shrink(0);
	}
	
	iterator insert(const_iterator pos, const value_type& value)
	{
		return _insert_single(pos, value);
	}
	
	iterator insert(const_iterator pos, value_type&& value)
	{
		return _insert_single(pos, std::move(value));
	}
	
	iterator insert(const_iterator pos, size_type count, const value_type& value)
	{
		return _insert_multi(pos, count, value);
	}
	
	template<typename InputIt>
	iterator insert(const_iterator pos, InputIt first, InputIt last)
	{
		return _insert_multi(pos, std::distance(first, last), first, last);
	}
	
	iterator insert(const_iterator pos, std::initializer_list<value_type> ilist)
	{
		return _insert_multi(pos, ilist.size(), ilist.begin(), ilist.end());
	}
	
	template<typename... Args>
	iterator emplace(const_iterator pos, Args&&... args)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		if (pos == end())
		{
			emplace_back(std::forward<Args>(args)...);
			return end()-1;
		}
		return insert(pos, value_type(std::forward<Args>(args)...));
	}
	
	iterator erase(const_iterator pos)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) < _size);
		for (auto it = _iterator(pos); it != end()-1; ++it)
			*it = std::move(*(it+1));
		pop_back();
		return _iterator(pos);
	}
	
	iterator erase(const_iterator first, const_iterator last)
	{
		assert((first-begin()) >= 0 && (last-begin()) <= _size);
		if (first != last)
		{
			auto it = _iterator(first);
			for (; last != end(); ++it,++last)
				*it = std::move(*last);
			_shrink(it - begin());
		}
		return _iterator(first);
	}
	
	void push_back(const value_type& value)
	{
		if (_size == page_capacity)
			throw std::runtime_error("page_vector::push_back(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		_construct(_size, value);
		++_size;
	}

	void push_back(value_type&& value)
	{
		if (_size == page_capacity)
			throw std::runtime_error("page_vector::push_back(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		_construct(_size, std::move(value));
		++_size;
	}

	template<typename... Args>
	void emplace_back(Args&&... args)
	{
		if (_size == page_capacity)
			throw std::runtime_error("page_vector::emplace_back(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		_construct(_size, std::forward<Args>(args)...);
		++_size;
	}
	
	void pop_back()
	{
		assert(_size > 0);
		--_size;
		_destruct(_size);
	}
	
	void resize(size_type count)
	{
		if (count < _size)
			_shrink(count);
		else if (count > _size)
			_grow(count);
	}
	
	void resize(size_type count, const value_type& value)
	{
		if (count < _size)
			_shrink(count);
		else if (count > _size)
			_grow(count, value);
	}

	void swap(page_vector& other)
	{
		std::swap(_page, other._page);
		std::swap(_data, other._data);
		std::swap(_size, other._size);
	}	

private:
	// upgrade a const_iterator into an iterator
	iterator _iterator(const_iterator it) 
	{
		return begin()+(it-begin());
	}
	
	template<typename T>
	iterator _insert_single(const_iterator pos, T&& value)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		// simplest case: insert at end, in particular when _size == 0
		if (pos == end())
		{
			emplace_back(value);
			return end()-1;
		}
		// case _size >= 1 && begin() <= pos < end():
		emplace_back(std::move(back()));
		// pos <= end()-2
		auto it = end()-2;
		for (; it != pos; --it)
			*it = std::move(*(it-1));
		*it = std::forward<T>(value);
		return it;
	}

	iterator _insert_multi(const_iterator pos, size_type count, const value_type& value)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		if (_size+count > page_capacity)
			throw std::runtime_error("page_vector::_insert_multi(): capacity exceeded");
		size_type movecount = end()-pos;
		// handle quick path: insert at end
		// but also potentially needs page allocation which invalidates pos
		if (movecount == 0)
		{
			if (_page == nullptr)
				_alloc_page();
			for (; count > 0; --count)
				emplace_back(value);
			return end();
		}
		// movecount > 0: thus no page allocation needed and pos will not be invalidated
		if (movecount <= count)
		{
			// in this case: pos+count >= end, thus:
			// - existing elements [pos,end) are moved to newly constructed places [pos+count,end+count)
			// - [pos,end): copy value into existing elements
			// - [end,pos+count): construct new elements with value
			auto it = _iterator(pos);
			for (; it != end(); ++it)
			{
				_construct(it+count, std::move(*it));
				*it = value;
			}
			for (; it != _iterator(pos)+count; ++it)
				_construct(it, value);
			_size += count;
		} else
		{
			// in this case: pos+count < end, thus:
			// - existing elements [end-count,end) are moved to newly constructed places [end,end+count)
			// - existing elements [pos,end-count) are moved to [pos+count,end), backwards
			// - [pos,pos+count): copy value into existing elements
			auto it = end()-1;
			for (; it != end()-count-1; --it)
				_construct(it+count, std::move(*it));
			for (; it != pos-1; --it)
				*(it+count) = std::move(*it);
			for (++it; it != pos+count; ++it)
				*it = value;
			_size += count;
		}
		return _iterator(pos);
	}

	template<typename InputIt>
	iterator _insert_multi(const_iterator pos, size_type count, InputIt first, InputIt last)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		if (_size+count > page_capacity)
			throw std::runtime_error("page_vector::_insert_multi(): capacity exceeded");
		size_type movecount = end()-pos;
		// handle quick path: insert at end
		// but also potentially needs page allocation which invalidates pos
		if (movecount == 0)
		{
			if (_page == nullptr)
				_alloc_page();
			for (; count > 0; --count)
				emplace_back(value);
			return end();
		}
		// movecount > 0: thus no page allocation needed and pos will not be invalidated
		if (movecount <= count)
		{
			// in this case: pos+count >= end, thus:
			// - existing elements [pos,end) are moved to newly constructed places [pos+count,end+count)
			// - [pos,end): copy value into existing elements
			// - [end,pos+count): construct new elements with value
			auto it = _iterator(pos);
			for (; it != end(); ++it,++first)
			{
				_construct(it+count, std::move(*it));
				*it = *first;
			}
			for (; it != _iterator(pos)+count; ++it,++first)
				_construct(it, *first);
			_size += count;
		} else
		{
			// in this case: pos+count < end, thus:
			// - existing elements [end-count,end) are moved to newly constructed places [end,end+count)
			// - existing elements [pos,end-count) are moved to [pos+count,end), backwards
			// - [pos,pos+count): copy value into existing elements
			auto it = end()-1;
			for (; it != end()-count-1; --it)
				_construct(it+count, std::move(*it));
			for (; it != pos-1; --it)
				*(it+count) = std::move(*it);
			for (++it; it != pos+count; ++it,++first)
				*it = *first;
			_size += count;
		}
		return _iterator(pos);
	}

	template<typename InputIt>
	void _assign(size_type count, InputIt first, InputIt last)
	{
		if (count < _size)
			_shrink(count);
		for (it = begin(); it != end(); ++it,++first)
			*it = *first;
		for (; first != last; ++first)
			emplace_back(*first);
	}

	// shrink vector
	// must be called with count <= _size
	void _shrink(size_type count)
	{
		assert(count <= _size);
		for (auto p = count; p != _size; ++p)
			_destruct(p);
		_size = count;
	}

	// grow vector
	// must be called with count > _size
	void _grow(size_type count)
	{
		assert(count > _size);
		if (count > page_capacity)
			throw std::runtime_error("page_vector::_grow(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		for (auto p = _size; p != count; ++p)
			_construct(p);
		_size = count;
	}

	// grow vector
	// must be called with count > _size
	void _grow(size_type count, const value_type& value)
	{
		assert(count > _size);
		if (count > page_capacity)
			throw std::runtime_error("page_vector::_grow(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		for (auto p = _size; p != count; ++p)
			_construct(p, value);
		_size = count;
	}
	
	template<typename... Args>
	void _construct(size_type pos, Args&&... args)
	{
		new (_data+pos) value_type(std::forward<Args>(args)...);
	}
	template<typename... Args>
	void _construct(iterator pos, Args&&... args)
	{
		new (pos) value_type(std::forward<Args>(args)...);
	}
	void _destruct(size_type pos)
	{
		(_data+pos)->~value_type();
	}
	
	void _alloc_page()
	{
		if (_page != nullptr)
			return;
		// obtain page
		_page = page_allocator_type::alloc_page();
		if (_page == nullptr)
			throw std::runtime_error("page_vector::_alloc_page(): allocation failed");
		// ensure alignment for value_type
		std::uintptr_t data = std::uintptr_t(_page);
		data += sizeof(value_type);
		data -= data % alignof(value_type);
		// set _data: casts between uintptr_t and value_type* must use intermediate cast to void*
		_data = static_cast<value_type*>( static_cast<void*>(data) );
		_size = 0;
	}
	
	void _free_page()
	{
		assert(_size == 0);
		if (_page == nullptr)
			return;
		page_allocator_type::free_page(_page);
		_page = nullptr;
		_data = nullptr;
	}

	void*       _page;
	value_type* _data;
	size_type   _size;
}

template<typename T>
bool operator == (const page_vector<T>& lhs, const page_vector<T>& rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	for (auto it1=lhs.begin(),it2=rhs.begin(),it1end=lhs.end(); it1 != it1end; ++it1,++it2)
		if (*it1 != *it2)
			return false;
	return true;
}

template<typename T>
bool operator <  (const page_vector<T>& lhs, const page_vector<T>& rhs)
{
	return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template<typename T>
bool operator != (const page_vector<T>& lhs, const page_vector<T>& rhs) { return !(lhs == rhs); }
template<typename T>
bool operator >= (const page_vector<T>& lhs, const page_vector<T>& rhs) { return !(lhs < rhs); }
template<typename T>
bool operator >  (const page_vector<T>& lhs, const page_vector<T>& rhs) { return rhs < lhs; }
template<typename T>
bool operator <= (const page_vector<T>& lhs, const page_vector<T>& rhs) { return !(rhs < lhs); }


MCCL_END_NAMESPACE

#endif
