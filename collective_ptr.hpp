#include <cassert>
#include <cstdio>
#include <memory>

template<typename T>
  class collective_ptr
{
  public:
    typedef T * pointer;
    typedef T   element_type;

    /*! \pre This thread block shall be converged.
     */
    __device__
    collective_ptr()
    {
      construct_owner();
    }

    /*!
     *  XXX figure out a way for collective_ptr to deallocate p
     *      perhaps there's a way to check a pointer's address to see if it points to __shared__
     *
     *  \param p A pointer to unintialized storage for this collective_ptr to manage.
     *         It is the responsibility of the caller to deallocate the storage after
     *         this collective_ptr is destroyed.
     *
     *  \pre This thread block shall be converged.
     *
     *       The value of p shall be identical for all threads in this thread block.
     */
    __device__
    collective_ptr(pointer p)
    {
      construct_owner();

      barrier();
      if(threadIdx.x == 0)
      {
        reset(p);
      }
      barrier();

      construct_element();
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    ~collective_ptr()
    {
      destroy_element();

      barrier();

      destroy_owner();
    }

    __device__
    T *operator->()
    {
      // gain exclusive access
      assert_atomically_obtain_ownership(threadIdx.x);

      return m_ptr;
    }

    __device__
    const T *operator->() const
    {
      assert_ownership_or_no_owner(threadIdx.x);

      return m_ptr;
    }

    __device__
    T &operator*()
    {
      return *operator->();
    }

    __device__
    const T &operator*() const
    {
      return *operator->();
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    void barrier()
    {
      disown_and_synchronize();
    }

    __device__
    void reset(pointer p)
    {
      // XXX assert that the tag is unowned
      
      m_ptr = p;
    }

  private:
    /*! \pre This thread block shall be converged.
     */
    __device__
    void construct_owner()
    {
      __shared__ int owner;

      __syncthreads();
      if(threadIdx.x == 0)
      {
        m_owner = &owner;
      }
      __syncthreads();

      disown_and_synchronize();
    }

    __device__
    void destroy_owner()
    {
      // nothing to do
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    void construct_element()
    {
      if(threadIdx.x == 0)
      {
        ::new (static_cast<void*>(m_ptr)) T;
      }

      barrier();
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    void destroy_element()
    {
      if(threadIdx.x == 0)
      {
        m_ptr->~T();
      }
      barrier();
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    void disown_and_synchronize()
    {
      const int no_owner = blockDim.x + 1;

      if(threadIdx.x == 0)
      {
        *m_owner = no_owner;
      }

      __syncthreads();
    }

    __device__
    bool atomically_obtain_ownership(int thread_idx)
    {
      const int no_owner = blockDim.x + 1;

      // only grab the tag if it has no other owner
      int old_owner = atomicCAS(m_owner, no_owner, thread_idx);

      // we are the owner if it had no previous owner or if we already owned it
      return (old_owner == no_owner) || (old_owner == thread_idx);
    }

    __device__
    void assert_atomically_obtain_ownership(int thread_idx)
    {
      if(!atomically_obtain_ownership(threadIdx.x))
      {
        // multiple writers
        printf("Write after write hazard detected in thread %d of block %d\n", threadIdx.x, blockIdx.x);
        assert(false);
      }
    }

    __device__
    void assert_ownership_or_no_owner(int thread_idx) const
    {
      const int no_owner = blockDim.x + 1;

      if(*m_owner != no_owner && *m_owner != thread_idx)
      {
        printf("Read after write hazard detected in thread %d of block %d\n", threadIdx.x, blockIdx.x);
        assert(false);
      }
    }

    pointer m_ptr;
    int *m_owner;
};

