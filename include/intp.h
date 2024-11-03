// 2031617b0148cf4ab19c1374d16b04e087e6a025

#ifndef INTP_INTERPOLATION
#define INTP_INTERPOLATION

#include <cmath>  // ceil
#include <initializer_list>

#ifdef INTP_TRACE
#ifndef INTP_DEBUG
#define INTP_DEBUG
#endif
#endif

// Begin content of InterpolationTemplate.hpp
#ifndef INTP_TEMPLATE
#define INTP_TEMPLATE

// Begin content of BSpline.hpp
#ifndef INTP_BSPLINE
#define INTP_BSPLINE

#include <algorithm>  // upper_bound
#include <array>
#include <cmath>        // fmod
#include <functional>   // ref
#include <iterator>     // distance
#include <type_traits>  // is_same, is_arithmatic
#include <vector>

#ifdef INTP_MULTITHREAD
// Begin content of DedicatedThreadPool.hpp
#ifndef INTP_THREAD_POOL
#define INTP_THREAD_POOL

#include <future>   //unique_lock, packaged_task
#include <mutex>    // mutex
#include <queue>    // deque
#include <thread>   // hardware_concurrency
#include <utility>  // move

namespace intp {

/**
 * @brief A thread pool managing a bunch of threads and a queue of tasks. It can
 * spawn a specified number of threads during construction and joins all the
 * threads in destruction. During its lifetime, it can accepts and dispatch
 * tasks to those threads.
 *
 * @tparam T The return type of tasks
 */
template <typename T>
class DedicatedThreadPool {
   private:
    using lock_type = std::unique_lock<std::mutex>;
    using task_type = std::packaged_task<T()>;

    /**
     * @brief A queue of tasks that can be stolen from. Tasks are pushed and
     * popped in back, and being stolen from front.
     *
     */
    struct shared_working_queue {
       public:
#ifdef INTP_DEBUG
        size_t submitted{};
        size_t executed{};
        size_t stolen{};
        size_t stealing{};
#endif

        shared_working_queue() = default;
        void push(task_type&& task) {
            lock_type lk(deq_mutex);
#ifdef INTP_DEBUG
            ++submitted;
#endif
            deq.push_back(std::move(task));
        }
        bool try_pop(task_type& task) {
            lock_type lk(deq_mutex);
            if (deq.empty()) { return false; }
            task = std::move(deq.back());
            deq.pop_back();
#ifdef INTP_DEBUG
            ++executed;
#endif
            return true;
        }
        bool try_steal(task_type& task) {
            lock_type lk(deq_mutex);
            if (deq.empty()) { return false; }
            task = std::move(deq.front());
            deq.pop_front();
#ifdef INTP_DEBUG
            ++stolen;
#endif
            return true;
        }
        bool empty() const {
            bool empty_;
            {
                lock_type lk(deq_mutex);
                empty_ = deq.empty();
            }
            return empty_;
        }

       private:
        std::deque<task_type> deq;
        mutable std::mutex deq_mutex;
    };

    /**
     * @brief Construct a new Thread Pool
     *
     * @param num Thread number in the pool
     */
    DedicatedThreadPool(size_t num = std::thread::hardware_concurrency())
        : join_threads(threads) {
        auto t_num = num == 0 ? DEFAULT_THREAD_NUM : num;
        try {
            for (size_t i = 0; i < t_num; i++) {
                worker_queues.emplace_back(new shared_working_queue{});
            }
            for (size_t i = 0; i < t_num; i++) {
                threads.emplace_back(&DedicatedThreadPool::thread_loop, this,
                                     i);
            }
#ifdef INTP_DEBUG
            std::cout << "[DEBUG] Thread pool initialized with " << t_num
                      << " threads.\n";
#endif
        } catch (...) {  // in this case dtor is not called, so threads
                         // termination should be proper handled here
            should_terminate = true;
            cv.notify_all();
            throw;
        }
    }

   public:
    using return_type = T;

    ~DedicatedThreadPool() {
        should_terminate = true;
        cv.notify_all();

#ifdef INTP_DEBUG
        std::cout << "\n[DEBUG] The thread pool has " << thread_num()
                  << " threads.\n"
                  << "[DEBUG]    Main queue has " << submitted
                  << " submission.\n"
                  << "[DEBUG] Worker queue stats:\n"
                  << "[DEBUG]\t\tSubmitted  Executed  Stolen  Stealing\n";
        for (size_t i = 0; i < worker_queues.size(); ++i) {
            auto ptr = worker_queues[i].get();
            std::cout << "[DEBUG] Thread " << i << ": \t" << ptr->submitted
                      << '\t' << ptr->executed + ptr->stealing << '\t'
                      << ptr->stolen << '\t' << ptr->stealing << '\n';
        }
        std::cout << "[DEBUG] NOTE: Submitted = Executed - Stolen + Stealing\n";
#endif
    }

    /**
     * @brief Add a task in queue, the Func type should not have any parameters
     * (use lambda capture). Return a future containing function return value.
     */
    template <typename Func>
    std::future<return_type> queue_task(Func func) {
        task_type task(std::move(func));
        auto res = task.get_future();
        {
            lock_type lk(main_queue_mutex);
            main_queue.push(std::move(task));
#ifdef INTP_DEBUG
            ++submitted;
#endif
        }
        cv.notify_one();
        return res;
    }

    /**
     * @brief Return true if there are no tasks in queue. (But there may be
     * tasks being executing by threads.)
     */
    bool is_main_queue_empty() {
        bool queue_empty;
        {
            lock_type lk(main_queue_mutex);
            queue_empty = main_queue.empty();
        }
        return queue_empty;
    }

    size_t thread_num() const { return threads.size(); }

    /**
     * @brief Singleton style instance getter
     *
     * @param num Thread number in the pool, default to be
     * hardware_concurrency()
     */
    static DedicatedThreadPool& get_instance(
        size_t num = std::thread::hardware_concurrency()) {
        static DedicatedThreadPool thread_pool(num);
        return thread_pool;
    }

   private:
    static constexpr size_t DEFAULT_THREAD_NUM = 8;
    static constexpr size_t BATCH_SIZE = 16;

    /**
     * @brief Hold a ref of thread vector, use RAII to ensure all the threads is
     * joined.
     *
     */
    struct JoinThreads {
        std::vector<std::thread>& ts_;

        JoinThreads(std::vector<std::thread>& ts) : ts_(ts) {}
        JoinThreads(const JoinThreads&) = delete;
        JoinThreads(JoinThreads&) = delete;
        JoinThreads& operator=(JoinThreads&) = delete;
        ~JoinThreads() {
            for (auto& t : ts_) {
                if (t.joinable()) { t.join(); }
            }
        }
    };

    bool try_pop_from_main(task_type& task) {
        {
            lock_type lk(main_queue_mutex);
            if (main_queue.empty()) { return false; }
            size_t c{};
            while (c++ < BATCH_SIZE && !main_queue.empty()) {
                worker_queue_ptr->push(std::move(main_queue.front()));
                main_queue.pop();
            }
        }
        return worker_queue_ptr->try_pop(task);
    }

    bool try_steal_from_others(task_type& task) {
        for (size_t i = 0; i < worker_queues.size() - 1; ++i) {
            const auto idx = (thread_idx + i + 1) % worker_queues.size();
            if (worker_queues[idx]->try_steal(task)) {
#ifdef INTP_DEBUG
                ++(worker_queue_ptr->stealing);
#endif
                return true;
            }
        }
        return false;
    }

    /**
     * @brief  scheduler function
     *
     */
    void thread_loop(size_t idx) {
        thread_idx = idx;
        worker_queue_ptr = worker_queues[idx].get();
        task_type task;
        // Fetch task from local queue, main queue and other thread's local
        // queue in order.
        while (!should_terminate) {
            if (worker_queue_ptr->try_pop(task) || try_pop_from_main(task) ||
                try_steal_from_others(task)) {
                task();
            } else {
                lock_type lk(main_queue_mutex);
                // Wait until there are tasks in main queue (awaked by
                // notify_one in queue_task), or the thread pool is being
                // shutdown (awaked by notify_all in destructor)
                cv.wait(lk, [this] {
                    return !main_queue.empty() || should_terminate;
                });
            }
        }
    }

#ifdef INTP_DEBUG
    size_t submitted{};
#endif
    bool should_terminate = false;  // Tells threads to stop looking for tasks
    std::mutex main_queue_mutex;    // Protects main task queue
    std::condition_variable cv;     // Signals for thread sleep/awake
    std::vector<std::thread> threads;  // Thread container

    std::queue<task_type> main_queue;  // Main queue for tasks
#if __cplusplus >= 201703L
    inline static thread_local std::size_t thread_idx{};
    // Local queue ptr for tasks
    inline static thread_local shared_working_queue* worker_queue_ptr{};
#else
    static thread_local std::size_t thread_idx;
    static thread_local shared_working_queue* worker_queue_ptr;
#endif
    std::vector<std::unique_ptr<shared_working_queue>> worker_queues;

    JoinThreads join_threads;  // Defined last to ensure destruct first
};

#if __cplusplus < 201703L
template <typename T>
thread_local std::size_t DedicatedThreadPool<T>::thread_idx{};

template <typename T>
thread_local typename DedicatedThreadPool<T>::shared_working_queue*
    DedicatedThreadPool<T>::worker_queue_ptr{};
#endif

}  // namespace intp

#endif  // INTP_THREAD_POOL
// End content of "DedicatedThreadPool.hpp"
#endif

#ifdef INTP_DEBUG
#include <iostream>
#endif

// Begin content of Mesh.hpp
#ifndef INTP_MESH
#define INTP_MESH

#include <array>
#include <vector>

// Begin content of util.hpp
#ifndef INTP_UTIL
#define INTP_UTIL

#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace intp {

namespace util {

/**
 * @brief Polyfill for C++14 integer_sequence, but with [T = size_t] only
 *
 */
template <size_t... Indices>
struct index_sequence {
    using val_type = size_t;
    const static size_t size = sizeof...(Indices);
};

template <size_t N, size_t... Indices>
struct make_index_sequence_impl
    : make_index_sequence_impl<N - 1, N - 1, Indices...> {};

template <size_t... Indices>
struct make_index_sequence_impl<0, Indices...> {
    using type = index_sequence<Indices...>;
};

template <size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

template <typename... T>
using make_index_sequence_for = make_index_sequence<sizeof...(T)>;

/**
 * @brief Compile time power for unsigned exponent
 *
 */
template <typename T1, typename T2>
constexpr typename std::enable_if<std::is_unsigned<T2>::value, T1>::type pow(
    T1 base,
    T2 exp) {
    return exp == 0 ? T1{1} : base * pow(base, exp - 1);
}

template <typename Func, typename... Args, unsigned... indices>
void dispatch_indexed_helper(index_sequence<indices...>,
                             Func& func,
                             Args&&... args) {
    // polyfill of C++17 fold expression over comma
    std::array<std::nullptr_t, sizeof...(Args)>{
        (func(indices, std::forward<Args>(args)), nullptr)...};
}

/**
 * @brief dispatch_indexed(f, x0, x1, ...) invokes f(0, x0), f(1, x1), ..., and
 * ignores their return values.
 *
 */
template <typename Func, typename... Args>
void dispatch_indexed(Func&& func, Args&&... args) {
    dispatch_indexed_helper(util::make_index_sequence_for<Args...>{}, func,
                            std::forward<Args>(args)...);
}

#ifdef INTP_STACK_ALLOCATOR

/**
 * @brief A simple stack allocator, with fixed size and a LIFO allocation
 * strategy, by courtesy of Charles Salvia, in his SO answer
 * https://stackoverflow.com/a/28574062/7255197 .
 *
 * @tparam T allocation object type
 * @tparam N buffer size
 */
template <typename T, std::size_t N>
class stack_allocator {
   public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;

    using const_void_pointer = const void*;

   private:
    pointer m_begin;
    pointer m_end;
    pointer m_stack_pointer;

   public:
    explicit stack_allocator(pointer buffer)
        : m_begin(buffer), m_end(buffer + N), m_stack_pointer(buffer){};

    template <typename U>
    stack_allocator(const stack_allocator<U, N>& other)
        : m_begin(other.m_begin),
          m_end(other.m_end),
          m_stack_pointer(other.m_stack_pointer) {}

    constexpr static size_type capacity() { return N; }

    pointer allocate(size_type n,
                     const_void_pointer hint = const_void_pointer()) {
        if (n <= size_type(m_end - m_stack_pointer)) {
            pointer result = m_stack_pointer;
            m_stack_pointer += n;
            return result;
        }
        throw std::bad_alloc{};
    }

    void deallocate(pointer ptr, size_type n) { m_stack_pointer -= n; }

    size_type max_size() const noexcept { return N; }

    pointer address(reference x) const noexcept { return std::addressof(x); }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    template <typename U>
    struct rebind {
        using other = stack_allocator<U, N>;
    };

    pointer buffer() const noexcept { return m_begin; }
};

template <typename T, std::size_t N, typename U>
bool operator==(const stack_allocator<T, N>& lhs,
                const stack_allocator<U, N>& rhs) noexcept {
    return lhs.buffer() == rhs.buffer();
}

template <typename T, std::size_t N, typename U>
bool operator!=(const stack_allocator<T, N>& lhs,
                const stack_allocator<U, N>& rhs) noexcept {
    return !(lhs == rhs);
}

#endif

struct _is_iterable_impl {
    template <typename T_,
              typename = typename std::enable_if<std::is_convertible<
                  typename std::iterator_traits<
                      decltype(std::declval<T_&>().begin())>::iterator_category,
                  std::input_iterator_tag>::value>::type,
              typename = typename std::enable_if<std::is_convertible<
                  typename std::iterator_traits<
                      decltype(std::declval<T_&>().end())>::iterator_category,
                  std::input_iterator_tag>::value>::type>
    static std::true_type test_(int);

    template <typename>
    static std::false_type test_(...);
};

struct _is_indexed_impl {
    template <typename T_,
              typename = decltype(std::declval<const T_&>().operator[](0))>
    static std::true_type test_(int);

    template <typename>
    static std::false_type test_(...);
};

/**
 * @brief Check if a type has begin() and end() method that returns iterator
 *
 * @tparam T a type to check
 */
template <typename T>
struct is_iterable : _is_iterable_impl {
    static constexpr bool value = decltype(test_<T>(0))::value;
};

template <typename T>
struct is_indexed : _is_indexed_impl {
    static constexpr bool value = decltype(test_<T>(0))::value;
};

/**
 * @brief Polyfill for C++20 stl function with the same name
 *
 */
template <typename Iter>
using remove_cvref_t =
    typename std::remove_cv<typename std::remove_reference<Iter>::type>::type;

#if __cplusplus >= 201402L
#define CPP14_CONSTEXPR_ constexpr
#else
#define CPP14_CONSTEXPR_
#endif

#if __cplusplus >= 201703L
#define CPP17_CONSTEXPR_ constexpr
#else
#define CPP17_CONSTEXPR_
#endif

/**
 * @brief CRTP helper, used for downward casting.
 *
 */
template <typename T, typename...>
struct CRTP {
    CPP14_CONSTEXPR_ T& cast() { return static_cast<T&>(*this); }
    CPP14_CONSTEXPR_ const T& cast() const {
        return static_cast<const T&>(*this);
    }
};

template <bool B,
          template <typename...>
          class TrueTemplate,
          template <typename...>
          class FalseTemplate,
          typename... Args>
struct lazy_conditional;

template <template <typename...> class TrueTemplate,
          template <typename...>
          class FalseTemplate,
          typename... Args>
struct lazy_conditional<true, TrueTemplate, FalseTemplate, Args...> {
    using type = TrueTemplate<Args...>;
};

template <template <typename...> class TrueTemplate,
          template <typename...>
          class FalseTemplate,
          typename... Args>
struct lazy_conditional<false, TrueTemplate, FalseTemplate, Args...> {
    using type = FalseTemplate<Args...>;
};

#ifdef INTP_DEBUG
#define INTP_ENABLE_ASSERTION
#endif

#ifdef INTP_ENABLE_ASSERTION
#define INTP_ASSERT(assertion, msg)                          \
    do {                                                     \
        if (!(assertion)) { throw std::runtime_error(msg); } \
    } while (0)
#else
#define INTP_ASSERT(assertion, msg)
#endif

/**
 * @brief Get the being/end iterator pair of a (stl) container
 *
 * @tparam T Container type
 * @param c Container
 */
template <typename T>
inline auto get_range(T& c)
    -> std::pair<decltype(c.begin()), decltype(c.end())> {
    // Use trailing return type to be C++11 compatible.
    return std::make_pair(c.begin(), c.end());
}

/**
 * @brief An allocator adaptor to do default initialization instead of value
 * initialization when argument list is empty.
 *
 */
template <typename T, typename A = std::allocator<T>>
struct default_init_allocator : public A {
    template <typename U>
    struct rebind {
        using other = default_init_allocator<
            U,
            typename std::allocator_traits<A>::template rebind_alloc<U>>;
    };

    using A::A;

    // default initialization
    template <typename U>
    void construct(U* ptr) noexcept(
        std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(ptr)) U;
    }

    // delegate to constructor of A
    template <typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        std::allocator_traits<A>::construct(static_cast<A&>(*this), ptr,
                                            std::forward<Args>(args)...);
    }
};

}  // namespace util

}  // namespace intp

#endif
// End content of "util.hpp"

namespace intp {

template <size_t D>
class MeshDimension {
   public:
    using size_type = size_t;
    constexpr static size_type dim = D;
    using index_type = std::array<size_type, dim>;

   private:
    index_type dim_size_;

   public:
    MeshDimension() = default;

    MeshDimension(index_type dim_size) : dim_size_(dim_size) {}

    template <typename... Args,
              typename = typename std::enable_if<sizeof...(Args) == dim>::type>
    MeshDimension(Args... args) : dim_size_{static_cast<size_type>(args)...} {}

    MeshDimension(size_type n) {
        std::fill(dim_size_.begin(), dim_size_.end(), n);
    }

    // properties

    size_type size() const {
        size_type s = 1;
        for (auto&& d : dim_size_) { s *= d; }
        return s;
    }

    size_type dim_size(size_type dim_ind) const { return dim_size_[dim_ind]; }
    size_type& dim_size(size_type dim_ind) { return dim_size_[dim_ind]; }

    size_type dim_acc_size(size_type dim_ind) const {
        size_type das = 1;
        for (size_type d = 0; d < dim_ind; ++d) {
            das *= dim_size_[dim - d - 1];
        }
        return das;
    }

    /**
     * @brief Convert to underlying dimensions array.
     */
    operator index_type() const { return dim_size_; }

    /**
     * @brief Convert multi-dimension index to one dimension index in storage
     * vector.
     *
     * @return size_type
     */
    template <typename... Indices>
    size_type indexing_safe(Indices... indices) const {
        return indexing_safe(index_type{static_cast<size_type>(indices)...});
    }

    size_type indexing_safe(const index_type& ind_arr) const {
        size_type ind{};
        size_type sub_mesh_size = 1;
        for (size_type d = 0; d < dim; ++d) {
            const size_type d_r = dim - d - 1;
            INTP_ASSERT(ind_arr[d_r] >= 0 && ind_arr[d_r] < dim_size_[d_r],
                        std::string("Mesh access out of range at dim ") +
                            std::to_string(d_r) + std::string(", index ") +
                            std::to_string(ind_arr[d_r]) +
                            std::string(" is out of [0, ") +
                            std::to_string(dim_size_[d_r] - 1) +
                            std::string("]."));
            ind += ind_arr[d_r] * sub_mesh_size;
            sub_mesh_size *= dim_size_[d_r];
        }
        return ind;
    }

    size_type indexing(const index_type& ind_arr) const {
        size_type ind{};
        size_type sub_mesh_size = 1;
        for (size_type d = 0; d < dim; ++d) {
            ind += ind_arr[dim - d - 1] * sub_mesh_size;
            sub_mesh_size *= dim_size_[dim - d - 1];
        }
        return ind;
    }

    /**
     * @brief Convert one dimension index in storage vector to multi-dimension
     * indices
     *
     */
    index_type dimwise_indices(size_type total_ind) const {
        index_type indices;

        for (size_type d = 0; d < dim; ++d) {
            indices[dim - d - 1] = total_ind % dim_size_[dim - d - 1];
            total_ind /= dim_size_[dim - d - 1];
        }

        return indices;
    }

    // modifiers

    void resize(index_type sizes) { dim_size_ = sizes; }
};

/**
 * @brief A multi dimension mesh storing data on each mesh point
 *
 * @tparam T Type of data stored
 * @tparam D Dimension
 * @tparam Alloc Allocator type, defaulted to std::allocator<T>
 */
template <typename T, size_t D, typename Alloc = std::allocator<T>>
class Mesh {
   public:
    using size_type = size_t;
    using val_type = T;
    const static size_type dim = D;
    using index_type = typename MeshDimension<dim>::index_type;
    using allocator_type = Alloc;

   private:
    using container_type = std::vector<val_type, allocator_type>;
    using const_iterator = typename container_type::const_iterator;

    template <typename U>
    class skip_iterator {
       public:
        using value_type = U;
        using difference_type = std::ptrdiff_t;
        using pointer = U*;
        using const_pointer = const U*;
        using reference = U&;
        using iterator_category = std::random_access_iterator_tag;

       private:
        pointer ptr_;
        difference_type step_length_;

       public:
        skip_iterator(value_type* ptr, difference_type step_length)
            : ptr_(ptr), step_length_(step_length) {}

        // allow iterator to const_iterator conversion
        operator skip_iterator<const T>() { return {ptr_, step_length_}; }
        // cast to (const) underlying pointer type
        explicit operator const_pointer() const { return ptr_; }

        // forward iterator requirement

        reference operator*() { return *ptr_; }
        reference operator->() { return ptr_; }

        bool operator==(const skip_iterator& other) const {
            return this->ptr_ == other.ptr_ &&
                   this->step_length_ == other.step_length_;
        }
        bool operator!=(const skip_iterator& other) const {
            return !operator==(other);
        }

        skip_iterator& operator++() {
            ptr_ += step_length_;
            return *this;
        }
        skip_iterator operator++(int) {
            skip_iterator tmp(*this);
            operator++();
            return tmp;
        }

        // bidirectional iterator requirement

        skip_iterator& operator--() {
            ptr_ -= step_length_;
            return *this;
        }
        skip_iterator operator--(int) {
            skip_iterator tmp(*this);
            operator--();
            return tmp;
        }

        // random access iterator requirement

        skip_iterator& operator+=(difference_type n) {
            ptr_ += n * step_length_;
            return *this;
        }
        skip_iterator operator+(difference_type n) {
            skip_iterator tmp(*this);
            return tmp += n;
        }
        friend skip_iterator operator+(difference_type n, skip_iterator it) {
            return it += n;
        }

        skip_iterator& operator-=(difference_type n) {
            ptr_ -= n * step_length_;
            return *this;
        }
        skip_iterator operator-(difference_type n) {
            skip_iterator tmp(*this);
            return tmp -= n;
        }

        difference_type operator-(skip_iterator other) {
            return (ptr_ - other.ptr_) / step_length_;
        }

        reference operator[](difference_type n) {
            return *(ptr_ + n * step_length_);
        }

        bool operator<(const skip_iterator& other) const {
            return other - *this > 0;
        }
        bool operator>(const skip_iterator& other) const {
            return other < *this;
        }
        bool operator<=(const skip_iterator& other) const {
            return !(*this > other);
        }
        bool operator>=(const skip_iterator& other) const {
            return !(*this < other);
        }
    };

    /**
     * @brief Stores the mesh content in row-major format.
     */
    container_type storage_;

    MeshDimension<dim> dimension_;

   public:
    explicit Mesh(const MeshDimension<dim>& mesh_dimension,
                  const allocator_type& alloc = allocator_type())
        : storage_(alloc), dimension_(mesh_dimension) {
        storage_.resize(dimension_.size(), val_type{});
    }

    explicit Mesh(size_type n, const allocator_type& alloc = allocator_type())
        : Mesh(MeshDimension<dim>(n), alloc) {}

    template <typename... Args,
              typename = typename std::enable_if<sizeof...(Args) == dim>::type,
              typename = typename std::enable_if<std::is_integral<
                  typename std::common_type<Args...>::type>::value>::type>
    explicit Mesh(Args... args)
        : Mesh(MeshDimension<dim>(static_cast<size_type>(args)...)) {}

    template <typename InputIter,
              typename = typename std::enable_if<
                  dim == 1u &&
                  std::is_convertible<typename std::iterator_traits<
                                          InputIter>::iterator_category,
                                      std::input_iterator_tag>::value>::type>
    explicit Mesh(std::pair<InputIter, InputIter> range,
                  const allocator_type& alloc = allocator_type())
        : storage_(range.first, range.second, alloc),
          dimension_{static_cast<size_type>(storage_.size())} {}

    // convert constructor from mesh using different allocator
    template <typename Allocator>
    Mesh(const Mesh<val_type, dim, Allocator>& mesh,
         const allocator_type& alloc = allocator_type())
        : Mesh(mesh.dimension(), alloc) {
        std::copy(mesh.begin(), mesh.end(), storage_.begin());
    }

    // properties

    size_type size() const { return storage_.size(); }

    size_type dim_size(size_type dim_ind) const {
        return dimension_.dim_size(dim_ind);
    }

    /**
     * @brief Get the underlying mesh dimension object
     *
     */
    const MeshDimension<dim>& dimension() const { return dimension_; }

    // modifiers

    void resize(index_type sizes) {
        dimension_.resize(sizes);
        storage_.resize(dimension_.size());
    }

    // element access

    template <typename... Indices>
    val_type& operator()(Indices... indices) {
        return storage_[dimension_.indexing_safe(indices...)];
    }

    template <typename... Indices>
    const val_type& operator()(Indices... indices) const {
        return storage_[dimension_.indexing_safe(indices...)];
    }

    val_type& operator()(index_type indices) {
        return storage_[dimension_.indexing(indices)];
    }

    const val_type& operator()(index_type indices) const {
        return storage_[dimension_.indexing(indices)];
    }

    const val_type* data() const { return storage_.data(); }

    // iterator

    /**
     * @brief Begin const_iterator to underlying container.
     *
     * @return iterator
     */
    const_iterator begin() const { return storage_.cbegin(); }
    /**
     * @brief End const_iterator to underlying container.
     *
     * @return iterator
     */
    const_iterator end() const { return storage_.cend(); }

    skip_iterator<val_type> begin(size_type dim_ind, index_type indices) {
        indices[dim_ind] = 0;
        return skip_iterator<val_type>(
            storage_.data() + dimension_.indexing_safe(indices),
            static_cast<typename skip_iterator<val_type>::difference_type>(
                dimension_.dim_acc_size(dim - dim_ind - 1)));
    }
    skip_iterator<val_type> end(size_type dim_ind, index_type indices) {
        indices[dim_ind] = dimension_.dim_size(dim_ind);
        return skip_iterator<val_type>(
            storage_.data() + dimension_.indexing(indices),
            static_cast<typename skip_iterator<val_type>::difference_type>(
                dimension_.dim_acc_size(dim - dim_ind - 1)));
    }
    skip_iterator<const val_type> begin(size_type dim_ind,
                                        index_type indices) const {
        indices[dim_ind] = 0;
        return skip_iterator<const val_type>(
            storage_.data() + dimension_.indexing_safe(indices),
            static_cast<typename skip_iterator<val_type>::difference_type>(
                dimension_.dim_acc_size(dim - dim_ind - 1)));
    }
    skip_iterator<const val_type> end(size_type dim_ind,
                                      index_type indices) const {
        indices[dim_ind] = dimension_.dim_size(dim_ind);
        return skip_iterator<const val_type>(
            storage_.data() + indexing(indices),
            dimension_.dim_acc_size(dim - dim_ind - 1));
    }

    index_type iter_indices(const_iterator iter) const {
        return dimension_.dimwise_indices(
            static_cast<size_type>(std::distance(begin(), iter)));
    }

    index_type iter_indices(skip_iterator<const val_type> skip_iter) const {
        return dimension_.dimwise_indices(static_cast<size_type>(
            static_cast<const typename skip_iterator<const val_type>::pointer>(
                skip_iter) -
            data()));
    }
};

}  // namespace intp

#endif
// End content of "Mesh.hpp"
// Begin content of aligned-allocator.hpp
#ifndef INTP_ALIGN_ALLOC
#define INTP_ALIGN_ALLOC

/**
 * @file aligned-allocator.hpp
 * @brief Use the content of this
 * [question](https://stackoverflow.com/questions/12942548/making-stdvector-allocate-aligned-memory)
 * for the allocator class design and this
 * [question](https://stackoverflow.com/questions/16376942/best-cross-platform-method-to-get-aligned-memory)
 * for the actual memory allocation implementation.
 * @date 2022-09-08
 *
 */

#include <cstddef>
#include <cstdlib>

namespace intp {

enum class Alignment : std::size_t {
    NORMAL = sizeof(void*),
    SSE = 16,
    AVX = 32,
    AVX512 = 64,
};

namespace detail {
static inline void* allocate_aligned_memory(std::size_t align,
                                            std::size_t size) {
    void* ptr;
#ifdef _MSC_VER
    ptr = _aligned_malloc(size, align);
#else
    if (posix_memalign(&ptr, align, size)) ptr = nullptr;
#endif
    return ptr;
}
static inline void deallocate_aligned_memory(void* ptr) noexcept {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
}  // namespace detail

template <typename T, Alignment Align = Alignment::AVX>
class AlignedAllocator;

template <Alignment Align>
class AlignedAllocator<void, Align> {
   public:
    using pointer = void*;
    using const_pointer = const void*;
    using value_type = void;

    template <class U>
    struct rebind {
        typedef AlignedAllocator<U, Align> other;
    };
};

template <typename T, Alignment Align>
class AlignedAllocator {
   public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        typedef AlignedAllocator<U, Align> other;
    };

   public:
    AlignedAllocator() noexcept {}

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    size_type max_size() const noexcept {
        return (size_type(~0) - size_type(Align)) / sizeof(T);
    }

    pointer address(reference x) const noexcept { return std::addressof(x); }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    pointer allocate(
        size_type n,
        typename AlignedAllocator<void, Align>::const_pointer = nullptr) {
        const size_type alignment = static_cast<size_type>(Align);
        void* ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) { throw std::bad_alloc(); }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        return detail::deallocate_aligned_memory(p);
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T, Alignment Align>
class AlignedAllocator<const T, Align> {
   public:
    using value_type = T;
    using pointer = const T*;
    using const_pointer = const T*;
    using reference = const T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        typedef AlignedAllocator<U, Align> other;
    };

   public:
    AlignedAllocator() noexcept {}

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    size_type max_size() const noexcept {
        return (size_type(~0) - size_type(Align)) / sizeof(T);
    }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    pointer allocate(
        size_type n,
        typename AlignedAllocator<void, Align>::const_pointer = 0) {
        const size_type alignment = static_cast<size_type>(Align);
        void* ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) { throw std::bad_alloc(); }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        return detail::deallocate_aligned_memory(p);
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T, Alignment TAlign, typename U, Alignment UAlign>
inline bool operator==(const AlignedAllocator<T, TAlign>&,
                       const AlignedAllocator<U, UAlign>&) noexcept {
    return TAlign == UAlign;
}

template <typename T, Alignment TAlign, typename U, Alignment UAlign>
inline bool operator!=(const AlignedAllocator<T, TAlign>&,
                       const AlignedAllocator<U, UAlign>&) noexcept {
    return TAlign != UAlign;
}

}  // namespace intp

#endif  // INTP_ALIGN_ALLOC
// End content of "aligned-allocator.hpp"

namespace intp {

/**
 * @brief B-Spline function
 *
 * @tparam T Type of control point
 * @tparam D Dimension
 */
template <typename T, std::size_t D, std::size_t O, typename U = double>
class BSpline {
   public:
    using spline_type = BSpline<T, D, O, U>;

    using size_type = std::size_t;
    using val_type = T;
    using knot_type = U;
    constexpr static size_type dim = D;
    constexpr static size_type order = O;

    using KnotContainer = std::vector<knot_type>;

    using ControlPointContainer =
        Mesh<val_type,
             dim,
             util::default_init_allocator<
                 val_type,
                 AlignedAllocator<val_type, Alignment::AVX>>>;
#ifdef INTP_CELL_LAYOUT
    using ControlPointCellContainer =
        Mesh<val_type,
             dim + 1,
             util::default_init_allocator<
                 val_type,
                 AlignedAllocator<val_type, Alignment::AVX>>>;
    using control_point_type = ControlPointCellContainer;
#else
    using control_point_type = ControlPointContainer;
#endif

    using BaseSpline = std::array<knot_type, order + 1>;
    using diff_type = typename KnotContainer::iterator::difference_type;
    using knot_const_iterator = typename KnotContainer::const_iterator;

    // Container for dimension-wise storage
    template <typename T_>
    using DimArray = std::array<T_, dim>;

    /**
     * @brief Calculate values on base spline function. This is the core of
     * B-Spline. Note: when the given order is smaller than order of spline
     * (used in calculating derivative), spline value is aligned at right in
     * result vector.
     *
     * @param seg_idx_iter the iterator points to left knot point of a segment
     * @param x coordinate
     * @param spline_order order of base spline, defaulted to be spline function
     * order
     * @return a reference to local buffer
     */
    inline const BaseSpline base_spline_value(
        size_type,
        knot_const_iterator seg_idx_iter,
        knot_type x,
        size_type spline_order = order) const {
        BaseSpline base_spline{};
        base_spline[order] = 1;

        for (size_type i = 1; i <= spline_order; ++i) {
            // Each iteration will expand buffer zone by one, from back
            // to front.
            const size_type idx_begin = order - i;
            for (size_type j = 0; j <= i; ++j) {
                const auto left_iter =
                    seg_idx_iter - static_cast<diff_type>(i - j);
                const auto right_iter =
                    seg_idx_iter + static_cast<diff_type>(j + 1);
                base_spline[idx_begin + j] =
                    (j == 0 ? 0
                            : base_spline[idx_begin + j] * (x - *left_iter) /
                                  (*(right_iter - 1) - *left_iter)) +
                    (idx_begin + j == order
                         ? 0
                         : base_spline[idx_begin + j + 1] * (*right_iter - x) /
                               (*right_iter - *(left_iter + 1)));
            }
        }
        return base_spline;
    }

    /**
     * @brief Get the lower knot iter points to the segment where given x
     * locates. If x is out of range of knot vector, the iterator is rather
     * begin or end of knot vector.
     *
     * @param dim_ind specify the dimension
     * @param x coordinate
     * @param hint a hint for iter offset
     * @param last an upper bound for iter offset, this function will not search
     * knots beyond it.
     * @return knot_const_iterator
     */
    inline knot_const_iterator get_knot_iter(size_type dim_ind,
                                             knot_type& x,
                                             size_type hint,
                                             size_type last) const {
        const auto iter = knots_begin(dim_ind) + static_cast<diff_type>(hint);
        if (periodicity_[dim_ind]) {
            const knot_type period =
                range(dim_ind).second - range(dim_ind).first;
            x = range(dim_ind).first +
                std::fmod(x - range(dim_ind).first, period) +
                (x < range(dim_ind).first ? period : knot_type{});
        }
#ifdef INTP_TRACE
        if ((*iter > x || *(iter + 1) < x) && x >= range(dim_ind).first &&
            x <= range(dim_ind).second) {
            std::cout << "[TRACE] knot hint miss at dim = " << dim_ind
                      << ", hint = " << hint << ", x = " << x << '\n';
        }
#endif
        // I tried return the iter without checking, but the speed has no
        // significant improves.
        return *iter <= x && *(iter + 1) > x
                   // If the hint is accurate, use that iter
                   ? iter
                   // else, use binary search in the range of distinct knots
                   // (excluding beginning and ending knots that have same
                   // value)
                   : --(std::upper_bound(knots_begin(dim_ind) +
                                             static_cast<diff_type>(order + 1),
                                         knots_begin(dim_ind) +
                                             static_cast<diff_type>(last + 1),
                                         x));
    }

    inline knot_const_iterator get_knot_iter(size_type dim_ind,
                                             knot_type& x,
                                             size_type hint) const {
        return get_knot_iter(dim_ind, x, hint, knots_num(dim_ind) - order - 2);
    }

    template <typename C, size_type... indices>
    inline DimArray<knot_const_iterator> get_knot_iters(
        util::index_sequence<indices...>,
        C&& coords) const {
        return {get_knot_iter(indices, std::get<0>(coords[indices]),
                              std::get<1>(coords[indices]))...};
    }

    /**
     * @brief Construct a new BSpline object, with periodicity of each dimension
     * specified.
     *
     */
    explicit BSpline(DimArray<bool> periodicity)
        : periodicity_(periodicity), control_points_(size_type{}) {}

    /**
     * @brief Basically the default constructor, initialize an empty, non-closed
     * B-Spline
     *
     */
    explicit BSpline() : BSpline(DimArray<bool>{}) {}

    template <typename... InputIters>
    BSpline(DimArray<bool> periodicity,
            ControlPointContainer ctrl_pts,
            std::pair<InputIters, InputIters>... knot_iter_pairs)
        : periodicity_(periodicity),
          knots_{
              KnotContainer(knot_iter_pairs.first, knot_iter_pairs.second)...},
#ifdef INTP_CELL_LAYOUT
          control_points_(generate_cell_layout(ctrl_pts)),
#else
          control_points_(std::move(ctrl_pts)),
#endif
          range_{std::make_pair(
              (knot_iter_pairs.first)[order],
              (knot_iter_pairs.second)[-static_cast<int>(order) - 1])...} {
        for (size_type d = 0; d < dim; ++d) {
            INTP_ASSERT(knots_[d].size() - ctrl_pts.dim_size(d) ==
                            (periodicity_[d] ? 2 * order + 1 : order + 1),
                        std::string("Inconsistency between knot number and "
                                    "control point number at dimension ") +
                            std::to_string(d));
        }
    }

    template <typename... InputIters>
    BSpline(ControlPointContainer ctrl_points,
            std::pair<InputIters, InputIters>... knot_iter_pairs)
        : BSpline(DimArray<bool>{}, ctrl_points, knot_iter_pairs...) {}

    template <typename C>
    typename std::enable_if<
        std::is_same<typename std::remove_reference<C>::type,
                     KnotContainer>::value,
        void>::type
    load_knots(size_type dim_ind, C&& _knots) {
        knots_[dim_ind] = std::forward<C>(_knots);
        range_[dim_ind].first = knots_[dim_ind][order];
        range_[dim_ind].second =
            knots_[dim_ind][knots_[dim_ind].size() - order - (2 - order % 2)];
    }

#ifdef INTP_CELL_LAYOUT
    void load_ctrlPts(const ControlPointContainer& control_points) {
        control_points_ = generate_cell_layout(control_points);
    }
#else
    template <typename C>
    typename std::enable_if<
        std::is_same<typename std::remove_reference<C>::type,
                     ControlPointContainer>::value,
        void>::type
    load_ctrlPts(C&& control_points) {
        control_points_ = std::forward<C>(control_points);
    }
#endif

#ifdef INTP_CELL_LAYOUT
#if __cplusplus >= 201402L
    auto
#else
    std::function<val_type(const spline_type&)>
#endif
    pre_calc_coef(
        DimArray<std::pair<knot_type, size_type>> coord_with_hints) const {
        using Indices = util::make_index_sequence<dim>;
        // get knot point iter, it will modifies coordinate value into
        // interpolation range of periodic dimension.
        const auto knot_iters = get_knot_iters(Indices{}, coord_with_hints);

        DimArray<size_type> spline_order;
        spline_order.fill(order);
        // calculate basic spline (out of boundary check also conducted here)
        const auto base_spline_values_1d = calc_base_spline_vals(
            Indices{}, knot_iters, spline_order, coord_with_hints);

        std::array<size_type, dim + 1> ind_arr{};
        for (size_type d = 0; d < dim; ++d) {
            ind_arr[d] = static_cast<size_type>(
                             distance(knots_begin(d), knot_iters[d])) -
                         order;
        }

        auto total_offset = calculate_cell_dim_from_knots().indexing(ind_arr);

        return
            [base_spline_values_1d, total_offset](const spline_type& spline) {
                val_type v{};

                const auto& control_points = spline.control_points();
                auto cell_iter = control_points.begin() +
                                 static_cast<std::ptrdiff_t>(total_offset);
                for (size_type i = 0;
                     i < control_points.dim_size(dim) * (order + 1); ++i) {
                    knot_type coef = 1;
                    if CPP17_CONSTEXPR_ (dim == 1) {
                        // helps with vectorization in 1D case
                        coef = base_spline_values_1d[0][i];
                    } else {
                        for (size_type d = 0, combined_ind = i; d < dim; ++d) {
                            coef *= base_spline_values_1d[d][combined_ind %
                                                             (order + 1)];
                            combined_ind /= (order + 1);
                        }
                    }
                    v += coef * (*cell_iter++);
                }
                return v;
            };
    }
#endif

    /**
     * @brief Get spline value at given pairs of coordinate and position hint
     * (hopefully lower knot point index of the segment where coordinate
     * locates, dimension wise).
     *
     */
    val_type operator()(
        DimArray<std::pair<knot_type, size_type>> coord_with_hints) const {
        using Indices = util::make_index_sequence<dim>;
        // get knot point iter, it will modifies coordinate value into
        // interpolation range of periodic dimension.
        const auto knot_iters = get_knot_iters(Indices{}, coord_with_hints);

        DimArray<size_type> spline_order;
        spline_order.fill(order);
        // calculate basic spline (out of boundary check also conducted here)
        const auto base_spline_values_1d = calc_base_spline_vals(
            Indices{}, knot_iters, spline_order, coord_with_hints);

        // combine control points and basic spline values to get spline value
        val_type v{};
#ifdef INTP_CELL_LAYOUT
        std::array<size_type, dim + 1> ind_arr{};
        for (size_type d = 0; d < dim; ++d) {
            ind_arr[d] = static_cast<size_type>(
                             distance(knots_begin(d), knot_iters[d])) -
                         order;
        }
        auto cell_iter = control_points_.begin(dim, ind_arr);
        for (size_type i = 0; i < buf_size_; ++i) {
            auto coef = *cell_iter++;
            for (size_type d = 0, combined_ind = i; d < dim; ++d) {
                coef *= base_spline_values_1d[d][combined_ind % (order + 1)];
                combined_ind /= (order + 1);
            }
            v += coef;
        }
#else
        MeshDimension<dim> local_mesh_dim(order + 1);
        for (size_type i = 0; i < buf_size_; ++i) {
            DimArray<size_type> ind_arr = local_mesh_dim.dimwise_indices(i);

            knot_type coef = 1;
            for (size_type d = 0; d < dim; ++d) {
                coef *= base_spline_values_1d[d][ind_arr[d]];

                // Shift index array according to knot iter of each dimension.
                // When the coordinate is out of range in some dimensions, the
                // corresponding iterator was set to be begin or end iterator of
                // knot vector in `get_knot_iters` method and it will be treated
                // separately.
                ind_arr[d] += knot_iters[d] == knots_begin(d) ? 0
                              : knot_iters[d] == knots_end(d)
                                  ? control_points_.dim_size(d) - order - 1
                                  : static_cast<size_type>(distance(
                                        knots_begin(d), knot_iters[d])) -
                                        order;

                // check periodicity, put out-of-right-boundary index to left
                if (periodicity_[d]) {
                    ind_arr[d] %= control_points_.dim_size(d);
                }
            }

            v += coef * control_points_(ind_arr);
        }
#endif

        return v;
    }

    /**
     * @brief Get spline value at given coordinates
     *
     * @param coords a bunch of cartesian coordinates
     * @return val_type
     */
    val_type operator()(DimArray<double> coords) const {
        DimArray<std::pair<knot_type, size_type>> coord_with_hints;
        for (std::size_t d = 0; d < dim; ++d) {
            coord_with_hints[d] = {coords[d], order};
        }
        return operator()(coord_with_hints);
    }

    /**
     * @brief Get derivative value at given pairs of coordinate and position
     * hint (possibly lower knot point index of the segment where coordinate
     * locates, dimension wise).
     *
     * @param coord_deriOrder_hint_tuple a bunch of (coordinate, position hint,
     * derivative order) tuple
     * @return val_type
     */
    val_type derivative_at(DimArray<std::tuple<knot_type, size_type, size_type>>
                               coord_deriOrder_hint_tuple) const {
        // get spline order
        DimArray<size_type> spline_order;
        for (size_type d = 0; d < dim; ++d) {
            spline_order[d] =
                order >= std::get<2>(coord_deriOrder_hint_tuple[d])
                    ? order - std::get<2>(coord_deriOrder_hint_tuple[d])
                    : order + 1;
        }

        // if derivative order is larger than spline order, derivative is 0.
        for (auto o : spline_order) {
            if (o > order) { return val_type{}; }
        }

        using Indices = util::make_index_sequence<dim>;
        // get knot point iter (out of boundary check also conducted here)
        const auto knot_iters =
            get_knot_iters(Indices{}, coord_deriOrder_hint_tuple);

        // calculate basic spline
        const auto base_spline_values_1d = calc_base_spline_vals(
            Indices{}, knot_iters, spline_order, coord_deriOrder_hint_tuple);

#ifdef STACK_ALLOCATOR
        // create local buffer
        val_type buffer[MAX_BUF_SIZE_];
        util::stack_allocator<val_type, MAX_BUF_SIZE_> alloc(buffer);

        Mesh<val_type, dim, util::stack_allocator<val_type, MAX_BUF_SIZE_>>
            local_control_points(order + 1, alloc);
        auto local_spline_val = local_control_points;
#else
        Mesh<val_type, dim> local_control_points(order + 1);
        auto local_spline_val = local_control_points;
#endif

        // get local control points and basic spline values

#ifdef INTP_CELL_LAYOUT
        std::array<size_type, dim + 1> ind_arr{};
        for (size_type d = 0; d < dim; ++d) {
            ind_arr[d] = static_cast<size_type>(
                             distance(knots_begin(d), knot_iters[d])) -
                         order;
        }
        auto cell_iter = control_points_.begin(dim, ind_arr);
        for (size_type i = 0; i < buf_size_; ++i) {
            DimArray<size_type> local_ind_arr{};
            for (size_type d = 0, combined_ind = i; d < dim; ++d) {
                local_ind_arr[d] = combined_ind % (order + 1);
                combined_ind /= (order + 1);
            }

            knot_type coef = 1;
            for (size_type d = 0; d < dim; ++d) {
                coef *= base_spline_values_1d[d][local_ind_arr[d]];
            }

            local_spline_val(local_ind_arr) = coef;
            local_control_points(local_ind_arr) = *cell_iter++;
        }
#else
        for (size_type i = 0; i < buf_size_; ++i) {
            DimArray<size_type> local_ind_arr{};
            for (size_type d = 0, combined_ind = i; d < dim; ++d) {
                local_ind_arr[d] = combined_ind % (order + 1);
                combined_ind /= (order + 1);
            }

            knot_type coef = 1;
            DimArray<size_type> ind_arr{};
            for (size_type d = 0; d < dim; ++d) {
                coef *= base_spline_values_1d[d][local_ind_arr[d]];

                ind_arr[d] = local_ind_arr[d] +
                             (knot_iters[d] == knots_begin(d) ? 0
                              : knot_iters[d] == knots_end(d)
                                  ? control_points_.dim_size(d) - order - 1
                                  : static_cast<size_t>(distance(
                                        knots_begin(d), knot_iters[d])) -
                                        order);

                // check periodicity, put out-of-right-boundary index to
                // left
                if (periodicity_[d]) {
                    ind_arr[d] %= control_points_.dim_size(d);
                }
            }

            local_spline_val(local_ind_arr) = coef;
            local_control_points(local_ind_arr) = control_points_(ind_arr);
        }
#endif

        for (size_type d = 0; d < dim; ++d) {
            if (spline_order[d] == order) { continue; }
            // calculate control points for derivative along this dimension

            const size_type hyper_surface_size =
                local_control_points.size() / local_control_points.dim_size(d);
            // transverse the hyper surface of fixing dimension d
            for (size_type i = 0; i < hyper_surface_size; ++i) {
                DimArray<size_type> local_ind_arr{};
                for (size_type dd = 0, combined_ind = i; dd < dim; ++dd) {
                    if (dd == d) { continue; }
                    local_ind_arr[dd] = combined_ind % (order + 1);
                    combined_ind /= (order + 1);
                }

                auto iter = local_control_points.begin(d, local_ind_arr);
                // Taking derivative is effectively computing new control
                // points. Number of iteration is order of derivative.
                for (diff_type k = static_cast<diff_type>(order);
                     k > static_cast<diff_type>(spline_order[d]); --k) {
                    // Each reduction reduce control points number by one.
                    // Reduce backward to match pattern of local_spline_val.
                    for (diff_type j = k; j > 0; --j) {
                        iter[static_cast<diff_type>(order) + j - k] =
                            static_cast<val_type>(k) *
                            (iter[static_cast<diff_type>(order) + j - k] -
                             iter[static_cast<diff_type>(order) + j - k - 1]) /
                            (knot_iters[d][j] - knot_iters[d][j - k]);
                    }
                }
            }
        }

        // combine spline value and control points to get spline derivative
        // value
        val_type v{};
        for (auto s_it = local_spline_val.begin(),
                  c_it = local_control_points.begin();
             s_it != local_spline_val.end(); ++s_it, ++c_it) {
            v += (*s_it) * (*c_it);
        }

        return v;
    }

    /**
     * @brief Get derivative value at given coordinates
     *
     * @param coord_deriOrders a bunch of (coordinate, derivative order) tuple
     * @return val_type
     */
    val_type derivative_at(
        DimArray<std::pair<knot_type, size_type>> coord_deriOrders) const {
        DimArray<std::tuple<knot_type, size_type, size_type>>
            coord_deriOrder_hint_tuple;
        for (std::size_t d = 0; d < dim; ++d) {
            coord_deriOrder_hint_tuple[d] = {std::get<0>(coord_deriOrders[d]),
                                             order,
                                             std::get<1>(coord_deriOrders[d])};
        }
        return derivative_at(coord_deriOrder_hint_tuple);
    }

    // iterators

    /**
     * @brief Returns a read-only (constant) iterator that points to the
     *  first element in the knot vector of one dimension.
     *
     * @param dim_ind dimension index
     */
    inline knot_const_iterator knots_begin(size_type dim_ind) const {
        return knots_[dim_ind].cbegin();
    }
    /**
     * @brief Returns a read-only (constant) iterator that points to the
     *  first element in the knot vector of one dimension.
     *
     * @param dim_ind dimension index
     */
    inline knot_const_iterator knots_end(size_type dim_ind) const {
        return knots_[dim_ind].cend();
    }

    // properties

    inline const control_point_type& control_points() const {
        return control_points_;
    }

    /**
     * @brief Get range of one dimension
     *
     */
    inline const std::pair<knot_type, knot_type>& range(
        size_type dim_ind) const {
        return range_[dim_ind];
    }

    /**
     * @brief Get knot number of one dimension
     *
     */
    inline size_type knots_num(size_type dim_ind) const {
        return knots_[dim_ind].size();
    }

    /**
     * @brief Get periodicity of one dimension
     *
     * @param dim_ind dimension index
     * @return a bool
     */
    inline bool periodicity(size_type dim_ind) const {
        return periodicity_[dim_ind];
    }

    inline constexpr size_type get_order() const { return order; }

#ifdef INTP_DEBUG
    void debug_output() const {
        std::cout << "\n[DEBUG] Control Points (raw data):\n";

        // 17 digits for double precision
        std::cout.precision(17);
        size_type idx = 1;
        for (auto v : control_points_) {
            if (idx % control_points_.dim_size(dim - 1) == 1) {
                std::cout << "[DEBUG] ";
            }
            std::cout << v << ' ';
            if (idx++ % control_points_.dim_size(dim - 1) == 0) {
                std::cout << '\n';
            }
        }
        std::cout << '\n';
    }
#endif

   private:
    DimArray<bool> periodicity_;

    DimArray<KnotContainer> knots_;
#ifdef INTP_CELL_LAYOUT
    ControlPointCellContainer control_points_;
#else
    ControlPointContainer control_points_;
#endif

    DimArray<std::pair<knot_type, knot_type>> range_;

    constexpr static size_type buf_size_ = util::pow(order + 1, dim);

    // maximum stack buffer size
    // This buffer is for storing weights when calculating spline derivative
    // value.
    constexpr static size_type MAX_BUF_SIZE_ = 1000;

    // auxiliary methods

    /**
     * @brief Calculate base spline value of each dimension
     *
     */
    template <typename C, size_type... indices>
    inline DimArray<BaseSpline> calc_base_spline_vals(
        util::index_sequence<indices...>,
        const DimArray<knot_const_iterator>& knot_iters,
        const DimArray<size_type>& spline_order,
        const C& coords) const {
        return {base_spline_value(indices, knot_iters[indices],
                                  std::get<0>(coords[indices]),
                                  spline_order[indices])...};
    }

#ifdef INTP_CELL_LAYOUT
    MeshDimension<dim + 1> calculate_cell_dim_from_knots() const {
        MeshDimension<dim + 1> cell_dim(util::pow(order + 1, dim - 1));
        for (size_type d = 0; d < dim; ++d) {
            cell_dim.dim_size(d) = knots_[d].size() -
                                   (periodicity(d) ? order + 1 : order + 1) -
                                   (d == dim - 1 ? 0 : order);
        }
        return cell_dim;
    }

    ControlPointCellContainer generate_cell_layout(
        const ControlPointContainer& ctrl_pts) const {
        MeshDimension<dim + 1> cell_container_dim(
            util::pow(order + 1, dim - 1));
        for (size_type d = 0; d < dim; ++d) {
            cell_container_dim.dim_size(d) = ctrl_pts.dim_size(d) -
                                             (d == dim - 1 ? 0 : order) +
                                             (periodicity(d) ? order : 0);
        }

        // Size of the last two dimension of control point cell container. The
        // (dim-1)th dimention of cell container has the same length (order
        // points more in periodic case) as the last dimension (which is also
        // the (dim-1)th dimension) of the origin container, while other
        // dimensions are #order shorter, except the last dimension with length
        // (order+1)^(dim-1).
        const auto line_size = cell_container_dim.dim_size(dim - 1) *
                               cell_container_dim.dim_size(dim);

        ControlPointCellContainer control_point_cell(cell_container_dim);
        // size of hyperplane orthogonal to last 2 dimension
        const size_type hyper_surface_size =
            control_point_cell.size() / line_size;

        auto fill_cell = [&](size_type begin, size_type end) {
            // iterate over hyperplane
            for (size_type h_ind = begin; h_ind < end; ++h_ind) {
                const auto ind_arr_on_hyper_surface =
                    control_point_cell.dimension().dimwise_indices(h_ind *
                                                                   line_size);
                const auto line_begin =
                    control_point_cell.begin(dim - 1, ind_arr_on_hyper_surface);
                const auto line_end =
                    control_point_cell.end(dim - 1, ind_arr_on_hyper_surface);
                // iterate along the (dim-1)th dimension
                for (auto iter = line_begin; iter != line_end; ++iter) {
                    auto cell_ind_arr = control_point_cell.iter_indices(iter);
                    MeshDimension<dim - 1> cell_dim(order + 1);
                    // iterate the (dim)th dimension
                    for (size_type i = 0; i < cell_dim.size(); ++i) {
                        auto local_shift_ind = cell_dim.dimwise_indices(i);
                        cell_ind_arr[dim] = i;
                        auto cp_ind_arr =
                            typename ControlPointContainer::index_type{};
                        for (size_type d = 0; d < dim; ++d) {
                            cp_ind_arr[d] =
                                (cell_ind_arr[d] +
                                 (d == dim - 1
                                      ? 0
                                      : local_shift_ind[dim - 2 - d])) %
                                ctrl_pts.dim_size(d);
                        }
                        control_point_cell(cell_ind_arr) = ctrl_pts(cp_ind_arr);
                    }
                }
            }
        };
#ifdef INTP_MULTITHREAD
        const size_type block_num = static_cast<size_type>(
            std::sqrt(static_cast<double>(hyper_surface_size)));
        const size_type task_per_block = block_num == 0
                                             ? hyper_surface_size
                                             : hyper_surface_size / block_num;

        auto& thread_pool = DedicatedThreadPool<void>::get_instance(8);
        std::vector<std::future<void>> res;

        for (size_type i = 0; i < block_num; ++i) {
            res.push_back(thread_pool.queue_task([=]() {
                fill_cell(i * task_per_block, (i + 1) * task_per_block);
            }));
        }
        // main thread deals with the remaining part in case hyper_surface_size
        // not divisible by thread_num
        fill_cell(block_num * task_per_block, hyper_surface_size);
        // wait for all tasks are complete
        for (auto&& f : res) { f.get(); }
#else
        fill_cell(0, hyper_surface_size);
#endif  // INTP_MULTITHREAD
        return control_point_cell;
    }
#endif  // INTP_CELL_LAYOUT
};

}  // namespace intp

#endif
// End content of "BSpline.hpp"
// Begin content of BandLU.hpp
#ifndef INTP_BANDLU
#define INTP_BANDLU

#include <type_traits>

// Begin content of BandMatrix.hpp
#ifndef INTP_BANDMATRIX
#define INTP_BANDMATRIX

#include <iostream>
#include <type_traits>
#include <vector>

namespace intp {

/**
 * @brief Band square matrix A is stored in n by (1+p+q) matrix B, with A_{i,j}=
 * B_{j,i+q-j}, i.e. each diagonal is stored as a column in B, aligned by column
 * index in A.
 *
 * @tparam T value type of matrix element
 */
template <typename T, typename Alloc = std::allocator<T>>
class BandMatrix {
   public:
    using size_type = size_t;
    using val_type = T;
    using allocator_type = Alloc;
    using matrix_type = BandMatrix<val_type, allocator_type>;

    // Create a zero band matrix with given dimension, lower and upper
    // bandwidth.
    BandMatrix(size_type n, size_type p, size_type q)
        : n_(n), p_(p), q_(q), bands_{n_, 1 + p_ + q_} {}

    BandMatrix() : BandMatrix(0, 0, 0) {}

    // properties

    size_type dim() const noexcept { return n_; }

    size_type lower_band_width() const noexcept { return p_; }

    size_type upper_band_width() const noexcept { return q_; }

    /**
     * @brief Return read/write reference to matrix element, indices are
     * zero-based.
     *
     * @param i row index
     * @param j column index
     * @return val_type&
     */
    val_type& operator()(size_type i, size_type j) {
        INTP_ASSERT(j + p_ >= i && i + q_ >= j,
                    "Given i and j not in main bands.");
        return bands_(j, i + q_ - j);
    }

    val_type operator()(size_type i, size_type j) const {
        INTP_ASSERT(j + p_ >= i && i + q_ >= j,
                    "Given i and j not in main bands.");
        return bands_(j, i + q_ - j);
    }

    /**
     * @brief Matrix-Vector multiplication
     *
     * @param x vector to be multiplied
     */
    template <typename Vec>
    util::remove_cvref_t<Vec> operator*(const Vec& x) const {
        util::remove_cvref_t<Vec> xx(x.size());
        for (size_type i = 0; i < x.size(); ++i) {
            for (size_type j = p_ > i ? p_ - i : 0, k = i > p_ ? i - p_ : 0;
                 j < std::min(p_ + q_ + 1, n_ + p_ - i); ++j, ++k) {
                xx[i] += bands_(i, j) * x[k];
            }
        }
        return xx;
    }

    /**
     * @brief Insertion operator, used for debug mostly.
     *
     */
    friend std::ostream& operator<<(std::ostream& os, const BandMatrix& mat) {
        for (size_t i = 0; i < mat.n_; ++i) {
            for (size_t j = i > mat.p_ ? i - mat.p_ : 0;
                 j<i + mat.q_ + 1> mat.n_ ? mat.n_ : i + mat.q_ + 1; ++j) {
                os << "{" << i << ", " << j << "}->" << mat(i, j) << '\n';
            }
        }
        return os;
    }

   protected:
    size_type n_;
    size_type p_, q_;
    Mesh<val_type, 2, allocator_type> bands_;
};

template <typename T, typename Alloc = std::allocator<T>>
class ExtendedBandMatrix : public BandMatrix<T, Alloc> {
   public:
    using base_type = BandMatrix<T, Alloc>;
    using size_type = typename base_type::size_type;
    using val_type = typename base_type::val_type;
    using allocator_type = typename base_type::allocator_type;

    ExtendedBandMatrix(size_type dim, size_type lower, size_type upper)
        : base_type(dim, lower, upper),
          right_side_bands_{dim - upper - 1, lower},
          bottom_side_bands_{dim - lower - 1, upper} {}

    ExtendedBandMatrix() : ExtendedBandMatrix(1, 0, 0) {}

    val_type& main_bands_val(size_type i, size_type j) {
        return base_type::operator()(i, j);
    }

    val_type main_bands_val(size_type i, size_type j) const {
        return base_type::operator()(i, j);
    }

    val_type& side_bands_val(size_type i, size_type j) {
        INTP_ASSERT(j >= std::max(n_ - p_, i + q_ + 1) ||
                        i >= std::max(n_ - q_, j + p_ + 1),
                    "Given i and j not in side bands.");
        return j > i + q_ ? right_side_bands_(i, j + p_ - n_)
                          : bottom_side_bands_(j, i + q_ - n_);
    }

    val_type side_bands_val(size_type i, size_type j) const {
        INTP_ASSERT(j >= std::max(n_ - p_, i + q_ + 1) ||
                        i >= std::max(n_ - q_, j + p_ + 1),
                    "Given i and j not in side bands.");
        return j > i + q_ ? right_side_bands_(i, j + p_ - n_)
                          : bottom_side_bands_(j, i + q_ - n_);
    }

    val_type& operator()(size_type i, size_type j) {
        return j > i + q_ || i > j + p_ ? side_bands_val(i, j)
                                        : main_bands_val(i, j);
    }

    val_type operator()(size_type i, size_type j) const {
        return j > i + q_ || i > j + p_ ? side_bands_val(i, j)
                                        : main_bands_val(i, j);
    }

    template <typename Iter>
    util::remove_cvref_t<Iter> operator*(const Iter& x) const {
        util::remove_cvref_t<Iter> xx(x.size());
        for (size_type i = 0; i < x.size(); ++i) {
            for (size_type j = i > p_ ? i - p_ : 0;
                 j < std::min(i + q_ + 1, n_); ++j) {
                xx[i] += main_bands_val(i, j) * x[j];
            }

            // right side bands
            if (i < n_ - q_ - 1) {
                for (size_type j = std::max(n_ - p_, i + q_ + 1); j < n_; ++j) {
                    xx[i] += side_bands_val(i, j) * x[j];
                }
            }
        }

        // bottom side bands
        for (size_type j = 0; j < n_ - p_ - 1; ++j) {
            for (size_type i = std::max(n_ - q_, j + p_ + 1); i < n_; ++i) {
                xx[i] += side_bands_val(i, j) * x[j];
            }
        }
        return xx;
    }

   private:
    Mesh<val_type, 2, allocator_type> right_side_bands_;
    Mesh<val_type, 2, allocator_type> bottom_side_bands_;

    using base_type::n_;
    using base_type::p_;
    using base_type::q_;
};

}  // namespace intp

#endif
// End content of "BandMatrix.hpp"

namespace intp {

/**
 * @brief Base class of band matrix LU solver. The actual solver should inherits
 * from it and implement corresponding impl methods, per the CRTP.
 */
template <template <typename> class Solver, typename Matrix>
class BandLUBase : public util::CRTP<Solver<Matrix>> {
   public:
    using matrix_type = Matrix;

    BandLUBase() noexcept : is_computed_(false) {}

    template <typename Mat_>
    BandLUBase(Mat_&& mat) : lu_store_(std::forward<Mat_>(mat)) {
        static_assert(
            std::is_same<util::remove_cvref_t<Mat_>, matrix_type>::value,
            "Matrix type mismatch");
        this->cast().compute_impl();
        is_computed_ = true;
    }

    template <typename Mat_>
    void compute(Mat_&& mat) {
        static_assert(
            std::is_same<util::remove_cvref_t<Mat_>, matrix_type>::value,
            "Matrix type mismatch");
        if (!is_computed_) {
            lu_store_ = std::forward<matrix_type>(mat);
            this->cast().compute_impl();
            is_computed_ = true;
        }
    }

    template <typename Vec>
    util::remove_cvref_t<Vec> solve(Vec&& vec) const {
        util::remove_cvref_t<Vec> vec_tmp(std::forward<Vec>(vec));
        solve_in_place(vec_tmp);
        return vec_tmp;  // Thanks to NRVO, no copy/move will be performed here.
        // But `return std::move(solve_in_place(vec_tmp));` will do extra moves.
    }

    template <typename Iter>
    void solve_in_place(Iter&& iter) const {
        this->cast().solve_in_place_impl(iter);
    }

   protected:
    bool is_computed_;
    matrix_type lu_store_;

    // Get the type of parameter in array subscript operator of given type U. It
    // is assumed that type U is either a container or an iterator, or a raw
    // pointer.
    template <typename U>
    struct ind_type_ {
       private:
        template <typename V>
        using get_size_type = typename V::size_type;
        template <typename V>
        using get_iter_difference_type = typename V::difference_type;
        template <typename V>
        using get_ptr_difference_type = std::ptrdiff_t;
        template <typename V>
        using get_difference_type =
            typename util::lazy_conditional<std::is_pointer<V>::value,
                                            get_ptr_difference_type,
                                            get_iter_difference_type,
                                            V>::type;

       public:
        using type =
            typename util::lazy_conditional<util::is_iterable<U>::value,
                                            get_size_type,
                                            get_difference_type,
                                            U>::type;
    };
};

template <typename>
class BandLU;

template <typename... Ts>
class BandLU<BandMatrix<Ts...>> : public BandLUBase<BandLU, BandMatrix<Ts...>> {
   public:
    using base_type = BandLUBase<intp::BandLU, BandMatrix<Ts...>>;
    using matrix_type = typename base_type::matrix_type;
    using size_type = typename matrix_type::size_type;

   private:
    friend base_type;
    using base_type::base_type;
    using base_type::lu_store_;

    void compute_impl() {
        const size_type n = lu_store_.dim();
        const size_type p = lu_store_.lower_band_width();
        const size_type q = lu_store_.upper_band_width();

        for (size_type k = 0; k < n - 1; ++k) {
            for (size_type i = k + 1; i < std::min(k + p + 1, n); ++i) {
                lu_store_(i, k) /= lu_store_(k, k);
            }
            for (size_type j = k + 1; j < std::min(k + q + 1, n); ++j) {
                for (size_type i = k + 1; i < std::min(k + p + 1, n); ++i) {
                    lu_store_(i, j) -= lu_store_(i, k) * lu_store_(k, j);
                }
            }
        }
    }

    template <typename Iter>
    void solve_in_place_impl(Iter& iter) const {
        const size_type n = lu_store_.dim();
        const size_type p = lu_store_.lower_band_width();
        const size_type q = lu_store_.upper_band_width();

        using ind_type = typename base_type::template ind_type_<
            util::remove_cvref_t<Iter>>::type;
        // applying l matrix
        for (size_type j = 0; j < n; ++j) {
            for (size_type i = j + 1; i < std::min(j + p + 1, n); ++i) {
                iter[static_cast<ind_type>(i)] -=
                    lu_store_(i, j) * iter[static_cast<ind_type>(j)];
            }
        }
        // applying u matrix
        for (size_type j = n - 1; j < n; --j) {
            iter[static_cast<ind_type>(j)] /= lu_store_(j, j);
            for (size_type i = j < q ? 0 : j - q; i < j; ++i) {
                iter[static_cast<ind_type>(i)] -=
                    lu_store_(i, j) * iter[static_cast<ind_type>(j)];
            }
        }
    }
};

template <typename... Ts>
class BandLU<ExtendedBandMatrix<Ts...>>
    : public BandLUBase<BandLU, ExtendedBandMatrix<Ts...>> {
   public:
    using base_type = BandLUBase<intp::BandLU, ExtendedBandMatrix<Ts...>>;
    using matrix_type = typename base_type::matrix_type;
    using size_type = typename matrix_type::size_type;

   private:
    friend base_type;
    using base_type::base_type;
    using base_type::lu_store_;

    void compute_impl() {
        const size_type n = lu_store_.dim();
        const size_type p = lu_store_.lower_band_width();
        const size_type q = lu_store_.upper_band_width();

        for (size_type k = 0; k < n - 1; ++k) {
            // update main bands
            for (size_type i = k + 1; i < std::min(k + p + 1, n); ++i) {
                lu_store_.main_bands_val(i, k) /=
                    lu_store_.main_bands_val(k, k);
            }

            // update bottom side bands
            for (size_type i = std::max(n - q, k + p + 1); i < n; ++i) {
                lu_store_.side_bands_val(i, k) /=
                    lu_store_.main_bands_val(k, k);
            }

            // update main bands
            for (size_type j = k + 1; j < std::min(k + q + 1, n); ++j) {
                for (size_type i = k + 1; i < std::min(k + p + 1, n); ++i) {
                    lu_store_.main_bands_val(i, j) -=
                        lu_store_.main_bands_val(i, k) *
                        lu_store_.main_bands_val(k, j);
                }
            }

            // update upper right corner due to right side bands
            for (size_type i = k + 1; i < std::min(k + p + 1, n); ++i) {
                for (size_type j = std::max(n - p, k + q + 1); j < n; ++j) {
                    lu_store_(i, j) -= lu_store_.main_bands_val(i, k) *
                                       lu_store_.side_bands_val(k, j);
                }
            }

            // update lower left corner due to bottom side bands
            for (size_type j = k + 1; j < std::min(k + q + 1, n); ++j) {
                for (size_type i = std::max(n - q, k + p + 1); i < n; ++i) {
                    lu_store_(i, j) -= lu_store_.side_bands_val(i, k) *
                                       lu_store_.main_bands_val(k, j);
                }
            }

            // update main bands due to side bands
            if (k < std::max(n - p - 1, n - q - 1)) {
                for (size_type i = std::max(n - q, k + p + 1); i < n; ++i) {
                    for (size_type j = std::max(n - p, k + q + 1); j < n; ++j) {
                        lu_store_.main_bands_val(i, j) -=
                            lu_store_.side_bands_val(i, k) *
                            lu_store_.side_bands_val(k, j);
                    }
                }
            }
        }
    }

    template <typename Iter>
    void solve_in_place_impl(Iter& iter) const {
        const size_type n = lu_store_.dim();
        const size_type p = lu_store_.lower_band_width();
        const size_type q = lu_store_.upper_band_width();

        using ind_type = typename base_type::template ind_type_<
            util::remove_cvref_t<Iter>>::type;

        // apply l matrix
        for (size_type j = 0; j < n; ++j) {
            for (size_type i = j + 1; i < std::min(j + p + 1, n); ++i) {
                iter[static_cast<ind_type>(i)] -=
                    lu_store_.main_bands_val(i, j) *
                    iter[static_cast<ind_type>(j)];
            }

            // bottom side bands
            if (j < n - p - 1) {
                for (size_type i = std::max(n - q, j + p + 1); i < n; ++i) {
                    iter[static_cast<ind_type>(i)] -=
                        lu_store_.side_bands_val(i, j) *
                        iter[static_cast<ind_type>(j)];
                }
            }
        }
        // apply u matrix
        for (size_type j = n - 1; j < n; --j) {
            iter[static_cast<ind_type>(j)] /= lu_store_.main_bands_val(j, j);
            for (size_type i = j < q ? 0 : j - q; i < j; ++i) {
                iter[static_cast<ind_type>(i)] -=
                    lu_store_.main_bands_val(i, j) *
                    iter[static_cast<ind_type>(j)];
            }

            // right side bands
            if (j > n - p - 1) {
                for (size_type i = 0; i < j - q; ++i) {
                    iter[static_cast<ind_type>(i)] -=
                        lu_store_.side_bands_val(i, j) *
                        iter[static_cast<ind_type>(j)];
                }
            }
        }
    }
};

}  // namespace intp

#endif
// End content of "BandLU.hpp"

#ifdef INTP_MULTITHREAD
#endif

#if __cplusplus >= 201703L
#include <variant>
#endif

#ifdef INTP_TRACE
#include <iostream>
#endif

namespace intp {

template <typename T, std::size_t D, std::size_t O, typename U = double>
class InterpolationFunction;  // Forward declaration, since template has
                              // a member of it.

/**
 * @brief Template for interpolation with only coordinates, and generate
 * interpolation function when fed by function values.
 *
 */
template <typename T, std::size_t D, std::size_t O, typename U = double>
class InterpolationFunctionTemplate {
   public:
    using function_type = InterpolationFunction<T, D, O, U>;
    using size_type = typename function_type::size_type;
    using coord_type = typename function_type::coord_type;
    using val_type = typename function_type::val_type;
    using diff_type = typename function_type::diff_type;

    using ctrl_pt_type =
        typename function_type::spline_type::ControlPointContainer;

    static constexpr size_type dim = D;
    static constexpr size_type order = O;

    template <typename V>
    using DimArray = std::array<V, dim>;

    using MeshDim = MeshDimension<dim>;

    /**
     * @brief Construct a new Interpolation Function Template object, all other
     * constructors delegate to this one
     *
     * @param periodicity Periodicity of each dimension
     * @param interp_mesh_dimension The structure of coordinate mesh
     * @param x_ranges Begin and end iterator/value pairs of each dimension
     */
    template <typename... Ts>
    InterpolationFunctionTemplate(DimArray<bool> periodicity,
                                  MeshDim interp_mesh_dimension,
                                  std::pair<Ts, Ts>... x_ranges)
        : input_coords_{},
          mesh_dimension_(interp_mesh_dimension),
          base_(periodicity, input_coords_, mesh_dimension_, x_ranges...),
          solvers_{} {
        // active union member accordingly
        for (size_type i = 0; i < dim; ++i) {
#if __cplusplus >= 201703L
            if (periodicity[i]) {
                solvers_[i].template emplace<extended_solver_type>();
            } else {
                solvers_[i].template emplace<base_solver_type>();
            }
#endif
        }
        build_solver_();
    }

    /**
     * @brief Construct a new 1D Interpolation Function Template object.
     *
     * @param periodicity whether to construct a periodic spline
     * @param f_length point number of to-be-interpolated data
     * @param x_range a pair of x_min and x_max
     */
    template <typename C1, typename C2>
    InterpolationFunctionTemplate(bool periodicity,
                                  size_type f_length,
                                  std::pair<C1, C2> x_range)
        : InterpolationFunctionTemplate({periodicity},
                                        MeshDim{f_length},
                                        x_range) {
        static_assert(
            dim == size_type{1},
            "You can only use this overload of constructor in 1D case.");
    }

    template <typename C1, typename C2>
    InterpolationFunctionTemplate(size_type f_length, std::pair<C1, C2> x_range)
        : InterpolationFunctionTemplate(false, f_length, x_range) {}

    /**
     * @brief Construct a new (aperiodic) Interpolation Function Template
     * object
     *
     * @param interp_mesh_dimension The structure of coordinate mesh
     * @param x_ranges Begin and end iterator/value pairs of each dimension
     */
    template <typename... Ts>
    InterpolationFunctionTemplate(MeshDim interp_mesh_dimension,
                                  std::pair<Ts, Ts>... x_ranges)
        : InterpolationFunctionTemplate({},
                                        interp_mesh_dimension,
                                        x_ranges...) {}

    template <typename MeshOrIterPair>
    function_type interpolate(MeshOrIterPair&& mesh_or_iter_pair) const& {
        function_type interp{base_};
        interp.spline_.load_ctrlPts(
            solve_for_control_points_(Mesh<val_type, dim>{
                std::forward<MeshOrIterPair>(mesh_or_iter_pair)}));
        return interp;
    }

    template <typename MeshOrIterPair>
    function_type interpolate(MeshOrIterPair&& mesh_or_iter_pair) && {
        base_.spline_.load_ctrlPts(
            solve_for_control_points_(Mesh<val_type, dim>{
                std::forward<MeshOrIterPair>(mesh_or_iter_pair)}));
        return std::move(base_);
    }

    // modify the given interpolation function
    template <typename MeshOrIterPair>
    void interpolate(function_type& interp,
                     MeshOrIterPair&& mesh_or_iter_pair) const& {
        interp = base_;
        interp.spline_.load_ctrlPts(
            solve_for_control_points_(Mesh<val_type, dim>{
                std::forward<MeshOrIterPair>(mesh_or_iter_pair)}));
    }

#ifdef INTP_CELL_LAYOUT
    // pre calculate base spline values that can be reused in evaluating the
    // spline before actually providing weights
    template <typename... Coords,
              typename = typename std::enable_if<std::is_arithmetic<
                  typename std::common_type<Coords...>::type>::value>::type>
#if __cplusplus >= 201402L
    auto
#else
    std::function<val_type(const function_type&)>
#endif
    eval_proxy(Coords... x) const {
        return base_.eval_proxy(DimArray<coord_type>{x...});
    }

#if __cplusplus >= 201402L
    auto
#else
    std::function<val_type(const function_type&)>
#endif
    eval_proxy(DimArray<coord_type> coord) const {
        return base_.eval_proxy(coord);
    }

#if __cplusplus >= 201402L
    using eval_proxy_t = decltype(std::declval<function_type>().eval_proxy(
        DimArray<coord_type>{}));
#else
    using eval_proxy_t = std::function<val_type(const function_type&)>;
#endif

#endif  // INTP_CELL_LAYOUT

   private:
    using base_solver_type = BandLU<BandMatrix<coord_type>>;
    using extended_solver_type = BandLU<ExtendedBandMatrix<coord_type>>;

    // input coordinates, needed only in nonuniform case
    DimArray<typename function_type::spline_type::KnotContainer> input_coords_;

    MeshDim mesh_dimension_;

    // the base interpolation function with unspecified weights
    function_type base_;

#if __cplusplus >= 201703L
    using EitherSolver = std::variant<base_solver_type, extended_solver_type>;
#else
    // A union-like class storing BandLU solver for Either band matrix or
    // extended band matrix
    struct EitherSolver {
        // set default active union member, or g++ compiled code will throw
        // err
        EitherSolver() {}
        // Active union member and tag it.
        EitherSolver(bool is_periodic)
            : is_active_(true), is_periodic_(is_periodic) {
            if (is_periodic_) {
                new (&solver_periodic) extended_solver_type;
            } else {
                new (&solver_aperiodic) base_solver_type;
            }
        }
        EitherSolver& operator=(bool is_periodic) {
            if (is_active_) {
                throw std::runtime_error("Can not switch solver type.");
            }

            is_active_ = true;
            is_periodic_ = is_periodic;
            if (is_periodic_) {
                new (&solver_periodic) extended_solver_type;
            } else {
                new (&solver_aperiodic) base_solver_type;
            }

            return *this;
        }

        // Copy constructor is required by aggregate initialization, but never
        // invoked in this code since the content of this union is never
        // switched.
        EitherSolver(const EitherSolver&) {}
        // Destructor needed and it invokes either member's destructor according
        // to the tag "is_periodic";
        ~EitherSolver() {
            if (is_active_) {
                if (is_periodic_) {
                    solver_periodic.~extended_solver_type();
                } else {
                    solver_aperiodic.~base_solver_type();
                }
            }
        }

        union {
            base_solver_type solver_aperiodic;
            extended_solver_type solver_periodic;
        };

        bool is_active_ = false;
        // tag for union member
        bool is_periodic_ = false;
    };
#endif

    // solver for weights
    DimArray<EitherSolver> solvers_;

    void build_solver_() {
#ifndef INTP_PERIODIC_NO_DUMMY_POINT
        // adjust dimension according to periodicity
        {
            DimArray<size_type> dim_size_tmp;
            for (size_type d = 0; d < dim; ++d) {
                dim_size_tmp[d] = mesh_dimension_.dim_size(d) -
                                  (base_.periodicity(d) ? 1 : 0);
            }
            mesh_dimension_.resize(dim_size_tmp);
        }
#endif
        DimArray<typename function_type::spline_type::BaseSpline>
            base_spline_vals_per_dim;

        const auto& spline = base_.spline();

        // pre-calculate base spline of periodic dimension, since it never
        // changes due to its even-spaced knots
        for (size_type d = 0; d < dim; ++d) {
            if (base_.periodicity(d) && base_.uniform(d)) {
                base_spline_vals_per_dim[d] = spline.base_spline_value(
                    d, spline.knots_begin(d) + static_cast<diff_type>(order),
                    spline.knots_begin(d)[static_cast<diff_type>(order)] +
                        (1 - order % 2) * base_.dx_[d] * coord_type{.5});
            }
        }

#ifdef INTP_TRACE
        std::cout << "\n[TRACE] Coefficient Matrices\n";
#endif

        // loop through each dimension to construct coefficient matrix
        for (size_type d = 0; d < dim; ++d) {
            bool periodic = base_.periodicity(d);
            bool uniform = base_.uniform(d);
            auto mat_dim = mesh_dimension_.dim_size(d);
            auto band_width = periodic ? order / 2 : order == 0 ? 0 : order - 1;

#if __cplusplus >= 201703L
            std::variant<typename base_solver_type::matrix_type,
                         typename extended_solver_type::matrix_type>
                coef_mat;
            if (periodic) {
                coef_mat.template emplace<
                    typename extended_solver_type::matrix_type>(
                    mat_dim, band_width, band_width);
            } else {
                coef_mat
                    .template emplace<typename base_solver_type::matrix_type>(
                        mat_dim, band_width, band_width);
            }
#else
            typename extended_solver_type::matrix_type coef_mat(
                mat_dim, band_width, band_width);
#endif

#ifdef INTP_TRACE
            std::cout << "\n[TRACE] Dimension " << d << '\n';
            std::cout << "[TRACE] {0, 0} -> 1\n";
#endif

            for (size_type i = 0; i < mesh_dimension_.dim_size(d); ++i) {
                if (!periodic) {
                    // In aperiodic case, first and last data point can only
                    // be covered by one base spline, and the base spline at
                    // these ending points eval to 1.
                    if (i == 0 || i == mesh_dimension_.dim_size(d) - 1) {
#if __cplusplus >= 201703L
                        std::visit([i](auto& m) { m(i, i) = 1; }, coef_mat);
#else
                        coef_mat.main_bands_val(i, i) = 1;
#endif
                        continue;
                    }
                }

                const auto knot_num = spline.knots_num(d);
                // This is the index of knot point to the left of i-th
                // interpolated value's coordinate, notice that knot points has
                // a larger gap in both ends in non-periodic case.
                size_type knot_ind{};
                // flag for internal points in uniform aperiodic case
                const bool is_internal =
                    i > order / 2 &&
                    i < mesh_dimension_.dim_size(d) - order / 2 - 1;

                if (uniform) {
                    knot_ind =
                        periodic ? i + order
                                 : std::min(knot_num - order - 2,
                                            i > order / 2 ? i + (order + 1) / 2
                                                          : order);
                    if (!periodic) {
                        if (knot_ind <= 2 * order + 1 ||
                            knot_ind >= knot_num - 2 * order - 2) {
                            // out of the zone of even-spaced knots, update base
                            // spline
                            const auto iter = spline.knots_begin(d) +
                                              static_cast<diff_type>(knot_ind);
                            const coord_type x =
                                spline.range(d).first +
                                static_cast<coord_type>(i) * base_.dx_[d];
                            base_spline_vals_per_dim[d] =
                                spline.base_spline_value(d, iter, x);
                        }
                    }
                } else {
                    coord_type x = input_coords_[d][i];
                    // using BSpline::get_knot_iter to find current
                    // knot_ind
                    const auto iter =
                        periodic ? spline.knots_begin(d) +
                                       static_cast<diff_type>(i + order)
                        : i == 0 ? spline.knots_begin(d) +
                                       static_cast<diff_type>(order)
                        : i == input_coords_[d].size() - 1
                            ? spline.knots_end(d) -
                                  static_cast<diff_type>(order + 2)
                            : spline.get_knot_iter(
                                  d, x, i + 1,
                                  std::min(knot_num - order - 1, i + order));
                    knot_ind =
                        static_cast<size_type>(iter - spline.knots_begin(d));
                    base_spline_vals_per_dim[d] =
                        spline.base_spline_value(d, iter, x);
                }

                // number of base spline that covers present data point.
                const size_type s_num = periodic                 ? order | 1
                                        : order == 1             ? 1
                                        : uniform && is_internal ? order | 1
                                                                 : order + 1;
                for (size_type j = 0; j < s_num; ++j) {
#if __cplusplus >= 201703L
                    const size_type row = (i + (periodic ? band_width : 0)) %
                                          mesh_dimension_.dim_size(d);
                    const size_type col =
                        (knot_ind - order + j) % mesh_dimension_.dim_size(d);
                    std::visit(
                        [&](auto& m) {
                            m(row, col) = base_spline_vals_per_dim[d][j];
                        },
                        coef_mat);
#else
                    if (periodic) {
                        coef_mat((i + band_width) % mesh_dimension_.dim_size(d),
                                 (knot_ind - order + j) %
                                     mesh_dimension_.dim_size(d)) =
                            base_spline_vals_per_dim[d][j];
                    } else {
                        coef_mat.main_bands_val(i, knot_ind - order + j) =
                            base_spline_vals_per_dim[d][j];
                    }
#endif
#ifdef INTP_TRACE
                    std::cout
                        << "[TRACE] {"
                        << (periodic
                                ? (i + band_width) % mesh_dimension_.dim_size(d)
                                : i)
                        << ", "
                        << (knot_ind - order + j) % mesh_dimension_.dim_size(d)
                        << "} -> " << base_spline_vals_per_dim[d][j] << '\n';
#endif
                }
            }

#ifdef INTP_TRACE
            std::cout << "[TRACE] {" << mesh_dimension_.dim_size(d) - 1 << ", "
                      << mesh_dimension_.dim_size(d) - 1 << "} -> 1\n";
#endif

#if __cplusplus >= 201703L
            std::visit(
                [&](auto& solver) {
                    using matrix_t = typename util::remove_cvref_t<
                        decltype(solver)>::matrix_type;
                    solver.compute(std::get<matrix_t>(std::move(coef_mat)));
                },
                solvers_[d]);
#else
            solvers_[d] = periodic;
            if (periodic) {
                solvers_[d].solver_periodic.compute(std::move(coef_mat));
            } else {
                solvers_[d].solver_aperiodic.compute(
                    static_cast<typename base_solver_type::matrix_type&&>(
                        coef_mat));
            }
#endif
        }
    }

    ctrl_pt_type solve_for_control_points_(
        const Mesh<val_type, dim>& f_mesh) const {
        ctrl_pt_type weights{mesh_dimension_};
#ifdef INTP_PERIODIC_NO_DUMMY_POINT
        for (auto it = f_mesh.begin(); it != f_mesh.end(); ++it) {
            auto f_indices = f_mesh.iter_indices(it);
            for (size_type d = 0; d < dim; ++d) {
                if (base_.periodicity(d)) {
                    f_indices[d] =
                        (f_indices[d] + weights.dim_size(d) + order / 2) %
                        weights.dim_size(d);
                }
            }
            weights(f_indices) = *it;
        }
#else
        {
            auto check_idx =
                [&](typename Mesh<val_type, dim>::index_type& indices) {
                    bool keep_flag = true;
                    for (size_type d = 0; d < dim; ++d) {
                        if (base_.periodicity(d)) {
                            // Skip last point of periodic dimension
                            keep_flag =
                                keep_flag && indices[d] != weights.dim_size(d);
                            indices[d] =
                                (indices[d] + weights.dim_size(d) + order / 2) %
                                weights.dim_size(d);
                        }
                    }
                    return keep_flag;
                };

            // Copy interpolating values into weights mesh as the initial state
            // of the iterative control points solving algorithm
            for (auto it = f_mesh.begin(); it != f_mesh.end(); ++it) {
                auto f_indices = f_mesh.iter_indices(it);
                if (check_idx(f_indices)) { weights(f_indices) = *it; }
            }
        }
#endif

        ctrl_pt_type weights_tmp(1);  // auxilary weight for swapping between
        if CPP17_CONSTEXPR_ (dim > 1) { weights_tmp.resize(mesh_dimension_); }

        auto array_right_shift = [](DimArray<size_type> arr) {
            DimArray<size_type> new_arr{};
            for (size_type d_ = 0; d_ < dim; ++d_) {
                new_arr[d_] = arr[(d_ + dim - 1) % dim];
            }
            return new_arr;
        };

        // loop through each dimension to solve for control points
        for (size_type d = 0; d < dim; ++d) {
            ctrl_pt_type& old_weight = d % 2 == 0 ? weights : weights_tmp;
            ctrl_pt_type& new_weight = d % 2 != 0 ? weights : weights_tmp;

            if CPP17_CONSTEXPR_ (dim > 1) {
                new_weight.resize(array_right_shift(old_weight.dimension()));
            }

            const auto line_size = old_weight.dim_size(dim - 1);
            // size of hyperplane orthogonal to last dim axis
            const auto hyperplane_size = old_weight.size() / line_size;

            // prepare variables being captured by lambda
            auto& solver_wrapper = solvers_[dim - 1 - d];
            auto solve_and_rearrange_block = [&](size_type begin,
                                                 size_type end) {
                for (size_type j = begin; j < end; ++j) {
                    auto ind_arr =
                        old_weight.dimension().dimwise_indices(j * line_size);

                    auto old_iter_begin = old_weight.begin(dim - 1, ind_arr);
                    auto old_iter_end = old_weight.end(dim - 1, ind_arr);
                    auto new_iter_begin =
                        new_weight.begin(0, array_right_shift(ind_arr));
#if __cplusplus >= 201703L
                    std::visit(
                        [&](auto& solver) { solver.solve(old_iter_begin); },
                        solver_wrapper);
#else
                    if (base_.periodicity(dim - 1 - d)) {
                        solver_wrapper.solver_periodic.solve(old_iter_begin);
                    } else {
                        solver_wrapper.solver_aperiodic.solve(old_iter_begin);
                    }
#endif
                    if CPP17_CONSTEXPR_ (dim > 1) {
                        for (auto old_it = old_iter_begin,
                                  new_it = new_iter_begin;
                             old_it != old_iter_end; ++old_it, ++new_it) {
                            *new_it = *old_it;
                        }
                    }
                }
            };

#ifdef INTP_MULTITHREAD
            // TODO: use a more robust task division strategy
            const size_type block_num = static_cast<size_type>(
                std::sqrt(static_cast<double>(hyperplane_size)));
            const size_type task_per_block =
                block_num == 0 ? hyperplane_size : hyperplane_size / block_num;

            auto& thread_pool = DedicatedThreadPool<void>::get_instance(8);
            std::vector<std::future<void>> res;

            for (size_type i = 0; i < block_num; ++i) {
                // use thread pool here may seem to be a bit overhead
                res.push_back(thread_pool.queue_task([=]() {
                    solve_and_rearrange_block(i * task_per_block,
                                              (i + 1) * task_per_block);
                }));
            }
            // main thread deals with the remaining part in case hyperplane_size
            // not divisible by thread_num
            solve_and_rearrange_block(block_num * task_per_block,
                                      hyperplane_size);
            // wait for all tasks are complete
            for (auto&& f : res) { f.get(); }
#else
            solve_and_rearrange_block(0, hyperplane_size);
#endif  // INTP_MULTITHREAD
        }

        if CPP17_CONSTEXPR_ (dim % 2 == 0 || dim == 1) {
            return weights;
        } else {
            return weights_tmp;
        }
    }
};

template <std::size_t O, typename T = double, typename U = double>
class InterpolationFunctionTemplate1D
    : public InterpolationFunctionTemplate<T, size_t{1}, O, U> {
   private:
    using base = InterpolationFunctionTemplate<T, size_t{1}, O, U>;

   public:
    InterpolationFunctionTemplate1D(typename base::size_type f_length,
                                    bool periodicity = false)
        : InterpolationFunctionTemplate1D(
              std::make_pair(
                  typename base::coord_type{},
                  static_cast<typename base::coord_type>(f_length - 1)),
              f_length,
              periodicity) {}

    template <typename C1, typename C2>
    InterpolationFunctionTemplate1D(std::pair<C1, C2> x_range,
                                    typename base::size_type f_length,
                                    bool periodicity = false)
        : base(periodicity, f_length, x_range) {}
};

}  // namespace intp

#endif
// End content of "InterpolationTemplate.hpp"

namespace intp {

template <typename T, std::size_t D, std::size_t O, typename U>
class InterpolationFunction {
   public:
    using spline_type = BSpline<T, D, O, U>;

    using val_type = typename spline_type::val_type;
    using size_type = typename spline_type::size_type;
    using coord_type = typename spline_type::knot_type;
    using diff_type = typename spline_type::diff_type;

    constexpr static size_type dim = D;
    constexpr static size_type order = O;

    using function_type =
        InterpolationFunction<val_type, dim, order, coord_type>;

    template <typename T_>
    using DimArray = std::array<T_, dim>;

    friend class InterpolationFunctionTemplate<val_type,
                                               dim,
                                               order,
                                               coord_type>;

    /**
     * @brief Construct a new 1D Interpolation Function object, mimicking
     * Mathematica's `Interpolation` function, with option `Method->"Spline"`.
     *
     * @param periodic whether to construct a periodic spline
     * @param f_range a pair of iterators defining to-be-interpolated data
     * @param x_range a pair of x_min and x_max
     */
    template <
        typename InputIter,
        typename C1,
        typename C2,
        typename std::enable_if<
            dim == 1u &&
            std::is_convertible<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::input_iterator_tag>::value>::type* = nullptr>
    InterpolationFunction(bool periodic,
                          std::pair<InputIter, InputIter> f_range,
                          std::pair<C1, C2> x_range)
        : InterpolationFunction(
              {periodic},
              Mesh<val_type, 1u>{std::make_pair(f_range.first, f_range.second)},
              static_cast<std::pair<typename std::common_type<C1, C2>::type,
                                    typename std::common_type<C1, C2>::type>>(
                  x_range)) {}

    template <
        typename InputIter,
        typename C1,
        typename C2,
        typename std::enable_if<
            dim == 1u &&
            std::is_convertible<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::input_iterator_tag>::value>::type* = nullptr>
    InterpolationFunction(std::pair<InputIter, InputIter> f_range,
                          std::pair<C1, C2> x_range)
        : InterpolationFunction(false, f_range, x_range) {}

    /**
     * @brief Construct a new nD Interpolation Function object, mimicking
     * Mathematica's `Interpolation` function, with option `Method->"Spline"`.
     * Notice: last value of periodic dimension will be discarded since it is
     * considered same of the first value. Thus inconsistency input data for
     * periodic interpolation will be accepted.
     *
     * @param periodicity an array describing periodicity of each dimension
     * @param f_mesh a mesh containing data to be interpolated
     * @param x_ranges pairs of x_min and x_max or begin and end iterator
     */
    template <typename... Ts>
    InterpolationFunction(DimArray<bool> periodicity,
                          const Mesh<val_type, dim>& f_mesh,
                          std::pair<Ts, Ts>... x_ranges)
        : InterpolationFunction(
              InterpolationFunctionTemplate<val_type, dim, order, coord_type>(
                  periodicity,
                  f_mesh.dimension(),
                  x_ranges...)
                  .interpolate(f_mesh)) {}

    // Non-periodic for all dimension
    template <typename... Ts>
    InterpolationFunction(const Mesh<val_type, dim>& f_mesh,
                          std::pair<Ts, Ts>... x_ranges)
        : InterpolationFunction({}, f_mesh, x_ranges...) {}

    // constructor for partial construction, that is, without interpolated
    // values
    template <typename... Ts>
    InterpolationFunction(
        DimArray<bool> periodicity,
        DimArray<typename spline_type::KnotContainer>& input_coords,
        MeshDimension<dim> mesh_dimension,
        std::pair<Ts, Ts>... x_ranges)
        : spline_(periodicity) {
        // load knots into spline
        create_knots_(util::make_index_sequence_for<Ts...>{},
                      std::move(mesh_dimension), input_coords, x_ranges...);
    }

    // An empty interpolation function, can only be used after populated by
    // InterpolationTemplate::interpolate().
    InterpolationFunction() = default;

    /**
     * @brief Get spline value.
     *
     * @param x coordinates
     */
    template <typename... Coords,
              typename = typename std::enable_if<std::is_arithmetic<
                  typename std::common_type<Coords...>::type>::value>::type>
    val_type operator()(Coords... x) const {
        return call_op_helper({static_cast<coord_type>(x)...});
    }

    /**
     * @brief Get spline value.
     *
     * @param coord coordinate array
     */
    val_type operator()(DimArray<coord_type> coord) const {
        return call_op_helper(coord);
    }

    /**
     * @brief Get spline value, but with out of boundary check.
     *
     * @param coord coordinate array
     */
    val_type at(DimArray<coord_type> coord) const {
        boundary_check_(coord);
        return call_op_helper(coord);
    }

    /**
     * @brief Get spline value, but with out of boundary check.
     *
     * @param x coordinates
     */
    template <typename... Coords,
              typename Indices = util::make_index_sequence_for<Coords...>,
              typename = typename std::enable_if<std::is_arithmetic<
                  typename std::common_type<Coords...>::type>::value>::type>
    val_type at(Coords... x) const {
        return at(DimArray<coord_type>{static_cast<coord_type>(x)...});
    }

    /**
     * @brief Get spline derivative value.
     *
     * @param coord coordinate array
     * @param derivatives derivative order array
     */
    val_type derivative(DimArray<coord_type> coord,
                        DimArray<size_type> derivatives) const {
        return derivative_helper(util::make_index_sequence<dim>{}, coord,
                                 derivatives);
    }

    /**
     * @brief Get spline derivative value.
     *
     * @param coord coordinate array
     * @param deriOrder derivative orders
     */
    template <typename... Args>
    val_type derivative(DimArray<coord_type> coord, Args... deriOrder) const {
        return derivative(
            coord, DimArray<size_type>{static_cast<size_type>(deriOrder)...});
    }

    /**
     * @brief Get spline derivative value.
     *
     * @param coord_deriOrder_pair pairs of coordinate and derivative order
     */
    template <typename... CoordDeriOrderPair>
    val_type derivative(CoordDeriOrderPair... coord_deriOrder_pair) const {
        return derivative(DimArray<coord_type>{coord_deriOrder_pair.first...},
                          DimArray<size_type>{static_cast<size_type>(
                              coord_deriOrder_pair.second)...});
    }

    /**
     * @brief Get spline derivative value, but with out of boundary check.
     *
     * @param coord coordinate array
     * @param derivatives derivative order array
     */
    val_type derivative_at(DimArray<coord_type> coord,
                           DimArray<size_type> derivatives) const {
        boundary_check_(coord);
        return derivative_helper(util::make_index_sequence<dim>{}, coord,
                                 derivatives);
    }

    /**
     * @brief Get spline derivative value, but with out of boundary check.
     *
     * @param coord coordinate array
     * @param deriOrder derivative orders
     */
    template <typename... Args>
    val_type derivative_at(DimArray<coord_type> coord,
                           Args... deriOrder) const {
        return derivative_at(
            coord, DimArray<size_type>{static_cast<size_type>(deriOrder)...});
    }

    /**
     * @brief Get spline derivative value, but with out of boundary check.
     *
     * @param coord_deriOrder_pair pairs of coordinate and derivative order
     */
    template <typename... CoordDeriOrderPair>
    val_type derivative_at(CoordDeriOrderPair... coord_deriOrder_pair) const {
        return derivative_at(
            DimArray<coord_type>{coord_deriOrder_pair.first...},
            DimArray<size_type>{
                static_cast<size_type>(coord_deriOrder_pair.second)...});
    }

    // properties

    inline bool periodicity(size_type dim_ind) const {
        return spline_.periodicity(dim_ind);
    }

    inline bool uniform(size_type dim_ind) const { return uniform_[dim_ind]; }

    const std::pair<typename spline_type::knot_type,
                    typename spline_type::knot_type>&
    range(size_type dim_ind) const {
        return spline_.range(dim_ind);
    }

    /**
     * @brief Get a ref of underlying spline object
     *
     * @return spline_type&
     */
    const spline_type& spline() const { return spline_; }

    static constexpr size_type get_order() { return order; }

   private:
    spline_type spline_;

    DimArray<coord_type> dx_;
    DimArray<bool> uniform_;

    // auxiliary methods

    template <size_type... di>
    inline DimArray<std::pair<coord_type, size_type>> add_hint_for_spline(
        util::index_sequence<di...>,
        DimArray<coord_type> c) const {
        return {std::make_pair(
            c[di],
            uniform_[di]
                ? std::min(
                      spline_.knots_num(di) - order - 2,
                      static_cast<size_type>(std::ceil(std::max(
                          coord_type{0.},
                          (c[di] - range(di).first) / dx_[di] -
                              (periodicity(di)
                                   ? coord_type{1.}
                                   : coord_type{.5} * static_cast<coord_type>(
                                                          order + 1))))) +
                          order)
                : order)...};
    }

    inline val_type call_op_helper(DimArray<coord_type> c) const {
        return spline_(
            add_hint_for_spline(util::make_index_sequence<dim>{}, c));
    }

    template <size_type... di>
    inline val_type derivative_helper(util::index_sequence<di...>,
                                      DimArray<coord_type> c,
                                      DimArray<size_type> d) const {
        return spline_.derivative_at({std::make_tuple(
            static_cast<coord_type>(c[di]),
            uniform_[di]
                ? std::min(spline_.knots_num(di) - order - 2,
                           static_cast<size_type>(std::ceil(std::max(
                               0., (c[di] - range(di).first) / dx_[di] -
                                       (periodicity(di)
                                            ? 1.
                                            : .5 * static_cast<coord_type>(
                                                       order + 1))))) +
                               order)
                : order,
            static_cast<size_type>(d[di]))...});
    }

    // overload for uniform knots
    template <typename T_>
    typename std::enable_if<std::is_arithmetic<T_>::value>::type
    create_knot_vector_(size_type dim_ind,
                        const MeshDimension<dim>& mesh_dimension,
                        DimArray<typename spline_type::KnotContainer>&,
                        std::pair<T_, T_> x_range) {
        uniform_[dim_ind] = true;
        const auto periodic = periodicity(dim_ind);
        const size_type n = mesh_dimension.dim_size(dim_ind)
#ifdef INTP_PERIODIC_NO_DUMMY_POINT
                            + (periodic ? 1 : 0)
#endif
            ;
        dx_[dim_ind] =
            (x_range.second - x_range.first) / static_cast<coord_type>(n - 1);

        const size_t extra = periodic ? 2 * order + (1 - order % 2) : order + 1;

        std::vector<coord_type> xs(n + extra, x_range.first);

        if (periodic) {
            for (size_type i = 0; i < xs.size(); ++i) {
                xs[i] = x_range.first +
                        (static_cast<coord_type>(i) -
                         coord_type{.5} * static_cast<coord_type>(extra)) *
                            dx_[dim_ind];
            }
        } else {
            for (size_type i = order + 1; i < xs.size() - order - 1; ++i) {
                xs[i] = x_range.first +
                        (static_cast<coord_type>(i) -
                         coord_type{.5} * static_cast<coord_type>(extra)) *
                            dx_[dim_ind];
            }
            for (size_type i = xs.size() - order - 1; i < xs.size(); ++i) {
                xs[i] = x_range.second;
            }
        }

        spline_.load_knots(dim_ind, std::move(xs));
    }

    // overload for nonuniform knots, given by iterator pair
    template <typename T_>
    typename std::enable_if<std::is_convertible<
        typename std::iterator_traits<T_>::iterator_category,
        std::input_iterator_tag>::value>::type
    create_knot_vector_(
        size_type dim_ind,
        const MeshDimension<dim>& mesh_dimension,
        DimArray<typename spline_type::KnotContainer>& input_coords,
        std::pair<T_, T_> x_range) {
        uniform_[dim_ind] = false;
        const auto periodic = periodicity(dim_ind);

        const size_type n{static_cast<size_type>(
            std::distance(x_range.first, x_range.second))};
#ifdef INTP_PERIODIC_NO_DUMMY_POINT
        INTP_ASSERT(n == mesh_dimension.dim_size(dim_ind) + (periodic ? 1 : 0),
                    std::string("Inconsistency between knot number and "
                                "interpolated value number at dimension ") +
                        std::to_string(dim_ind));
#else
        INTP_ASSERT(n == mesh_dimension.dim_size(dim_ind),
                    std::string("Inconsistency between knot number and "
                                "interpolated value number at dimension ") +
                        std::to_string(dim_ind));
#endif
#ifndef INTP_ENABLE_ASSERTION
        // suppress unused parameter warning
        (void)mesh_dimension;
#endif
        typename spline_type::KnotContainer xs(
            periodic ? n + 2 * order + (1 - order % 2) : n + order + 1);

        auto& input_coord = input_coords[dim_ind];
        input_coord.reserve(n);
        // The x_range may be given by input iterators, which can not be
        // multi-passed.
        if (periodic) {
            // In periodic case, the knots are data points, shifted by half
            // of local grid size if spline order is odd.

            auto iter = x_range.first;
            input_coord.push_back(*iter);
            for (size_type i = order + 1; i < order + n; ++i) {
                coord_type present = *(++iter);
                xs[i] = order % 2 == 0 ? .5 * (input_coord.back() + present)
                                       : present;
                input_coord.push_back(present);
            }
            coord_type period = input_coord.back() - input_coord.front();
            for (size_type i = 0; i < order + 1; ++i) {
                xs[i] = xs[n + i - 1] - period;
                xs[xs.size() - i - 1] = xs[xs.size() - i - n] + period;
            }
        } else {
            // In aperiodic case, the internal knots are moving average of
            // data points with windows size equal to spline order.

            auto it = x_range.first;
            auto l_knot = *it;
            // fill leftmost *order+1* identical knots
            for (size_type i = 0; i < order + 1; ++i) { xs[i] = l_knot; }
            // first knot is same as first input coordinate
            input_coord.emplace_back(l_knot);
            // Every knot in middle is average of *order* input
            // coordinates. This var is to track the sum of a moving window
            // with width *order*.
            coord_type window_sum{};
            for (size_type i = 1; i < order; ++i) {
                input_coord.emplace_back(*(++it));
                window_sum += input_coord[i];
            }
            for (size_type i = order + 1; i < n; ++i) {
                input_coord.emplace_back(*(++it));
                window_sum += input_coord[i - 1];
                xs[i] = window_sum / static_cast<coord_type>(order);
                window_sum -= input_coord[i - order];
            }
            auto r_knot = *(++it);
            // fill rightmost *order+1* identical knots
            for (size_type i = n; i < n + order + 1; ++i) { xs[i] = r_knot; }
            // last knot is same as last input coordinate
            input_coord.emplace_back(r_knot);
        }
        // check whether input coordinates is monotonic
        for (std::size_t i = 0; i < input_coord.size() - 1; ++i) {
            INTP_ASSERT(input_coord[i + 1] > input_coord[i],
                        std::string("Given coordinate is not monotonically "
                                    "increasing at dimension ") +
                            std::to_string(dim_ind));
        }

#ifdef INTP_TRACE
        std::cout << "[TRACE] Nonuniform knots along dimension" << dim_ind
                  << ":\n";
        for (auto& c : xs) { std::cout << "[TRACE] " << c << '\n'; }
        std::cout << std::endl;
#endif

        spline_.load_knots(dim_ind, std::move(xs));
    }

    template <typename... Ts, size_type... di>
    void create_knots_(
        util::index_sequence<di...>,
        MeshDimension<dim> mesh_dimension,
        DimArray<typename spline_type::KnotContainer>& input_coords,
        std::pair<Ts, Ts>... x_ranges) {
#if __cplusplus >= 201703L
        (create_knot_vector_(di, mesh_dimension, input_coords, x_ranges), ...);
#else
        // polyfill of C++17 fold expression over comma
        static_cast<void>(std::array<std::nullptr_t, sizeof...(Ts)>{
            (create_knot_vector_(di, mesh_dimension, input_coords, x_ranges),
             nullptr)...});
#endif
    }

    inline void boundary_check_(const DimArray<coord_type>& coord) const {
        for (size_type d = 0; d < dim; ++d) {
            if (!periodicity(d) &&
                (coord[d] < range(d).first || coord[d] > range(d).second)) {
                throw std::domain_error(
                    "Given coordinate out of interpolation function "
                    "range!");
            }
        }
    }

#ifdef INTP_CELL_LAYOUT
#if __cplusplus >= 201402L
    auto
#else
    std::function<val_type(const function_type&)>
#endif
    eval_proxy(DimArray<coord_type> coords) const {
        auto spline_proxy = spline().pre_calc_coef(
            add_hint_for_spline(util::make_index_sequence<dim>{}, coords));
        return [spline_proxy](const function_type& interp) {
            return spline_proxy(interp.spline());
        };
    }
#endif  // INTP_CELL_LAYOUT
};

template <std::size_t O = std::size_t{3},
          typename T = double,
          typename U = double>
class InterpolationFunction1D
    : public InterpolationFunction<T, std::size_t{1}, O, U> {
   private:
    using base = InterpolationFunction<T, size_t{1}, O, U>;

   public:
    template <typename InputIter>
    InterpolationFunction1D(std::pair<InputIter, InputIter> f_range,
                            bool periodicity = false)
        : InterpolationFunction1D(
              std::make_pair(typename base::coord_type{},
                             static_cast<typename base::coord_type>(
                                 f_range.second - f_range.first
#ifdef INTP_PERIODIC_NO_DUMMY_POINT
                                 - (periodicity ? 0 : 1)
#else
                                 - 1
#endif
                                     )),
              f_range,
              periodicity) {
    }

    template <typename C1, typename C2, typename InputIter>
    InterpolationFunction1D(std::pair<C1, C2> x_range,
                            std::pair<InputIter, InputIter> f_range,
                            bool periodicity = false)
        : base(periodicity, f_range, x_range) {}
};

}  // namespace intp

#endif
// End content of "Interpolation.hpp"
