/**
 * @file ThreadPool.h
 * @brief 面向 CPU 密集型计算的固定大小工作窃取线程池
 *
 * 设计目标：
 * - 为 SFM 重建流程中的特征提取、匹配、光束法平差等计算密集型阶段提供并行能力。
 * - 默认线程数等于硬件并发核心数，避免超订导致的上下文切换开销。
 * - 采用工作窃取（work-stealing）调度：每个 worker 优先处理自己本地队列中的任务，
 *   本地为空时从其他 worker 窃取，最后访问全局队列。这种 LIFO-local + FIFO-steal
 *   的组合在递归/分治任务中既能保留缓存局部性，又能平衡负载。
 * - 提交接口返回 std::future，便于在 Pipeline 中收集并行结果；也提供 detach 接口
 *   用于纯副作用任务。
 *
 * 线程安全：
 * - ThreadPool 对象构造后，submit/detach/shutdown/wait_for_idle 可安全地从任意线程调用。
 * - 析构函数会自动调用 shutdown 并 join 所有工作线程，保证资源不泄漏。
 * - 任务 callable 必须可以拷贝（因为底层存储在 std::function 中）。若需传移动语义对象，
 *   请用 std::shared_ptr 包装后传入 lambda。
 */

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sfm {

/**
 * @brief 线程池构造选项
 *
 * 所有字段均有合理默认值；对 CPU 密集型任务建议直接使用 make_cpu_pool_options()。
 */
struct thread_pool_options {
    std::string name = "thread_pool";       ///< 线程池名称，仅用于日志/调试标识
    std::size_t thread_count = 0;           ///< 工作线程数；0 表示使用硬件并发数
    bool enable_work_stealing = true;       ///< 是否启用工作窃取（CPU 密集型建议开启）
    std::size_t max_queue_size = 0;         ///< 全局队列最大长度；0 表示无界队列
};

/**
 * @brief 构造适合 CPU 密集型计算的线程池配置
 *
 * 配置细节：
 * - thread_count 默认近似为物理核心数（std::thread::hardware_concurrency() / 2），
 *   若无法获取则至少 1 条线程。例如 4 核 8 线程 CPU 默认创建 4 条线程。
 * - enable_work_stealing 置为 true，使任务在 worker 间自动均衡。
 * - max_queue_size 为 0，表示全局队列无界，避免 CPU 计算任务因队列满而被拒绝。
 *
 * @param threads 显式指定线程数；传 0 表示使用物理核心数近似值
 * @return 预调优的 CPU 密集型线程池配置
 */
thread_pool_options make_cpu_pool_options(std::size_t threads = 0);

namespace detail {

/**
 * @brief 将可调用对象与其参数绑定，形成无参数的可调用实体
 *
 * 作用类似 std::bind，但更清晰地处理 void 返回值与完美转发，
 * 并且保存参数时进行 decay，避免引用悬挂。
 *
 * @tparam Result 可调用对象返回值类型（可为 void）
 * @tparam F      可调用对象类型
 * @tparam Args   参数类型列表
 */
template <class Result, class F, class... Args>
class bound_task {
public:
    bound_task(F&& f, Args&&... args)
        : func_(std::forward<F>(f)),
          args_(std::forward<Args>(args)...) {}

    /**
     * @brief 执行绑定的可调用对象并返回结果
     *
     * 调用后本对象即被移动消耗，因此每个 bound_task 实例只应执行一次。
     */
    Result operator()() {
        if constexpr (std::is_void_v<Result>) {
            // void 返回类型：直接执行，不返回任何值
            std::apply(std::move(func_), std::move(args_));
        } else {
            // 非 void 返回类型：将执行结果返回给调用方
            return std::apply(std::move(func_), std::move(args_));
        }
    }

private:
    std::decay_t<F> func_;                      ///< decay 后的可调用对象副本
    std::tuple<std::decay_t<Args>...> args_;    ///< decay 后的参数元组
};

} // namespace detail

/**
 * @brief 支持工作窃取的固定大小线程池
 *
 * 内部结构：
 * - 每个工作线程拥有独立的 WorkQueue（本地队列）。
 * - 一个全局 WorkQueue 供外部线程提交任务。
 * - 一个条件变量 global_cv_ 用于唤醒等待任务的工作线程。
 * - 一个条件变量 idle_cv_ 用于 wait_for_idle() 等待所有任务完成。
 * - pending_tasks_ 原子计数跟踪“尚未完成”的任务数（含队列中 + 正在执行）。
 *
 * 任务调度顺序（try_take_task）：
 * 1. 从当前 worker 本地队列尾部 pop（LIFO）—— 最近放入的任务往往还在缓存中。
 * 2. 若本地为空且启用窃取，按 FIFO 从其他 worker 队列头部 steal。
 * 3. 若仍为空，从全局队列 pop。
 *
 * 任务入队顺序（enqueue）：
 * - 若调用者本身是某个 worker 线程且启用工作窃取，任务进入该 worker 的本地队列。
 * - 否则进入全局队列，并唤醒一个等待中的 worker。
 */
class ThreadPool {
public:
    /**
     * @brief 根据选项构造并启动线程池
     *
     * 构造函数会立即创建 options.thread_count 个工作线程并进入 worker_loop。
     * 若 options.thread_count 为 0，则自动使用硬件并发数。
     *
     * @param options 线程池配置选项
     */
    explicit ThreadPool(thread_pool_options options = {});

    /**
     * @brief 析构函数：安全关闭线程池并等待所有线程退出
     *
     * 析构时会：
     * 1. 调用 shutdown() 设置停止标志，禁止再接收新任务。
     * 2. 通知所有工作线程退出。
     * 3. join 所有 std::thread，确保线程资源被回收。
     *
     * 注意：若当前仍有任务在执行，析构会等待它们完成后再退出。
     */
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief 提交任务并返回 std::future 以获取结果
     *
     * 实现细节：
     * - 使用 detail::bound_task 将可调用对象与参数绑定，避免在任务包装中直接存储
     *   可变参数模板。
     * - 使用 std::promise / std::future 传递结果或异常。
     * - 捕获 std::shared_ptr 而非 std::unique_ptr，使 lambda 可被拷贝进 std::function。
     * - 任务执行时发生的异常会被捕获并通过 promise::set_exception 传给 future。
     *
     * @tparam F    可调用对象类型
     * @tparam Args 参数类型列表
     * @param f     可调用对象
     * @param args  传递给 f 的参数
     * @return 与提交任务关联的 std::future
     * @throws std::runtime_error 线程池已关闭或全局队列已满（有界队列时）
     */
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> {
        using result_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
        using task_type = detail::bound_task<result_type, F, Args...>;

        // 绑定任务与参数；使用 shared_ptr 是因为最终要存入可拷贝的 std::function
        auto shared_task = std::make_shared<task_type>(
            std::forward<F>(f), std::forward<Args>(args)...);

        // 使用 shared_ptr<promise> 保证 lambda 拷贝时 promise 生命周期安全
        auto promise = std::make_shared<std::promise<result_type>>();
        auto future = promise->get_future();

        // work lambda 捕获 shared_ptr，因此本身可拷贝，满足 std::function 要求
        auto work = [promise, shared_task]() mutable {
            try {
                if constexpr (std::is_void_v<result_type>) {
                    (*shared_task)();
                    promise->set_value();
                } else {
                    promise->set_value((*shared_task)());
                }
            } catch (...) {
                // 任何异常都传递给 future.get() 的调用方
                promise->set_exception(std::current_exception());
            }
        };

        if (!enqueue(std::move(work))) {
            throw std::runtime_error("thread pool is shutting down or queue is full");
        }

        return future;
    }

    /**
     * @brief 提交无需等待结果的 fire-and-forget 任务
     *
     * 与 submit 不同：
     * - 不返回 future，调用者无法获取返回值或异常。
     * - 任务执行过程中抛出的异常会被吞掉，避免 terminate 整个进程。
     * - 入队失败时返回 false，不会抛出异常。
     *
     * 适用场景：纯副作用任务，如日志写入、缓存更新、进度通知等。
     *
     * @tparam F    可调用对象类型
     * @tparam Args 参数类型列表
     * @param f     可调用对象
     * @param args  传递给 f 的参数
     * @return true 成功入队；false 线程池已关闭或队列已满
     */
    template <class F, class... Args>
    bool detach(F&& f, Args&&... args) {
        using result_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
        using task_type = detail::bound_task<result_type, F, Args...>;

        auto shared_task = std::make_shared<task_type>(
            std::forward<F>(f), std::forward<Args>(args)...);

        auto work = [shared_task]() mutable {
            try {
                (*shared_task)();
            } catch (...) {
                // detached 任务没有 future，无法传递异常，只能吞掉
            }
        };

        return enqueue(std::move(work));
    }

    /**
     * @brief 请求关闭线程池
     *
     * 效果：
     * - 设置 stop_ 原子标志，后续 submit/detach 将拒绝新任务。
     * - 设置 running_ 为 false。
     * - 唤醒所有在 global_cv_ 和 idle_cv_ 上等待的线程。
     *
     * 线程池不会立即停止正在执行的任务；工作线程会在当前任务完成后检查 stop_ 并退出。
     * 调用者通常应随后调用 wait_for_idle() 或等待析构完成。
     */
    void shutdown();

    /**
     * @brief 阻塞当前线程，直到所有已提交任务执行完毕
     *
     * 判断条件：pending_tasks_ == 0 或线程池已关闭。
     * 注意：该函数不会阻止新任务在 wait 期间被提交；若需精确控制，请配合 shutdown() 使用。
     */
    void wait_for_idle();

    /**
     * @brief 返回当前工作线程数量
     */
    std::size_t thread_count() const noexcept;

    /**
     * @brief 返回尚未完成的任务数量
     *
     * 包括：仍在队列中等待的 + 正在执行的。
     * 该值为原子变量快照，返回后可能立即变化。
     */
    std::size_t pending_tasks() const noexcept;

    /**
     * @brief 判断线程池是否处于运行状态
     *
     * shutdown() 调用后返回 false。
     */
    bool is_running() const noexcept;

    /**
     * @brief 调试辅助：返回当前线程的 worker_id
     */
    static std::size_t this_worker_id() noexcept { return current_worker_id_; }

private:
    using task_type = std::function<void()>;

    /**
     * @brief 线程安全的双端任务队列
     *
     * 使用单个 std::mutex 保护 std::deque。虽然高性能线程池常使用无锁双端队列，
     * 但此处优先保证正确性与可维护性；对 CPU 密集型任务，任务执行时间通常远大于
     * 队列操作时间，锁竞争影响有限。
     *
     * 操作语义：
     * - push / pop：操作队列尾部，供队列所有者使用（LIFO）。
     * - steal：操作队列头部，供其他 worker 使用（FIFO）。
     *   头尾分离可减少所有者 pop 与外部 steal 之间的锁冲突。
     */
    class WorkQueue {
    public:
        /**
         * @brief 将任务加入队列尾部
         */
        void push(task_type task);

        /**
         * @brief 从队列尾部弹出一个任务（LIFO）
         * @param task 输出参数
         * @return true 成功弹出；false 队列为空
         */
        bool pop(task_type& task);

        /**
         * @brief 从队列头部窃取一个任务（FIFO）
         * @param task 输出参数
         * @return true 成功窃取；false 队列为空
         */
        bool steal(task_type& task);

        /**
         * @brief 判断队列是否为空
         */
        bool empty() const;

        /**
         * @brief 返回队列当前长度
         */
        std::size_t size() const;

    private:
        mutable std::mutex mutex_;      ///< 保护 deque 的互斥锁
        std::deque<task_type> tasks_;   ///< 任务双端队列
    };

    /**
     * @brief 将任务加入合适的队列
     *
     * 选择策略：
     * - 若 stop_ 已设置，拒绝入队。
     * - 若当前线程是某个 worker 且启用工作窃取，优先放入该 worker 本地队列。
     * - 否则放入全局队列。
     *
     * @param task 任务包装
     * @return true 入队成功；false 线程池已关闭或全局队列已满
     */
    bool enqueue(task_type task);

    /**
     * @brief 尝试将任务放入当前 worker 的本地队列
     *
     * 只有在线程局部变量 current_worker_id_ 有效（即调用者是本线程池的 worker）时成功。
     * 入队后会 global_cv_.notify_one() 唤醒一个可能空闲的 worker 来窃取该任务。
     *
     * @param task 任务包装（仅在返回 true 时被移动）
     * @return true 成功放入本地队列；false 未启用窃取或调用者非 worker
     */
    bool try_enqueue_local(task_type& task);

    /**
     * @brief 将任务放入全局队列
     *
     * 在 max_queue_size > 0 且队列已满时返回 false。
     * 入队成功后通知一个等待中的 worker。
     *
     * @param task 任务包装
     * @return true 入队成功；false 全局队列已满
     */
    bool try_enqueue_global(task_type task);

    /**
     * @brief 工作线程主循环
     *
     * 每个线程启动时设置 thread-local current_worker_id_，然后循环：
     * 1. 尝试获取任务（本地 -> 窃取 -> 全局）。
     * 2. 若获取到任务则执行，完成时递减 pending_tasks_。
     * 3. 若未获取到任务则在 global_cv_ 上等待最多 10ms，随后重试。
     *    使用超时等待是为了避免本地队列入队的通知丢失导致线程永久睡眠。
     * 4. 检测到 stop_ 后退出循环。
     *
     * @param worker_id 当前 worker 在线程池中的索引
     */
    void worker_loop(std::size_t worker_id);

    /**
     * @brief 尝试为当前 worker 获取一个任务
     *
     * 优先级顺序：本地队列 > 窃取其他 worker > 全局队列。
     *
     * @param worker_id 当前 worker 索引
     * @param task 输出参数
     * @return true 获取到任务；false 所有队列均为空
     */
    bool try_take_task(std::size_t worker_id, task_type& task);

    /**
     * @brief 从其他 worker 的本地队列中窃取任务
     *
     * 遍历顺序从 (worker_id + 1) 开始，避免总是从 0 号线程开始造成热点。
     * 每个 victim 队列只尝试 steal 一次；若全部为空则返回 false。
     *
     * @param worker_id 当前 worker 索引
     * @param task 输出参数
     * @return true 窃取成功；false 无任务可窃取
     */
    bool steal_task(std::size_t worker_id, task_type& task);

    thread_pool_options options_;               ///< 线程池配置副本
    std::atomic<bool> running_{false};          ///< 线程池是否处于可运行状态
    std::atomic<bool> stop_{false};             ///< 是否已请求关闭
    std::atomic<std::size_t> pending_tasks_{0}; ///< 尚未完成的任务计数

    std::vector<std::thread> workers_;                  ///< 工作线程集合
    std::vector<std::unique_ptr<WorkQueue>> local_queues_; ///< 每个 worker 的本地队列
    WorkQueue global_queue_;                            ///< 全局任务队列

    std::mutex global_mutex_;                   ///< 保护全局队列及 global_cv_ 谓词
    std::condition_variable global_cv_;         ///< 唤醒等待任务的工作线程
    std::condition_variable idle_cv_;           ///< 用于 wait_for_idle() 等待空闲

    /**
     * @brief 线程局部变量：记录当前线程所在的 worker 索引
     *
     * 非线程池 worker 线程该值为 size_t(-1)。
     * 用于判断 submit/detach 是否应由 worker 线程将任务放入本地队列。
     */
    static thread_local std::size_t current_worker_id_;
};

} // namespace sfm

#endif // THREADPOOL_H
