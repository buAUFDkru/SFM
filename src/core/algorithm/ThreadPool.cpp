/**
 * @file ThreadPool.cpp
 * @brief ThreadPool 类的实现
 *
 * 实现要点：
 * - 固定线程数模型：构造时创建所有工作线程，析构时统一 join，不支持动态扩缩容。
 * - 工作窃取调度：每个 worker 维护自己的 WorkQueue，本地 LIFO + 远程 FIFO。
 * - 任务包装：统一使用 std::function<void()>，提交时的返回值/异常由 std::promise 处理。
 * - 关闭语义：shutdown() 仅设置停止标志，不会中断正在执行的任务；析构保证资源回收。
 * - 异常安全：worker 执行时捕获所有异常，避免单个任务异常导致整个线程池崩溃。
 */

#include "ThreadPool.h"

#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sfm {

/**
 * @brief 构造适合 CPU 密集型计算的线程池配置
 *
 * 默认行为：
 * - 线程数 = 硬件并发数（hardware_concurrency），至少 1 条。
 * - 启用工作窃取，使计算任务在核心间自动均衡。
 * - 全局队列无界，避免 CPU 任务因队列满被拒绝。
 */
thread_pool_options make_cpu_pool_options(std::size_t threads) {
    thread_pool_options options;
    options.name = "cpu_pool";
    // 环境变量 SFM_POOL_THREADS 可覆盖默认线程数（调试/基准测试用）
    if (threads == 0) {
        if (const char *env = std::getenv("SFM_POOL_THREADS")) {
            const long n = std::strtol(env, nullptr, 10);
            if (n >= 1) {
                threads = static_cast<std::size_t>(n);
            }
        }
    }
    if (threads == 0) {
        // 未指定线程数时，默认保留部分核心给 UI/OS，使用逻辑处理器数 / 2。
        // 例如 4 核 8 线程 CPU：hardware_concurrency() == 8，默认线程数为 4。
        unsigned int hc = std::thread::hardware_concurrency();
        options.thread_count = std::max<std::size_t>(1, hc > 3 ? hc / 2 : 1);
    } else {
        options.thread_count = threads;
    }
    // 限制线程池最多 2 个工作线程，避免在低功耗 CPU 上过度抢占 UI/系统线程调度时间。
    // 工作线程优先级已设为 BelowNormal，但线程数过多仍会导致整体吞吐下降（热降频）。
    // options.thread_count = std::min<std::size_t>(options.thread_count, 2);
    options.enable_work_stealing = true;
    options.max_queue_size = 0;
    return options;
}

ThreadPool &globalThreadPool(std::size_t threads) {
    // Meyers 单例：首次调用以传入线程数构造，生命周期至进程退出，
    // 析构时自动 shutdown 并 join 全部工作线程（线程安全由静态局部
    // 初始化的线程安全保证（C++11 magic statics））。
    static ThreadPool pool(make_cpu_pool_options(threads));
    return pool;
}

/**
 * @brief 线程局部变量定义
 *
 * 初始值为 size_t(-1)，表示当前线程不是本线程池的 worker。
 * 每个 worker 进入 worker_loop 时会将其设置为自己的 worker_id。
 */
thread_local std::size_t ThreadPool::current_worker_id_ = static_cast<std::size_t>(-1);

/**
 * @brief 构造函数：启动线程池
 *
 * 步骤：
 * 1. 规范化 thread_count（0 转硬件并发数）。
 * 2. 设置 running_ = true，stop_ = false。
 * 3. 为每个 worker 创建本地队列。
 * 4. 启动所有工作线程。
 *
 * 注意：构造过程中不会阻塞；工作线程启动后立即进入 worker_loop 等待任务。
 */
ThreadPool::ThreadPool(thread_pool_options options)
    : options_(std::move(options)) {
    if (options_.thread_count == 0) {
        options_.thread_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }

    // 发布 running_ 与 stop_ 状态，确保 worker 线程启动后能看到正确值
    running_.store(true, std::memory_order_release);
    stop_.store(false, std::memory_order_release);

    // 预先分配本地队列，避免后续 emplace 触发重新分配导致指针/引用失效
    local_queues_.reserve(options_.thread_count);
    for (std::size_t i = 0; i < options_.thread_count; ++i) {
        local_queues_.emplace_back(std::make_unique<WorkQueue>());
    }

    // 启动工作线程；每个线程绑定到自己的 worker_id
    workers_.reserve(options_.thread_count);
    for (std::size_t i = 0; i < options_.thread_count; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

/**
 * @brief 析构函数：安全关闭并回收所有线程
 *
 * 析构流程：
 * 1. shutdown() 设置停止标志并通知所有 worker。
 * 2. 逐个 join 工作线程；joinable() 检查防止重复 join。
 *
 * 析构会等待当前正在执行的任务完成，因此可能阻塞。
 */
ThreadPool::~ThreadPool() {
    shutdown();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

/**
 * @brief 请求关闭线程池
 *
 * 使用 compare_exchange_strong 保证 shutdown 只生效一次，避免重复通知。
 * 设置 stop_ 后：
 * - 新任务被拒绝（enqueue 检查 stop_ 返回 false）。
 * - running_ 设为 false，is_running() 返回 false。
 * - global_cv_ 唤醒所有等待任务的 worker，使其检查 stop_ 后退出。
 * - idle_cv_ 唤醒所有在 wait_for_idle() 中等待的调用者。
 */
void ThreadPool::shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return; // 已经关闭过，直接返回
    }
    running_.store(false, std::memory_order_release);
    global_cv_.notify_all();
    idle_cv_.notify_all();
}

/**
 * @brief 等待所有已提交任务完成
 *
 * 使用 idle_cv_ 进行阻塞等待，谓词为：
 * - 线程池已关闭，或
 * - pending_tasks_ == 0（没有排队或执行中的任务）。
 *
 * 注意：该函数不阻止其他线程继续提交任务；若调用者需要“提交-完成”语义，
 * 通常应在调用前停止继续提交，或配合 shutdown() 使用。
 */
void ThreadPool::wait_for_idle() {
    std::unique_lock<std::mutex> lock(global_mutex_);
    idle_cv_.wait(lock, [this] {
        return !running_.load(std::memory_order_acquire) ||
               pending_tasks_.load(std::memory_order_acquire) == 0;
    });
}

/**
 * @brief 返回工作线程总数
 */
std::size_t ThreadPool::thread_count() const noexcept {
    return workers_.size();
}

/**
 * @brief 返回尚未完成的任务数快照
 */
std::size_t ThreadPool::pending_tasks() const noexcept {
    return pending_tasks_.load(std::memory_order_acquire);
}

/**
 * @brief 返回线程池是否仍在运行
 */
bool ThreadPool::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

/**
 * @brief 任务入队总入口
 *
 * 选择路径：
 * - 若 stop_ 已设置，拒绝入队。
 * - 若调用者是本池 worker 且启用工作窃取，放入本地队列。
 * - 否则放入全局队列。
 *
 * pending_tasks_ 在任务真正加入队列后才递增；若入队失败则不会递增，
 * 因此 pending_tasks_ 始终精确反映“已接受但未完成”的任务数。
 */
bool ThreadPool::enqueue(task_type task) {
    if (stop_.load(std::memory_order_acquire)) {
        return false;
    }

    if (try_enqueue_local(task)) {
        return true;
    }

    return try_enqueue_global(std::move(task));
}

/**
 * @brief 尝试将任务放入当前 worker 的本地队列
 *
 * 本地入队条件：
 * - 线程池启用了工作窃取。
 * - 当前线程是本线程池的 worker（current_worker_id_ 有效）。
 *
 * 入队后调用 global_cv_.notify_one()：
 * 原因是从其他 worker 本地队列窃取的任务不会触发 global_cv_ 通知，
 * 因此需要唤醒一个可能正在休眠的 worker，让它有机会来窃取本任务。
 * 即使通知丢失或被当前忙碌的 worker 拿到，10ms 的超时等待也能保证最终进展。
 */
bool ThreadPool::try_enqueue_local(task_type& task) {
    if (!options_.enable_work_stealing) {
        return false;
    }

    std::size_t worker_id = current_worker_id_;
    if (worker_id == static_cast<std::size_t>(-1)) {
        return false; // 调用者不是本池 worker，无法放入本地队列
    }

    ++pending_tasks_;
    local_queues_[worker_id]->push(std::move(task));
    global_cv_.notify_one();
    return true;
}

/**
 * @brief 将任务放入全局队列
 *
 * 在 max_queue_size > 0 时执行有界检查；超过上限返回 false，不递增 pending_tasks_。
 * 入队成功后递增 pending_tasks_ 并通知一个等待中的 worker。
 *
 * global_mutex_ 同时保护全局队列状态和条件变量谓词，因此 notify_one 在锁内发出，
 * 避免 worker 在检查 global_queue_.empty() 和进入 wait 之间错过通知。
 */
bool ThreadPool::try_enqueue_global(task_type task) {
    std::lock_guard<std::mutex> lock(global_mutex_);

    if (options_.max_queue_size > 0 && global_queue_.size() >= options_.max_queue_size) {
        return false;
    }

    global_queue_.push(std::move(task));
    ++pending_tasks_;
    global_cv_.notify_one();
    return true;
}

/**
 * @brief 工作线程主循环
 *
 * 流程说明：
 * 1. 设置 thread-local current_worker_id_，使该线程后续 submit/detach 能进入本地队列。
 * 2. 循环直到 stop_ 被设置。
 * 3. 调用 try_take_task 按“本地 -> 窃取 -> 全局”顺序获取任务。
 * 4. 执行 task()；异常被捕获，防止单个任务搞垮整个 worker。
 * 5. 完成时 --pending_tasks_；若计数归零，通知 idle_cv_。
 * 6. 若未获取到任务，在 global_cv_ 上带 10ms 超时等待，随后重试。
 *    超时等待是防御性设计：本地队列入队时虽然会 notify_one，但无法保证
 *    每个空闲 worker 都能被精准唤醒；超时重试保证即使通知丢失也不会永久睡眠。
 */
void ThreadPool::worker_loop(std::size_t worker_id) {
    current_worker_id_ = worker_id;

    // 将工作线程优先级设为 BelowNormal，确保 UI 线程和系统线程优先调度。
    // 这对低功耗 CPU（如 i5-8250U）尤为重要：线程池满载时 UI 仍能保持响应。
#ifdef _WIN32
    HANDLE hThread = GetCurrentThread();
    if (hThread) {
        SetThreadPriority(hThread, THREAD_PRIORITY_BELOW_NORMAL);
    }
#else
    // POSIX: 使用 nice(10) 降低优先级（需注意权限，失败则忽略）
    nice(10);
#endif

    while (!stop_.load(std::memory_order_acquire)) {
        task_type task;
        bool taken = try_take_task(worker_id, task);
        if (taken) {
            try {
                if (task) {
                    task();
                }
            } catch (...) {
                // submit 任务的异常已由 promise 捕获；
                // 此处捕获的是 detach 任务或其他未处理的异常，作为兜底。
            }

            // 任务完成，递减待完成任务计数；归零时通知 wait_for_idle() 调用者
            if (--pending_tasks_ == 0) {
                idle_cv_.notify_all();
            }
        } else {
            std::unique_lock<std::mutex> lock(global_mutex_);
            global_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                return stop_.load(std::memory_order_acquire) || !global_queue_.empty();
            });
        }
    }
}

/**
 * @brief 按优先级获取任务
 *
 * 顺序：
 * 1. 本地队列 pop（LIFO）：最近提交的任务通常缓存最热。
 * 2. 窃取其他 worker（FIFO）：从其他队列头部取，减少与所有者 pop 的锁竞争。
 * 3. 全局队列 pop：兜底来源。
 */
bool ThreadPool::try_take_task(std::size_t worker_id, task_type& task) {
    // 1. 优先从本地队列取任务（LIFO，缓存友好）
    if (local_queues_[worker_id]->pop(task)) {
        return true;
    }

    // 2. 本地为空时尝试窃取其他线程的任务（FIFO，减少冲突）
    if (options_.enable_work_stealing && steal_task(worker_id, task)) {
        return true;
    }

    // 3. 最后从全局队列取任务（全局队列自带锁保护）
    return global_queue_.pop(task);
}

/**
 * @brief 从其他 worker 的本地队列窃取任务
 *
 * 遍历策略：
 * - 从 (worker_id + 1) % n 开始，依次访问其他 worker。
 * - 这种偏移起始点的方式避免所有空闲 worker 同时冲向 0 号队列，降低锁竞争热点。
 * - 每个 victim 只尝试 steal 一次；只要成功就返回，不追求一次性窃取多个任务。
 */
bool ThreadPool::steal_task(std::size_t worker_id, task_type& task) {
    const std::size_t n = local_queues_.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t victim = (worker_id + i + 1) % n;
        if (victim != worker_id && local_queues_[victim]->steal(task)) {
            return true;
        }
    }
    return false;
}

/* ============================================================================
 * WorkQueue 实现
 * ============================================================================ */

/**
 * @brief 将任务加入队列尾部
 *
 * 由队列所有者调用（worker 自己的本地队列，或外部线程操作全局队列）。
 */
void ThreadPool::WorkQueue::push(task_type task) {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push_back(std::move(task));
}

/**
 * @brief 从队列尾部弹出一个任务
 *
 * LIFO 语义。由队列所有者调用，与 steal() 操作的头部形成头尾分离，
 * 降低所有者与窃取者之间的锁竞争概率。
 */
bool ThreadPool::WorkQueue::pop(task_type& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.empty()) {
        return false;
    }
    task = std::move(tasks_.back());
    tasks_.pop_back();
    return true;
}

/**
 * @brief 从队列头部窃取一个任务
 *
 * FIFO 语义。由其他 worker 调用；只操作头部，与所有者 pop() 操作的尾部分离，
 * 在任务较多时可显著减少两者同时访问同一端造成的锁冲突。
 */
bool ThreadPool::WorkQueue::steal(task_type& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.empty()) {
        return false;
    }
    task = std::move(tasks_.front());
    tasks_.pop_front();
    return true;
}

/**
 * @brief 判断队列是否为空
 */
bool ThreadPool::WorkQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.empty();
}

/**
 * @brief 返回队列当前长度
 */
std::size_t ThreadPool::WorkQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

} // namespace sfm
