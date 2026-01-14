#include <algorithm>
#include "metrics.h"

namespace UC::Metrics {

std::atomic<bool> Metrics::is_inited_{false};
size_t Metrics::max_vector_len_{10000};

void Metrics::Flush() {

    std::unique_lock<std::shared_mutex> flush_write_lock(mutex_);
    while (!queue_.size_approx() || is_processing_task_.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
    }
}

void Metrics::WorkerLoop() {
    while (!stop_flag_.load(std::memory_order_relaxed)) {
        ProcessNextTask();
        std::unique_lock<std::mutex> lock(cv_mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() {
            return stop_flag_.load(std::memory_order_relaxed) || !queue_.size_approx();
        });
    }

    MetricTask remaining_task;
    while (queue_.try_dequeue(remaining_task)) {
        ProcessSingleTask(remaining_task);
    }
}

void Metrics::ProcessSingleTask(const MetricTask& update_task) {
    if (update_task.name.empty()) {
        return;
    }

    switch (update_task.op) {
        case MetricType::COUNTER: {
            counter_stats_[update_task.name] += update_task.value;
            break;
        }
        case MetricType::GAUGE: {
            gauge_stats_[update_task.name] = update_task.value;
            break;
        }
        case MetricType::HISTOGRAM: {
            auto& vec = histogram_stats_[update_task.name];
            if (vec.size() < max_vector_len_) {
                vec.push_back(update_task.value);
            }
            break;
        }
        default:
            break;
    }
}

void Metrics::ProcessNextTask() {
    MetricTask update_task;
    if (queue_.try_dequeue(update_task)) {
        is_processing_task_.store(true, std::memory_order_relaxed);
        ProcessSingleTask(update_task);
        is_processing_task_.store(false, std::memory_order_relaxed);
    }
}

void Metrics::CreateStats(const std::string& name, const std::string& type) {
    std::unique_lock<std::shared_mutex> rw_lock(mutex_);
    std::string type_upper = type;
    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); }); // 修复 toupper 安全问题
    if (stats_type_.count(name)) {
        return;
    }
    if (type_upper == "COUNTER") {
        stats_type_[name] = MetricType::COUNTER;
    } else if (type_upper == "GAUGE") {
        stats_type_[name] = MetricType::GAUGE;
    } else if (type_upper == "HISTOGRAM") {
        stats_type_[name] = MetricType::HISTOGRAM;
    }
}

void Metrics::UpdateStats(const std::string& name, double value) {
    std::shared_lock<std::shared_mutex> rw_lock(mutex_);
    auto it = stats_type_.find(name);
    if (it == stats_type_.end()) {
        return;
    }
    queue_.enqueue({it->second, name, value});
    cv_.notify_one();
}

void Metrics::UpdateStats(const std::unordered_map<std::string, double>& values) {
    std::shared_lock<std::shared_mutex> rw_lock(mutex_);
    std::vector<MetricTask> tasks;
    for (const auto& pair : values) {
        auto it = stats_type_.find(pair.first);
        if (it != stats_type_.end()) {
            tasks.emplace_back(MetricTask{it->second, pair.first, pair.second});
        }
    }
    for (const auto& task : tasks) {
        queue_.enqueue(task);
    }
    if (!tasks.empty()) {
        cv_.notify_one();
    }
}

std::tuple<
    std::unordered_map<std::string, double>,
    std::unordered_map<std::string, double>,
    std::unordered_map<std::string, std::vector<double>>
> Metrics::GetAllStatsAndClear() {
    Flush();

    auto counter = std::exchange(counter_stats_, {});
    auto gauge = std::exchange(gauge_stats_, {});
    auto histogram = std::exchange(histogram_stats_, {});

    return std::make_tuple(std::move(counter), std::move(gauge), std::move(histogram));
}

} // namespace UC::Metrics