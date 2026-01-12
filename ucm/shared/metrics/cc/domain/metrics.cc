/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <algorithm>
#include "metrics.h"

namespace UC::Metrics {

std::atomic<bool> Metrics::is_inited_{false};
size_t Metrics::max_vector_len_{10000};
void Metrics::WorkerLoop()
{
    while(!stop_flag_.load(std::memory_order_relaxed)){
        ProcessNextTask();
    }
}

void Metrics::ProcessNextTask()
{
    MetricTask update_task;
    if (queue_.wait_dequeue(update_task))
    {
        if (stop_flag_.load(std::memory_order_relaxed)) {
            return;
        }
        switch (update_task.op)
        {
        case MetricType::COUNTER:
            std::lock_guard<std::mutex> counter_lock(counter_mutex_);
            counter_stats_[update_task.name] += update_task.value;
            break;
        case MetricType::GAUGE:
            std::lock_guard<std::mutex> gauge_lock(gauge_mutex_);
            gauge_stats_[update_task.name] = update_task.value;
            break;
        case MetricType::HISTOGRAM:
            std::lock_guard<std::mutex> histogram_lock(histogram_mutex_);
            histogram_stats_[update_task.name].push_back(update_task.value);
            break;
        
        default:
            break;
        }
    }

}

void Metrics::CreateStats(const std::string& name, const std::string& type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string type_upper = type;
    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);
    if (stats_type_.count(name)) {
        return;
    } else {
        if (type_upper == "COUNTER") {
            stats_type_[name] = MetricType::COUNTER;
        } else if (type_upper == "GAUGE") {
            stats_type_[name] = MetricType::GAUGE;
        } else if (type_upper == "HISTOGRAM") {
            stats_type_[name] = MetricType::HISTOGRAM;
        } else {
            return;
        }
    }
}

void Metrics::UpdateStats(const std::string& name, double value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_type_.find(name);
    if (it == stats_type_.end()) { return; }
    queue_.enqueue({it->second, name, value});
}

void Metrics::UpdateStats(const std::unordered_map<std::string, double>& values)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for(const auto& pair: values) {
        auto it = stats_type_.find(pair.first);
        if (it == stats_type_.end()) { continue; }
        queue_.enqueue({it->second, pair.first, pair.second});
    }
}

std::tuple<
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, std::vector<double>>
    > Metrics::GetAllStatsAndClear()
{
    std::scoped_lock lock(counter_mutex_, gauge_mutex_, histogram_mutex_);
    auto result = std::make_tuple(
        std::move(counter_stats_),
        std::move(gauge_stats_),
        std::move(histogram_stats_)
    );
    counter_stats_.clear();
    gauge_stats_.clear();
    histogram_stats_.clear();
    return result;
}

} // namespace UC::Metrics