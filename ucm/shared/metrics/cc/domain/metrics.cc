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
#include <iostream>

namespace UC::Metrics {

std::atomic<bool> Metrics::is_inited_{false};
size_t Metrics::max_vector_len_{10000};

void Metrics::CreateStats(const std::string& name, const std::string& type)
{
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
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
            stats_mutex_[name] = std::make_shared<std::mutex>();
            stats_type_[name] = MetricType::HISTOGRAM;
        } else {
            return;
        }
    }
}

std::shared_ptr<std::mutex>& Metrics::GetStatMutex(const std::string &name)
{
    return stats_mutex_.at(name);
}

void Metrics::UpdateStats(const std::string& name, double value)
{
    MetricType type;
    {
        std::shared_lock<std::shared_mutex> read_lock(mutex_);
        auto it = stats_type_.find(name);
        if (it == stats_type_.end()) { return; }
        type = it->second;
    }
    
    int write_index = write_index_.load(std::memory_order_acquire);
    std::shared_ptr<MetricBuffer>& current_buffer = buffers[write_index];
    std::shared_lock<std::shared_mutex> read_lock(current_buffer->buffer_mutex_);
    
    switch (type)
    {
        case MetricType::COUNTER:
            current_buffer->counter_stats_[name] += value;
            break;
        case MetricType::GAUGE:
            current_buffer->gauge_stats_[name] = value;
            break;
        case MetricType::HISTOGRAM:
            if (current_buffer->histogram_stats_[name].size() < max_vector_len_) {
                std::shared_ptr<std::mutex>& stat_mutex = GetStatMutex(name);
                std::lock_guard<std::mutex> lock(*stat_mutex);
                current_buffer->histogram_stats_[name].push_back(value);
            }
            break;
        
        default:
            break;
    }
}

void Metrics::UpdateStats(const std::unordered_map<std::string, double>& values)
{
    for(const auto& pair: values) {
        UpdateStats(pair.first, pair.second);
    }
}

std::tuple<
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, std::vector<double>>
    > Metrics::GetAllStatsAndClear()
{
    int old_write_idx = write_index_.load(std::memory_order_acquire);
    int new_write_idx = 1 - old_write_idx;
    write_index_.store(new_write_idx, std::memory_order_release);

    std::shared_ptr<MetricBuffer>& current_buffer = buffers[old_write_idx];
    std::unique_lock<std::shared_mutex> old_buffer_lock(current_buffer->buffer_mutex_);

    std::unordered_map<std::string, double> counter_std;
    std::unordered_map<std::string, double> gauge_std;
    std::unordered_map<std::string, std::vector<double>> histogram_std;

    counter_std.reserve(current_buffer->counter_stats_.size());
    gauge_std.reserve(current_buffer->gauge_stats_.size());
    histogram_std.reserve(current_buffer->histogram_stats_.size());

    for (const auto& item : current_buffer->counter_stats_) {
        counter_std[item.first] = item.second;
    }

    for (const auto& item : current_buffer->gauge_stats_) {
        gauge_std[item.first] = item.second;
    }

    for (const auto& item : current_buffer->histogram_stats_) {
        histogram_std[item.first] = std::move(const_cast<std::vector<double>&>(item.second));
    }

    auto result = std::make_tuple(
        std::move(counter_std),
        std::move(gauge_std),
        std::move(histogram_std)
    );

    current_buffer->counter_stats_.clear();
    current_buffer->gauge_stats_.clear();
    current_buffer->histogram_stats_.clear();
    
    return result;
}

} // namespace UC::Metrics