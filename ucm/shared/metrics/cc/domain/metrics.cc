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
thread_local std::shared_ptr<MetricBuffer> Metrics::thread_buffer_ = std::make_shared<MetricBuffer>();
thread_local bool Metrics::is_registered_thread_ = false;
thread_local Metrics::ThreadGuard Metrics::guard_ = Metrics::ThreadGuard(thread_buffer_);

std::atomic<bool> Metrics::is_inited_{false};
size_t Metrics::max_vector_len_{10000};

void Metrics::CreateStats(const std::string& name, const std::string& type)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
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
    if (!is_registered_thread_)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        buffers_.push_back({ thread_buffer_ });
        is_registered_thread_ = true;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = stats_type_.find(name);
    if (it == stats_type_.end()) { return; }

    switch (it->second)
    {
    case MetricType::COUNTER:
        thread_buffer_->counter_stats_[name] += value;
        break;
    case MetricType::GAUGE:
        thread_buffer_->gauge_stats_[name] = value;
        break;
    case MetricType::HISTOGRAM:
        if (thread_buffer_->histogram_stats_[name].size() < max_vector_len_) {
            thread_buffer_->histogram_stats_[name].push_back(value);
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

    std::unique_lock<std::shared_mutex> write_lock(mutex_);

    std::unordered_map<std::string, double> total_counter;
    std::unordered_map<std::string, double> total_gauge;
    std::unordered_map<std::string, std::vector<double>> total_histogram;

    for (const auto& buf : buffers_) {
        for (const auto& [name, value] : buf->counter_stats_) {
            total_counter[name] += value;
        }

        for (const auto& [name, value] : buf->gauge_stats_) {
            total_gauge[name] = value;
        }

        for (auto& [name, values] : buf->histogram_stats_) {
            total_histogram[name].insert(
                total_histogram[name].end(),
                values.begin(),
                values.end()
            );
        }
        buf->is_data_fetched_.store(true, std::memory_order_relaxed);
        buf->counter_stats_.clear();
        buf->gauge_stats_.clear();
        buf->histogram_stats_.clear();
    }

    auto result = std::make_tuple(
        std::move(total_counter),
        std::move(total_gauge),
        std::move(total_histogram)
    );

    buffers_.erase(
        std::remove_if(buffers_.begin(), buffers_.end(),
            [](const std::shared_ptr<MetricBuffer>& buf) {
                if (!buf) return true;
                return !buf->is_thread_alive_.load(std::memory_order_relaxed) 
                        && buf->is_data_fetched_.load(std::memory_order_relaxed);
            }),
        buffers_.end()
        );
    return result;
}

} // namespace UC::Metrics