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
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    auto it = stats_type_.find(name);
    if (it == stats_type_.end()) { return; }
    switch (it->second)
    {
    case MetricType::COUNTER:
        counter_stats_[name] += value;
        break;
    case MetricType::GAUGE:
        gauge_stats_[name] = value;
        break;
    case MetricType::HISTOGRAM:
        if (histogram_stats_[name].size() < max_vector_len_) {
        {
            std::shared_ptr<std::mutex>& stat_mutex = GetStatMutex(name);
            std::lock_guard<std::mutex> lock(*stat_mutex);
            histogram_stats_[name].push_back(value);
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
    std::unique_lock<std::shared_mutex> read_lock(mutex_);

    std::unordered_map<std::string, double> counter_std;
    std::unordered_map<std::string, double> gauge_std;
    std::unordered_map<std::string, std::vector<double>> histogram_std;

    counter_stats_.swap(counter_std);
    gauge_stats_.swap(gauge_std);
    histogram_stats_.swap(histogram_std);

    auto result = std::make_tuple(
        std::move(counter_std),
        std::move(gauge_std),
        std::move(histogram_std)
    );
    return result;
}

} // namespace UC::Metrics