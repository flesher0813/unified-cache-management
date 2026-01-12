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
#ifndef UNIFIEDCACHE_METRICS_H
#define UNIFIEDCACHE_METRICS_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <atomic>
#include "concurrentqueue.h"
#include <iostream>

namespace UC::Metrics {

enum class MetricType { COUNTER, GAUGE, HISTOGRAM };

struct MetricTask
{
    MetricType op;
    std::string name;
    double value;
};
class Metrics {
public:
    static Metrics& GetInstance()
    {
        if (!is_inited_) {
            throw std::runtime_error("Please call SetUp() first!");
        }
        static Metrics inst;
        return inst;
    }

    static void SetUp(size_t maxVectorLen)
    {
        if (is_inited_.load(std::memory_order_acquire)) {
            return;
        }
        bool expected = false;
        if (is_inited_.compare_exchange_strong(
                expected,
                true,
                std::memory_order_release,
                std::memory_order_relaxed)) {
            max_vector_len_ = maxVectorLen;
        }
    }

    ~Metrics()
    {
        stop_flag_.store(true, std::memory_order_relaxed);
        size_t size = queue_.size_approx(); // wake up all worker threads
        std::cout << "Metrics destructor, queue size: " << size << std::endl;
        for (auto& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    void CreateStats(const std::string& name, const std::string& type);

    void UpdateStats(const std::string& name, double value);

    void UpdateStats(const std::unordered_map<std::string, double>& values);

    std::tuple<
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, std::vector<double>>
    > GetAllStatsAndClear();

private:
    std::mutex mutex_;
    std::mutex counter_mutex_, gauge_mutex_, histogram_mutex_;
    std::unordered_map<std::string, double> counter_stats_;
    std::unordered_map<std::string, double> gauge_stats_;
    std::unordered_map<std::string, std::vector<double>> histogram_stats_;
    std::unordered_map<std::string, MetricType> stats_type_;
    moodycamel::ConcurrentQueue<MetricTask> queue_;
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_flag_{false};

    void WorkerLoop();
    void ProcessNextTask();
    void ProcessSingleTask(const MetricTask& task);

    Metrics()
    {
        int threads_n = 4;
        for (int i = 0; i < threads_n; i++) {
            threads_.emplace_back(&Metrics::WorkerLoop, this);
        }
    }
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
    static std::atomic<bool> is_inited_;
    static size_t max_vector_len_;
};
} // namespace UC::Metrics

#endif // UNIFIEDCACHE_METRICS_H