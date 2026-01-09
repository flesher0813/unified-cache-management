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
#include <gtest/gtest.h>
#include <unistd.h>
#include "metrics_api.h"
#include <thread>
#include <iostream>
#include <chrono>

using namespace UC::Metrics;

std::atomic<bool> g_is_running(false);
std::atomic<uint64_t> g_update_count(0);

class UCStatsMonitorUT : public testing::Test {
protected:
    void SetUp() override
    {
        try {
            CreateStats("test_stats", "histogram");
            CreateStats("stats1", "histogram");
            CreateStats("stats2", "histogram");
        } catch (const std::exception& e) {
            throw;
        }
    }
};

TEST_F(UCStatsMonitorUT, SingleStatsUpdate)
{
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::string statsName = "test_stats";
    for (int i = 0; i < 10000; ++i) {
        UpdateStats(statsName, static_cast<double>(i));
    }
    auto stats = GetAllStatsAndClear();
    auto histogram_it = std::get<2>(stats).find(statsName);
    ASSERT_NE(histogram_it, std::get<2>(stats).end());
    const auto& histogram_values = histogram_it->second;
    ASSERT_EQ(histogram_values.size(), 10000);
    for (int i = 0; i < 10000; ++i) {
        ASSERT_EQ(histogram_values[i], static_cast<double>(i));
    }
}


TEST_F(UCStatsMonitorUT, DoubleBufferPerformance)
{
    std::string statsName = "test_stats";

    auto update_thread_func = []() {
        g_is_running.store(true, std::memory_order_release);
        const std::string counter_name = "test_counter";
        const std::string gauge_name = "test_gauge";
        const std::string histogram_name = "test_histogram";
        uint64_t local_count = 0; // 局部计数，减少原子操作开销

        // 持续更新Stats，直到g_is_running为false
        while (g_is_running.load(std::memory_order_acquire)) {
            // 批量更新3个指标（模拟大量更新）
            UpdateStats(
                {
                    {"test_stats", 1.0},
                    {"stats1", static_cast<double>(local_count % 1000)},
                    {"stats2", static_cast<double>(local_count % 100)},
                }
            );
            
            local_count++;
            // 每1000次更新同步一次原子变量（减少开销）
            if (local_count % 1000 == 0) {
                g_update_count.fetch_add(1000, std::memory_order_relaxed);
                local_count = 0;
            }
            // 可选：轻微休眠，避免CPU占满（根据测试需求调整）
            // std::this_thread::yield();
        }
        // 同步剩余的局部计数
        if (local_count > 0) {
            g_update_count.fetch_add(local_count, std::memory_order_relaxed);
        }
    };

    auto update_start_time = std::chrono::high_resolution_clock::now();
    std::thread update_thread(update_thread_func);

    const std::vector<int> check_points = {5, 10, 15, 20}; // 检查点（秒）
    for (int sec : check_points) {
        // 等待指定秒数
        std::this_thread::sleep_for(std::chrono::seconds(sec));
        
        // 记录Get开始时间
        auto get_start = std::chrono::high_resolution_clock::now();
        // 调用GetStatsAndClear
        auto histogram_stats = std::get<2>(GetAllStatsAndClear());
        std::cout << "histogram_stats size: "
                  << histogram_stats.size() << std::endl;
        for (const auto& pair : histogram_stats) {
            std::cout << "  " << pair.first << ": " << pair.second.size() << "." << std::endl;
        }
        // 记录Get结束时间并计算耗时
        auto get_end = std::chrono::high_resolution_clock::now();
        auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(get_end - get_start).count();
        
        // 计算截至当前的更新总耗时
        auto current_update_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            get_end - update_start_time
        ).count();
        
        // 打印统计信息
        std::cout << "===== 检查点：" << sec << "秒 =====" << std::endl;
        std::cout << "GetStatsAndClear耗时：" << get_duration << " 微秒" << std::endl;
        std::cout << "截至当前更新总耗时：" << current_update_duration << " 毫秒" << std::endl;
        std::cout << "截至当前更新总数：" << g_update_count.load(std::memory_order_relaxed) << " 次" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // ========== 步骤5：停止更新线程 + 最终读取 ==========
    // 停止更新线程
    g_is_running.store(false, std::memory_order_release);
    update_thread.join();
    // 记录更新总耗时
    auto update_end_time = std::chrono::high_resolution_clock::now();
    auto total_update_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        update_end_time - update_start_time
    ).count();
    
    // 最终读取并打印
    auto final_get_start = std::chrono::high_resolution_clock::now();
    auto final_stats = std::get<2>(GetAllStatsAndClear());
    for (const auto& pair : final_stats) {
            std::cout << "  " << pair.first << ": " << pair.second.size() << "." << std::endl;
        }
    auto final_get_end = std::chrono::high_resolution_clock::now();
    auto final_get_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        final_get_end - final_get_start
    ).count();

    std::cout << "===== 最终结果（更新结束） =====" << std::endl;
    std::cout << "最终GetStatsAndClear耗时：" << final_get_duration << " 微秒" << std::endl;
    std::cout << "更新总耗时：" << total_update_duration << " 毫秒" << std::endl;
    std::cout << "更新总次数：" << g_update_count.load(std::memory_order_relaxed) << " 次" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}