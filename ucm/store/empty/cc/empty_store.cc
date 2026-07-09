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
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include "ucmstore_v1.h"

namespace UC::EmptyStore {

std::vector<uint8_t> OnLookup(size_t num) { return std::vector<uint8_t>(num, false); }

class EmptyStore : public StoreV1 {
public:
    ~EmptyStore() override
    {
        if (mmapBuffer_ == MAP_FAILED) { return; }
        (void)munlock(mmapBuffer_, mmapSize_);
        (void)munmap(mmapBuffer_, mmapSize_);
    }
    Status Setup(const Detail::Dictionary& config)
    {
        size_t totalGb = 0;
        config.GetNumber("cache_buffer_capacity_gb", totalGb);
        if (totalGb == 0) { return Status::OK(); }

        int64_t deviceId = -1;
        config.GetNumber("device_id", deviceId);
        if (deviceId < 0) {
            std::cout << "EmptyStore skip mmap allocation on non-worker device(" << deviceId
                      << ")." << std::endl;
            return Status::OK();
        }
        if (totalGb > SIZE_MAX / GiB) {
            return Status::InvalidParam("invalid cache_buffer_capacity_gb");
        }

        mmapSize_ = AlignUp(totalGb * GiB, HugePageSize);
        mmapBuffer_ = mmap(nullptr, mmapSize_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mmapBuffer_ == MAP_FAILED) {
            return Status::OsApiError("empty mmap failed errno " + std::to_string(errno));
        }

        errno = 0;
        auto ret = madvise(mmapBuffer_, mmapSize_, MADV_HUGEPAGE);
        std::cout << "EmptyStore mmap probe madvise ret=" << ret << " errno="
                  << (ret == 0 ? 0 : errno) << " size=" << mmapSize_ << std::endl;

        std::cout << "EmptyStore mmap probe touch once begin size=" << mmapSize_ << std::endl;
        std::memset(mmapBuffer_, 0, mmapSize_);
        std::cout << "EmptyStore mmap probe touch once done size=" << mmapSize_ << std::endl;

        errno = 0;
        ret = mlock(mmapBuffer_, mmapSize_);
        std::cout << "EmptyStore mmap probe mlock ret=" << ret << " errno="
                  << (ret == 0 ? 0 : errno) << " size=" << mmapSize_ << std::endl;
        return Status::OK();
    }
    std::string Readme() const { return "EmptyStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num)
    {
        return OnLookup(num);
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) { return -1; }
    void Prefetch(const Detail::BlockId* blocks, size_t num) {}
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) { return NextId(); }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) { return NextId(); }
    Expected<bool> Check(Detail::TaskHandle taskId) { return true; }
    Status Wait(Detail::TaskHandle taskId) { return Status::OK(); }

private:
    static constexpr size_t KiB = 1024ULL;
    static constexpr size_t MiB = KiB * KiB;
    static constexpr size_t GiB = KiB * MiB;
    static constexpr size_t HugePageSize = 2 * MiB;
    void* mmapBuffer_{MAP_FAILED};
    size_t mmapSize_{0};

    static size_t AlignUp(size_t size, size_t alignment)
    {
        return (size + alignment - 1) / alignment * alignment;
    }
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::EmptyStore

extern "C" UC::StoreV1* MakeEmptyStore() { return new UC::EmptyStore::EmptyStore(); }
