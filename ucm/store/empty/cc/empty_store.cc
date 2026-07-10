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
#include <acl/acl.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>
#include "trans/buffer.h"
#include "ucmstore_v1.h"

namespace UC::EmptyStore {

namespace {

std::string ProcessResourceUsage()
{
    std::ostringstream os;
    os << "pid=" << getpid();

    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        os << " ru_maxrss_kb=" << usage.ru_maxrss;
    }

    auto appendStatusNumber = [](std::ostringstream& out, const std::string& line,
                                 const char* key, const char* name, const char* suffix) {
        if (line.rfind(key, 0) != 0) { return false; }
        auto begin = line.find_first_not_of(" \t", std::strlen(key));
        if (begin == std::string::npos) { return true; }
        auto end = line.find_first_of(" \t", begin);
        out << " " << name << suffix << "=" << line.substr(begin, end - begin);
        return true;
    };

    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (appendStatusNumber(os, line, "VmPeak:", "vm_peak", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "VmSize:", "vm_size", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "VmLck:", "vm_lck", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "VmHWM:", "vm_hwm", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "VmRSS:", "vm_rss", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "VmData:", "vm_data", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "VmSwap:", "vm_swap", "_kb")) { continue; }
        if (appendStatusNumber(os, line, "Threads:", "threads", "")) { continue; }
    }

    const char* cgroupPaths[] = {
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    };
    for (const auto* path : cgroupPaths) {
        std::ifstream cgroup(path);
        if (!cgroup.good()) { continue; }
        std::string value;
        if (std::getline(cgroup, value)) { os << " cg_mem_bytes=" << value; }
        break;
    }
    return os.str();
}

}  // namespace

std::vector<uint8_t> OnLookup(size_t num) { return std::vector<uint8_t>(num, false); }

class EmptyStore : public StoreV1 {
public:
    ~EmptyStore() override
    {
        if (registeredMappedHost_) { Trans::Buffer::UnregisterHostBuffer(hostBuffer_); }
        if (hostBuffer_ != nullptr) {
            auto ret = aclrtFreeHost(hostBuffer_);
            std::cout << "EmptyStore gqa iodirect=false probe aclrtFreeHost ret=" << ret
                      << " ptr=" << hostBuffer_ << " size=" << hostBufferSize_
                      << " resources: " << ProcessResourceUsage() << std::endl;
        }
        if (deviceSet_) {
            auto ret = aclrtResetDevice(deviceId_);
            std::cout << "EmptyStore gqa iodirect=false probe aclrtResetDevice ret=" << ret
                      << " device=" << deviceId_ << " resources: "
                      << ProcessResourceUsage() << std::endl;
        }
    }
    Status Setup(const Detail::Dictionary& config)
    {
        size_t totalGb = 0;
        config.GetNumber("cache_buffer_capacity_gb", totalGb);
        if (totalGb == 0) { return Status::OK(); }

        int64_t deviceId = -1;
        config.GetNumber("device_id", deviceId);
        if (deviceId < 0) {
            std::cout << "EmptyStore skip host allocation on non-worker device(" << deviceId
                      << ") resources: " << ProcessResourceUsage() << "." << std::endl;
            return Status::OK();
        }
        if (totalGb > SIZE_MAX / GiB) {
            return Status::InvalidParam("invalid cache_buffer_capacity_gb");
        }

        deviceId_ = static_cast<int32_t>(deviceId);
        auto ret = aclrtSetDevice(deviceId_);
        std::cout << "EmptyStore gqa iodirect=false probe aclrtSetDevice ret=" << ret
                  << " device=" << deviceId_ << " resources: "
                  << ProcessResourceUsage() << std::endl;
        if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
        deviceSet_ = true;

        size_t shardSize = 0;
        config.GetNumber("shard_size", shardSize);
        bool cacheSdmaDirect = false;
        config.Get("cache_sdma_direct", cacheSdmaDirect);
        return SetupGqaIoDirectFalse(totalGb * GiB, shardSize, cacheSdmaDirect);
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
    void* hostBuffer_{nullptr};
    size_t hostBufferSize_{0};
    void* deviceBuffer_{nullptr};
    bool registeredMappedHost_{false};
    int32_t deviceId_{0};
    bool deviceSet_{false};

    Status SetupGqaIoDirectFalse(size_t totalSize, size_t shardSize, bool cacheSdmaDirect)
    {
        if (shardSize != 0) { totalSize = totalSize / shardSize * shardSize; }
        if (totalSize == 0) {
            return Status::InvalidParam("cache_buffer_capacity_gb is smaller than shard_size");
        }

        hostBufferSize_ = totalSize;
        std::cout << "EmptyStore gqa iodirect=false probe aclrtMallocHost begin size="
                  << hostBufferSize_ << " shard_size=" << shardSize
                  << " resources: " << ProcessResourceUsage() << std::endl;
        errno = 0;
        auto ret = aclrtMallocHost(&hostBuffer_, hostBufferSize_);
        auto err = errno;
        std::cout << "EmptyStore gqa iodirect=false probe aclrtMallocHost ret=" << ret
                  << " errno=" << err << " ptr=" << hostBuffer_
                  << " size=" << hostBufferSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;
        if (ret != ACL_SUCCESS) {
            std::cerr << "EmptyStore gqa iodirect=false probe aclrtMallocHost failed ret="
                      << ret << " errno=" << err << " size=" << hostBufferSize_
                      << " resources: " << ProcessResourceUsage() << std::endl;
            return Status{ret, std::to_string(ret)};
        }
        if (cacheSdmaDirect) {
            std::cout << "EmptyStore gqa iodirect=false probe RegisterHostBuffer begin ptr="
                      << hostBuffer_ << " size=" << hostBufferSize_
                      << " resources: " << ProcessResourceUsage() << std::endl;
            errno = 0;
            auto s = Trans::Buffer::RegisterHostBuffer(hostBuffer_, hostBufferSize_,
                                                       &deviceBuffer_);
            err = errno;
            std::cout << "EmptyStore gqa iodirect=false probe RegisterHostBuffer status="
                      << s.ToString() << " errno=" << err << " ptr=" << hostBuffer_
                      << " size=" << hostBufferSize_ << " device_ptr=" << deviceBuffer_
                      << " resources: " << ProcessResourceUsage() << std::endl;
            if (s.Failure()) {
                std::cerr << "EmptyStore gqa iodirect=false probe RegisterHostBuffer failed"
                          << " status=" << s.ToString() << " errno=" << err
                          << " ptr=" << hostBuffer_ << " size=" << hostBufferSize_
                          << " resources: " << ProcessResourceUsage() << std::endl;
                return s;
            }
            registeredMappedHost_ = true;
        }
        return Status::OK();
    }
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::EmptyStore

extern "C" UC::StoreV1* MakeEmptyStore() { return new UC::EmptyStore::EmptyStore(); }
