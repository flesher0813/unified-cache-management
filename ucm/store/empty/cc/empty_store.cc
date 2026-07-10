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
#include <sys/mman.h>
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
        for (auto& chunk : mmapChunks_) {
            if (chunk.registered) { Trans::Buffer::UnregisterHostBuffer(chunk.ptr); }
            if (chunk.ptr == MAP_FAILED) { continue; }
            (void)munlock(chunk.ptr, chunk.size);
            (void)munmap(chunk.ptr, chunk.size);
        }
        if (registeredHostBuffer_) { Trans::Buffer::UnregisterHostBuffer(mmapBuffer_); }
        if (mmapBuffer_ != MAP_FAILED) {
            (void)munlock(mmapBuffer_, mmapSize_);
            (void)munmap(mmapBuffer_, mmapSize_);
        }
        if (deviceSet_) {
            auto ret = aclrtResetDevice(deviceId_);
            std::cout << "EmptyStore mmap probe aclrtResetDevice ret=" << ret
                      << " device=" << deviceId_ << " resources: "
                      << ProcessResourceUsage() << std::endl;
        }
    }
    Status Setup(const Detail::Dictionary& config)
    {
        size_t totalGb = 0;
        config.GetNumber("cache_buffer_capacity_gb", totalGb);
        if (totalGb == 0) { return Status::OK(); }
        size_t chunkGb = 0;
        config.GetNumber("empty_register_chunk_gb", chunkGb);

        int64_t deviceId = -1;
        config.GetNumber("device_id", deviceId);
        if (deviceId < 0) {
            std::cout << "EmptyStore skip mmap allocation on non-worker device(" << deviceId
                      << ") resources: " << ProcessResourceUsage() << "." << std::endl;
            return Status::OK();
        }
        if (totalGb > SIZE_MAX / GiB) {
            return Status::InvalidParam("invalid cache_buffer_capacity_gb");
        }
        if (chunkGb > SIZE_MAX / GiB) {
            return Status::InvalidParam("invalid empty_register_chunk_gb");
        }

        deviceId_ = static_cast<int32_t>(deviceId);
        auto ret = aclrtSetDevice(deviceId_);
        std::cout << "EmptyStore mmap probe aclrtSetDevice ret=" << ret
                  << " device=" << deviceId_ << " resources: "
                  << ProcessResourceUsage() << std::endl;
        if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
        deviceSet_ = true;

        if (chunkGb > 0) { return SetupByChunks(totalGb * GiB, chunkGb * GiB); }
        return SetupSingle(totalGb * GiB);
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
    struct MMapChunk {
        void* ptr{MAP_FAILED};
        size_t size{0};
        bool registered{false};
    };

    static constexpr size_t KiB = 1024ULL;
    static constexpr size_t MiB = KiB * KiB;
    static constexpr size_t GiB = KiB * MiB;
    static constexpr size_t HugePageSize = 2 * MiB;
    void* mmapBuffer_{MAP_FAILED};
    size_t mmapSize_{0};
    bool registeredHostBuffer_{false};
    int32_t deviceId_{0};
    bool deviceSet_{false};
    std::vector<MMapChunk> mmapChunks_;

    Status SetupSingle(size_t totalSize)
    {
        mmapSize_ = AlignUp(totalSize, HugePageSize);
        std::cout << "EmptyStore mmap probe mmap begin size=" << mmapSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;
        mmapBuffer_ = mmap(nullptr, mmapSize_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mmapBuffer_ == MAP_FAILED) {
            return Status::OsApiError("empty mmap failed errno " + std::to_string(errno));
        }
        std::cout << "EmptyStore mmap probe mmap done ptr=" << mmapBuffer_
                  << " size=" << mmapSize_ << " resources: " << ProcessResourceUsage()
                  << std::endl;

        errno = 0;
        auto ret = madvise(mmapBuffer_, mmapSize_, MADV_HUGEPAGE);
        std::cout << "EmptyStore mmap probe madvise ret=" << ret << " errno="
                  << (ret == 0 ? 0 : errno) << " size=" << mmapSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;
        if (ret != 0) {
            return Status::OsApiError("empty madvise failed errno " + std::to_string(errno));
        }

        std::cout << "EmptyStore mmap probe touch once begin size=" << mmapSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;
        std::memset(mmapBuffer_, 0, mmapSize_);
        std::cout << "EmptyStore mmap probe touch once done size=" << mmapSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;

        errno = 0;
        ret = mlock(mmapBuffer_, mmapSize_);
        std::cout << "EmptyStore mmap probe mlock ret=" << ret << " errno="
                  << (ret == 0 ? 0 : errno) << " size=" << mmapSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;
        if (ret != 0) {
            return Status::OsApiError("empty mlock failed errno " + std::to_string(errno));
        }

        std::cout << "EmptyStore mmap probe register begin ptr=" << mmapBuffer_
                  << " size=" << mmapSize_ << " resources: " << ProcessResourceUsage()
                  << std::endl;
        errno = 0;
        auto s = Trans::Buffer::RegisterHostBuffer(mmapBuffer_, mmapSize_);
        auto err = errno;
        std::cout << "EmptyStore mmap probe register status=" << s.ToString()
                  << " errno=" << err << " ptr=" << mmapBuffer_ << " size=" << mmapSize_
                  << " resources: " << ProcessResourceUsage() << std::endl;
        if (s.Failure()) {
            std::cerr << "EmptyStore mmap probe register failed status=" << s.ToString()
                      << " errno=" << err << " ptr=" << mmapBuffer_
                      << " size=" << mmapSize_ << " resources: "
                      << ProcessResourceUsage() << std::endl;
            return s;
        }
        registeredHostBuffer_ = true;
        return Status::OK();
    }

    Status SetupByChunks(size_t totalSize, size_t chunkSize)
    {
        if (chunkSize == 0) { return Status::InvalidParam("invalid empty_register_chunk_gb"); }
        auto remaining = totalSize;
        size_t index = 0;
        while (remaining > 0) {
            auto size = remaining < chunkSize ? remaining : chunkSize;
            auto s = AllocateRegisterChunk(index, size);
            if (s.Failure()) { return s; }
            remaining -= size;
            index++;
        }
        std::cout << "EmptyStore mmap probe chunk register done chunks=" << mmapChunks_.size()
                  << " total_size=" << totalSize << " chunk_size=" << chunkSize
                  << " resources: " << ProcessResourceUsage() << std::endl;
        return Status::OK();
    }

    Status AllocateRegisterChunk(size_t index, size_t size)
    {
        auto alignedSize = AlignUp(size, HugePageSize);
        MMapChunk chunk{MAP_FAILED, alignedSize, false};

        std::cout << "EmptyStore mmap probe chunk[" << index << "] mmap begin size="
                  << alignedSize << " resources: " << ProcessResourceUsage() << std::endl;
        chunk.ptr = mmap(nullptr, alignedSize, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (chunk.ptr == MAP_FAILED) {
            return Status::OsApiError("empty chunk mmap failed errno " + std::to_string(errno));
        }
        mmapChunks_.push_back(chunk);
        auto& current = mmapChunks_.back();

        std::cout << "EmptyStore mmap probe chunk[" << index << "] mmap done ptr="
                  << current.ptr << " size=" << current.size
                  << " resources: " << ProcessResourceUsage() << std::endl;

        errno = 0;
        auto ret = madvise(current.ptr, current.size, MADV_HUGEPAGE);
        std::cout << "EmptyStore mmap probe chunk[" << index << "] madvise ret=" << ret
                  << " errno=" << (ret == 0 ? 0 : errno) << " size=" << current.size
                  << " resources: " << ProcessResourceUsage() << std::endl;
        if (ret != 0) {
            return Status::OsApiError("empty chunk madvise failed errno " + std::to_string(errno));
        }

        std::cout << "EmptyStore mmap probe chunk[" << index << "] touch once begin size="
                  << current.size << " resources: " << ProcessResourceUsage() << std::endl;
        std::memset(current.ptr, 0, current.size);
        std::cout << "EmptyStore mmap probe chunk[" << index << "] touch once done size="
                  << current.size << " resources: " << ProcessResourceUsage() << std::endl;

        errno = 0;
        ret = mlock(current.ptr, current.size);
        std::cout << "EmptyStore mmap probe chunk[" << index << "] mlock ret=" << ret
                  << " errno=" << (ret == 0 ? 0 : errno) << " size=" << current.size
                  << " resources: " << ProcessResourceUsage() << std::endl;
        if (ret != 0) {
            return Status::OsApiError("empty chunk mlock failed errno " + std::to_string(errno));
        }

        std::cout << "EmptyStore mmap probe chunk[" << index << "] register begin ptr="
                  << current.ptr << " size=" << current.size
                  << " resources: " << ProcessResourceUsage() << std::endl;
        errno = 0;
        auto s = Trans::Buffer::RegisterHostBuffer(current.ptr, current.size);
        auto err = errno;
        std::cout << "EmptyStore mmap probe chunk[" << index << "] register status="
                  << s.ToString() << " errno=" << err << " ptr=" << current.ptr
                  << " size=" << current.size << " resources: " << ProcessResourceUsage()
                  << std::endl;
        if (s.Failure()) {
            std::cerr << "EmptyStore mmap probe chunk[" << index
                      << "] register failed status=" << s.ToString() << " errno=" << err
                      << " ptr=" << current.ptr << " size=" << current.size
                      << " resources: " << ProcessResourceUsage() << std::endl;
            return s;
        }
        current.registered = true;
        return Status::OK();
    }

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
