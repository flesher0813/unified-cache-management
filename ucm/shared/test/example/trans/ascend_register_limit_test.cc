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
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <vector>
#include "trans/buffer.h"

namespace {

constexpr size_t KiB = 1024ULL;
constexpr size_t MiB = KiB * KiB;
constexpr size_t GiB = KiB * MiB;
constexpr size_t HugePageSize = 2 * MiB;

struct Config {
    int32_t deviceId{0};
    size_t totalSize{0};
    size_t chunkSize{0};
    bool touchFull{true};
    bool doMlock{false};
    bool runSingle{true};
    bool runChunked{true};
};

void Usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog
        << " --total-gb N --chunk-gb N [options]\n\n"
        << "Options:\n"
        << "  --total-gb N      Total mmap size in GiB.\n"
        << "  --total-mb N      Total mmap size in MiB.\n"
        << "  --chunk-gb N      Per-register chunk size in GiB.\n"
        << "  --chunk-mb N      Per-register chunk size in MiB.\n"
        << "  --device-id N     Ascend device id, default 0.\n"
        << "  --touch full      memset the whole mapping before register, default.\n"
        << "  --touch none      do not touch the mapping before register.\n"
        << "  --mlock           call mlock(total) before register and log result.\n"
        << "  --mode both       Run single total register and chunked register, default.\n"
        << "  --mode single     Only register the whole total size once.\n"
        << "  --mode chunked    Only register chunk by chunk.\n";
}

bool ParseUnsigned(std::string_view s, size_t& out)
{
    if (s.empty()) { return false; }
    size_t value = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') { return false; }
        const auto digit = static_cast<size_t>(ch - '0');
        if (value > (SIZE_MAX - digit) / 10) { return false; }
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

bool SetScaledSize(std::string_view arg, size_t scale, size_t& out)
{
    size_t value = 0;
    if (!ParseUnsigned(arg, value)) { return false; }
    if (value > SIZE_MAX / scale) { return false; }
    out = value * scale;
    return true;
}

bool ParseArgs(int argc, char** argv, Config& config)
{
    for (int i = 1; i < argc; i++) {
        const std::string_view opt{argv[i]};
        auto requireValue = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return {};
            }
            return argv[++i];
        };

        if (opt == "--total-gb") {
            if (!SetScaledSize(requireValue(opt), GiB, config.totalSize)) { return false; }
        } else if (opt == "--total-mb") {
            if (!SetScaledSize(requireValue(opt), MiB, config.totalSize)) { return false; }
        } else if (opt == "--chunk-gb") {
            if (!SetScaledSize(requireValue(opt), GiB, config.chunkSize)) { return false; }
        } else if (opt == "--chunk-mb") {
            if (!SetScaledSize(requireValue(opt), MiB, config.chunkSize)) { return false; }
        } else if (opt == "--device-id") {
            size_t value = 0;
            if (!ParseUnsigned(requireValue(opt), value)) { return false; }
            config.deviceId = static_cast<int32_t>(value);
        } else if (opt == "--touch") {
            const auto value = requireValue(opt);
            if (value == "full") {
                config.touchFull = true;
            } else if (value == "none") {
                config.touchFull = false;
            } else {
                return false;
            }
        } else if (opt == "--mlock") {
            config.doMlock = true;
        } else if (opt == "--mode") {
            const auto value = requireValue(opt);
            if (value == "both") {
                config.runSingle = true;
                config.runChunked = true;
            } else if (value == "single") {
                config.runSingle = true;
                config.runChunked = false;
            } else if (value == "chunked") {
                config.runSingle = false;
                config.runChunked = true;
            } else {
                return false;
            }
        } else if (opt == "--help" || opt == "-h") {
            return false;
        } else {
            std::cerr << "Unknown option: " << opt << "\n";
            return false;
        }
    }
    return config.totalSize > 0 && config.chunkSize > 0;
}

size_t AlignUp(size_t size, size_t alignment)
{
    return (size + alignment - 1) / alignment * alignment;
}

std::string GiBString(size_t size)
{
    return std::to_string(static_cast<double>(size) / static_cast<double>(GiB));
}

double NowSec()
{
    using Clock = std::chrono::steady_clock;
    static const auto start = Clock::now();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void* MMapWithAdvice(size_t& size)
{
    size = AlignUp(size, HugePageSize);
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "mmap failed size=" << size << " errno=" << errno << "\n";
        return MAP_FAILED;
    }
    auto ret = madvise(ptr, size, MADV_HUGEPAGE);
    std::cout << "madvise ret=" << ret << " errno=" << (ret == 0 ? 0 : errno) << "\n";
    return ptr;
}

void TouchFull(void* ptr, size_t size)
{
    const auto start = NowSec();
    auto* base = static_cast<std::byte*>(ptr);
    constexpr size_t step = GiB;
    for (size_t offset = 0; offset < size; offset += step) {
        const auto len = std::min(step, size - offset);
        std::memset(base + offset, 0, len);
        std::cout << "touch progress " << offset + len << "/" << size << " ("
                  << GiBString(offset + len) << "/" << GiBString(size) << " GiB)\n";
    }
    std::cout << "touch full done size=" << size << " (" << GiBString(size)
              << " GiB) elapsed=" << NowSec() - start << "s\n";
}

void MaybeMlock(void* ptr, size_t size)
{
    errno = 0;
    const auto start = NowSec();
    auto ret = mlock(ptr, size);
    auto eno = errno;
    std::cout << "mlock ret=" << ret << " errno=" << (ret == 0 ? 0 : eno)
              << " elapsed=" << NowSec() - start << "s\n";
}

bool RegisterRange(const char* tag, void* ptr, size_t size)
{
    errno = 0;
    std::cout << tag << " register begin ptr=" << ptr << " size=" << size << " ("
              << GiBString(size) << " GiB)\n";
    const auto start = NowSec();
    auto status = UC::Trans::Buffer::RegisterHostBuffer(ptr, size);
    auto eno = errno;
    std::cout << tag << " register ptr=" << ptr << " size=" << size << " ("
              << GiBString(size) << " GiB) status=" << status.ToString()
              << " errno=" << eno << " elapsed=" << NowSec() - start << "s\n";
    return status.Success();
}

void UnregisterRanges(const std::vector<void*>& ranges)
{
    for (auto* ptr : ranges) { UC::Trans::Buffer::UnregisterHostBuffer(ptr); }
}

void RunSingle(void* ptr, size_t totalSize)
{
    std::cout << "\n[single] register total size once\n";
    if (RegisterRange("single", ptr, totalSize)) {
        UC::Trans::Buffer::UnregisterHostBuffer(ptr);
        std::cout << "single unregister done\n";
    }
}

void RunChunked(void* ptr, size_t totalSize, size_t chunkSize)
{
    std::cout << "\n[chunked] register total by chunks chunkSize=" << chunkSize << " ("
              << GiBString(chunkSize) << " GiB)\n";
    std::vector<void*> registered;
    auto* base = static_cast<std::byte*>(ptr);
    for (size_t offset = 0, index = 0; offset < totalSize; offset += chunkSize, index++) {
        const auto size = std::min(chunkSize, totalSize - offset);
        auto* chunk = base + offset;
        const auto ok = RegisterRange(("chunk[" + std::to_string(index) + "]").c_str(), chunk, size);
        if (!ok) {
            std::cout << "chunked failed at index=" << index << " offset=" << offset << " ("
                      << GiBString(offset) << " GiB)\n";
            break;
        }
        registered.push_back(chunk);
    }
    UnregisterRanges(registered);
    std::cout << "chunked unregister done registered_chunks=" << registered.size() << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    Config config;
    if (!ParseArgs(argc, argv, config)) {
        Usage(argv[0]);
        return 2;
    }

    std::cout << "deviceId=" << config.deviceId << " totalSize=" << config.totalSize << " ("
              << GiBString(config.totalSize) << " GiB) chunkSize=" << config.chunkSize << " ("
              << GiBString(config.chunkSize) << " GiB) touchFull=" << config.touchFull
              << " mlock=" << config.doMlock << "\n";

    auto ret = aclrtSetDevice(config.deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice(" << config.deviceId << ") failed ret=" << ret << "\n";
        return 1;
    }

    auto mapSize = config.totalSize;
    void* ptr = MMapWithAdvice(mapSize);
    if (ptr == MAP_FAILED) { return 1; }
    std::cout << "mmap ptr=" << ptr << " mapSize=" << mapSize << " (" << GiBString(mapSize)
              << " GiB)\n";

    if (config.touchFull) { TouchFull(ptr, mapSize); }
    if (config.doMlock) { MaybeMlock(ptr, mapSize); }
    if (config.runSingle) { RunSingle(ptr, mapSize); }
    if (config.runChunked) { RunChunked(ptr, mapSize, config.chunkSize); }

    munmap(ptr, mapSize);
    aclrtResetDevice(config.deviceId);
    return 0;
}
