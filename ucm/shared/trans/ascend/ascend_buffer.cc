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
#include "ascend_buffer.h"
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "logger/logger.h"

namespace UC::Trans {

namespace {

constexpr size_t kDirectIoAlignment = 4096;

bool IsAclrtHostDirectIoDebugEnabled()
{
    const char* env = std::getenv("UCM_DEBUG_ACLRT_HOST_DIRECT_IO");
    return env != nullptr && std::string_view(env) == "1";
}

std::string AclrtHostDirectIoDebugDir()
{
    const char* env = std::getenv("UCM_DEBUG_ACLRT_HOST_DIRECT_IO_DIR");
    if (env != nullptr && env[0] != '\0') { return env; }
    return "/tmp";
}

void TestHostBufferWrite(const char* tag, const char* path, int flags, const void* buffer,
                         size_t size)
{
    int fd = open(path, flags, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        UC_ERROR("Host buffer {} open({}) failed errno({}).", tag, path, errno);
        return;
    }

    errno = 0;
    auto nBytes = pwrite(fd, buffer, size, 0);
    auto eno = errno;
    UC_INFO("Host buffer {} pwrite path({}) buffer({}) size({}) nbytes({}) errno({}) "
            "align(buffer/size)=({}/{}).",
            tag, path, buffer, size, nBytes, eno,
            reinterpret_cast<uintptr_t>(buffer) % kDirectIoAlignment, size % kDirectIoAlignment);
    close(fd);
}

void DebugAclrtHostBufferDirectIo(void* raw, void* aligned, size_t size, size_t allocSize)
{
    const auto rawAddr = reinterpret_cast<uintptr_t>(raw);
    const auto alignedAddr = reinterpret_cast<uintptr_t>(aligned);
    const auto inRange = alignedAddr >= rawAddr && alignedAddr + size <= rawAddr + allocSize;
    UC_INFO("Aclrt host buffer raw({}) aligned({}) size({}) allocSize({}) delta({}) "
            "inRange({}) align(raw/aligned)=({}/{}).",
            raw, aligned, size, allocSize, alignedAddr - rawAddr, inRange,
            rawAddr % kDirectIoAlignment, alignedAddr % kDirectIoAlignment);

    if (!IsAclrtHostDirectIoDebugEnabled()) { return; }

    auto testSize = std::min<size_t>(size, 256 * 1024);
    testSize = testSize / kDirectIoAlignment * kDirectIoAlignment;
    if (testSize == 0) {
        UC_WARN("Skip aclrt host direct-io debug for too small buffer({}).", size);
        return;
    }

    std::memset(raw, 0x33, testSize);
    std::memset(aligned, 0x55, testSize);
    auto basePath = AclrtHostDirectIoDebugDir() + "/ucm_aclrt_host_direct_io_" +
                    std::to_string(getpid()) + "_" + std::to_string(alignedAddr);
    auto bufferedPath = basePath + "_buffered.bin";
    auto directPath = basePath + "_direct.bin";
    const int bufferedFlags = O_CREAT | O_TRUNC | O_WRONLY;
    const int directFlags = O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT;

    TestHostBufferWrite("raw-buffered", bufferedPath.c_str(), bufferedFlags, raw, testSize);
    TestHostBufferWrite("aligned-buffered", bufferedPath.c_str(), bufferedFlags, aligned,
                        testSize);
    TestHostBufferWrite("raw-direct", directPath.c_str(), directFlags, raw, testSize);
    TestHostBufferWrite("aligned-direct", directPath.c_str(), directFlags, aligned, testSize);
    unlink(bufferedPath.c_str());
    unlink(directPath.c_str());
}

}  // namespace

class HostHugePages : public std::enable_shared_from_this<HostHugePages> {
    struct ConstructorKey {};
    static constexpr auto HUGE_PAGE_SIZE = 2UL << 20;
    static constexpr auto GIGANTIC_PAGE_SIZE = 1UL << 30;
    static constexpr auto HUGE_PAGE_FLAG = 21 << MAP_HUGE_SHIFT;
    static constexpr auto GIGANTIC_PAGE_FLAG = 30 << MAP_HUGE_SHIFT;
    size_t size_;
    void* buffer_;

    static void* MMapWithTLB(size_t& size, bool useGiganticPages)
    {
        const auto pageSize = useGiganticPages ? GIGANTIC_PAGE_SIZE : HUGE_PAGE_SIZE;
        const auto alignedSize = (size + pageSize - 1) / pageSize * pageSize;
        const auto pageFlag = useGiganticPages ? GIGANTIC_PAGE_FLAG : HUGE_PAGE_FLAG;
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | pageFlag;
        void* ptr = mmap(nullptr, alignedSize, prot, flags, -1, 0);
        if (ptr == MAP_FAILED) {
            UC_WARN("Mmap({}) with TLB({}) return: {}.", alignedSize, pageSize, errno);
            return ptr;
        }
        size = alignedSize;
        return ptr;
    }
    static void* MMapWithAdvice(size_t& size)
    {
        const auto pageSize = HUGE_PAGE_SIZE;
        const auto alignedSize = (size + pageSize - 1) / pageSize * pageSize;
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS;
        void* ptr = mmap(nullptr, alignedSize, prot, flags, -1, 0);
        if (ptr == MAP_FAILED) {
            UC_WARN("Mmap({}) with advice({}) return: {}.", alignedSize, pageSize, errno);
            return ptr;
        }
        madvise(ptr, alignedSize, MADV_HUGEPAGE);
        size = alignedSize;
        return ptr;
    }

public:
    HostHugePages(size_t size, ConstructorKey) : size_(size), buffer_(MAP_FAILED) {}
    static std::shared_ptr<HostHugePages> Create(size_t size)
    {
        return std::make_shared<HostHugePages>(size, ConstructorKey{});
    }
    ~HostHugePages()
    {
        if (buffer_ == MAP_FAILED) { return; }
        Buffer::UnregisterHostBuffer(buffer_);
        munlock(buffer_, size_);
        munmap(buffer_, size_);
    }
    std::shared_ptr<void> Data()
    {
        if (buffer_ != MAP_FAILED) {
            return std::shared_ptr<void>(buffer_, [self = shared_from_this()](auto) {});
        }
        const auto useGiganticPages = size_ >= GIGANTIC_PAGE_SIZE;
        buffer_ = MMapWithTLB(size_, useGiganticPages);
        if (buffer_ == MAP_FAILED && useGiganticPages) { buffer_ = MMapWithTLB(size_, false); }
        if (buffer_ == MAP_FAILED) { buffer_ = MMapWithAdvice(size_); }
        if (buffer_ == MAP_FAILED) {
            UC_ERROR("Failed to make host buffer({}).", size_);
            return nullptr;
        }
        std::memset(buffer_, 0, size_);
        mlock(buffer_, size_);
        auto s = Buffer::RegisterHostBuffer(buffer_, size_);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to register buffer({}).", s, size_);
            munlock(buffer_, size_);
            munmap(buffer_, size_);
            buffer_ = MAP_FAILED;
            return nullptr;
        }
        return std::shared_ptr<void>(buffer_, [self = shared_from_this()](auto) {});
    }
};

std::shared_ptr<void> Trans::AscendBuffer::MakeDeviceBuffer(size_t size)
{
    void* device = nullptr;
    auto ret = aclrtMalloc(&device, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(device, aclrtFree); }
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = aclrtMallocHost(&host, size);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(host, aclrtFreeHost); }
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer4DirectIo(size_t size)
{
    constexpr size_t alignment = kDirectIoAlignment;
    const auto allocSize = size + alignment - 1;
    if (allocSize < size) [[unlikely]] { return nullptr; }

    void* raw = nullptr;
    auto ret = aclrtMallocHost(&raw, allocSize);
    if (ret != ACL_SUCCESS) [[unlikely]] {
        UC_ERROR("Failed({}) to make aligned host buffer({}).", ret, allocSize);
        return nullptr;
    }

    auto addr = reinterpret_cast<uintptr_t>(raw);
    auto alignedAddr = (addr + alignment - 1) & ~(alignment - 1);
    auto* aligned = reinterpret_cast<void*>(alignedAddr);
    DebugAclrtHostBufferDirectIo(raw, aligned, size, allocSize);
    return std::shared_ptr<void>(aligned, [raw](void*) { aclrtFreeHost(raw); });
}

Status Buffer::RegisterHostBuffer(void* host, size_t size, void** pDevice)
{
    void* device = nullptr;
#if ASCEND_SUPPORTS_REGISTER_PIN
    auto ret = aclrtHostRegisterV2(host, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    if (pDevice) { ret = aclrtHostGetDevicePointer(host, &device, 0); }
#else
    auto ret = aclrtHostRegister(host, size, ACL_HOST_REGISTER_MAPPED, &device);
#endif
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    if (pDevice) { *pDevice = device; }
    return Status::OK();
}

Status Buffer::GetHostDevicePointer(void* host, void** pDevice)
{
    void* device = nullptr;
    auto ret = aclrtHostGetDevicePointer(host, &device, 0);
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    if (pDevice) { *pDevice = device; }
    return Status::OK();
}

void Buffer::UnregisterHostBuffer(void* host) { aclrtHostUnregister(host); }

}  // namespace UC::Trans
