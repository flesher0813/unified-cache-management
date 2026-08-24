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
#include "trans_buffer.h"
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <thread>
#include <vector>
#include <unistd.h>
#include "logger/logger.h"
#include "posix_shm.h"
#include "trans/buffer.h"
#include "trans/device.h"

namespace UC::CacheStore {

static constexpr size_t nHashTableBucket = 16411;
static constexpr auto invalidIndex = std::numeric_limits<size_t>::max();

static inline size_t Hash(const Detail::BlockId& blockId, size_t shard)
{
    static UC::Detail::BlockIdHasher blockIdHasher;
    static std::hash<size_t> shardHasher;
    constexpr auto goldenSection = 0x9e3779b97f4a7c15ULL;
    size_t h1 = blockIdHasher(blockId);
    size_t h2 = shardHasher(shard);
    return (h1 ^ (h2 + goldenSection + (h1 << 6) + (h1 >> 2))) % nHashTableBucket;
}

struct BufferMetaNode {
    Detail::BlockId block;
    size_t shard;
    size_t reference;
    size_t hash;
    size_t prev;
    size_t next;
    TransBuffer::State state;
    int32_t errorCode;
    void Init()
    {
        reference = 0;
        hash = invalidIndex;
        prev = invalidIndex;
        next = invalidIndex;
        state = TransBuffer::State::LOADING;
        errorCode = Status::OK().Underlying();
    }
};

class BufferStrategy {
protected:
    struct BaseConfig {
        int32_t deviceId{-1};
        size_t nodeSize{0};
        size_t totalSize{0};
        size_t reservedNumber{0};
        size_t segmentSize{0};
    };
    BaseConfig base_;

public:
    BufferStrategy(int32_t deviceId, size_t nodeSize, size_t totalSize, size_t reservedNumber,
                   size_t segmentSize)
        : base_({deviceId, nodeSize, totalSize, reservedNumber, segmentSize})
    {
    }
    virtual ~BufferStrategy() = default;
    virtual Status Setup() = 0;
    virtual void BucketLock(size_t iBucket) = 0;
    virtual bool BucketTryLock(size_t iBucket) = 0;
    virtual void BucketUnlock(size_t iBucket) = 0;
    virtual void NodeLock(size_t iNode) = 0;
    virtual void NodeUnlock(size_t iNode) = 0;
    virtual size_t& FirstAt(size_t iBucket) = 0;
    virtual size_t FetchNode(bool allowReserved) = 0;
    virtual void* DataAt(size_t iNode) = 0;
    virtual void* DeviceDataAt(size_t iNode) = 0;
    virtual BufferMetaNode* MetaAt(size_t iNode) = 0;

protected:
    static constexpr size_t segmentLogInterval = 8;
    static bool ShouldLogSegmentProgress(size_t idx, size_t segmentCount)
    {
        return idx == 0 || idx + 1 == segmentCount || (idx + 1) % segmentLogInterval == 0;
    }
    size_t SuggestedCapacity() const noexcept
    {
        const auto segmentSize = base_.segmentSize == 0 ? base_.totalSize : base_.segmentSize;
        return base_.totalSize > segmentSize ? base_.totalSize - segmentSize : 0;
    }
    void LogReduceCapacityHint() const
    {
        UC_ERROR("Try reducing cache_buffer_capacity_gb by one cache_segment_size_gb segment: "
                 "current capacity({}GB), segment size({}GB), suggested capacity({}GB).",
                 base_.totalSize >> 30, base_.segmentSize >> 30, SuggestedCapacity() >> 30);
    }
};

class LocalBufferStrategy : public BufferStrategy {
    struct BufferHeader {
        size_t buckets[nHashTableBucket];
        size_t freeHead;
        size_t nodeSize;
        size_t nNode;
    };
    struct LocalMutex {
        pthread_mutex_t mutex;
        ~LocalMutex() { pthread_mutex_destroy(&mutex); }
        void Init()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_PRIVATE);
            pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
            pthread_mutex_init(&mutex, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        void Lock() { pthread_mutex_lock(&mutex); }
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct LocalLock {
        pthread_spinlock_t lock;
        ~LocalLock() { pthread_spin_destroy(&lock); }
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };

    bool ioDirect_{false};
    bool mapHostToDevice_{false};
    BufferHeader header_;
    LocalMutex bucketLocks_[nHashTableBucket];
    std::unique_ptr<LocalLock[]> nodeLocks_;
    std::unique_ptr<BufferMetaNode[]> meta_;
    struct Segment {
        size_t nodeBegin{0};
        size_t nodeCount{0};
        size_t dataSize{0};
        std::shared_ptr<void> data;
        std::byte* dataOnDevice{nullptr};
        bool registeredMappedHost{false};
    };
    std::vector<Segment> segments_;
    size_t segmentNodeCount_{0};

public:
    LocalBufferStrategy(int32_t deviceId, size_t nodeSize, size_t totalSize, size_t reservedNumber,
                        size_t segmentSize, bool ioDirect, bool mapHostToDevice)
        : BufferStrategy(deviceId, nodeSize, totalSize, reservedNumber, segmentSize),
          ioDirect_(ioDirect),
          mapHostToDevice_(mapHostToDevice)
    {
    }
    ~LocalBufferStrategy() override
    {
        for (auto& seg : segments_) {
            if (seg.registeredMappedHost && seg.data) {
                Trans::Buffer::UnregisterHostBuffer(seg.data.get());
            }
        }
    }
    Status Setup() override
    {
        const auto deviceId = base_.deviceId;
        const auto totalSize = base_.totalSize;
        const auto nodeSize = base_.nodeSize;
        auto nNode = totalSize / nodeSize;
        try {
            nodeLocks_ = std::make_unique<LocalLock[]>(nNode);
            meta_ = std::make_unique<BufferMetaNode[]>(nNode);
            for (size_t i = 0; i < nHashTableBucket; i++) { bucketLocks_[i].Init(); }
            for (size_t i = 0; i < nNode; i++) { nodeLocks_[i].Init(); }
        } catch (const std::exception& e) {
            UC_ERROR("Failed({}) to alloc buffer.", e.what());
            return Status::Error(e.what());
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        auto buffer = device.MakeBuffer();
        if (!buffer) [[unlikely]] {
            UC_ERROR("Failed to make buffer on device({}).", deviceId);
            return Status::Error();
        }
        s = SetupSegments(*buffer, deviceId, nNode, nodeSize);
        if (s.Failure()) [[unlikely]] { return s; }
        for (size_t i = 0; i < nHashTableBucket; i++) { header_.buckets[i] = invalidIndex; }
        for (size_t i = 0; i < nNode; i++) { meta_[i].Init(); }
        header_.freeHead = 0;
        header_.nodeSize = nodeSize;
        header_.nNode = nNode;
        return Status::OK();
    }
    void BucketLock(size_t iBucket) override { bucketLocks_[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return bucketLocks_[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { bucketLocks_[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { nodeLocks_[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { nodeLocks_[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_.buckets[iBucket]; }
    size_t FetchNode(bool allowReserved) override
    {
        const auto limit = header_.nNode - (allowReserved ? 0 : base_.reservedNumber);
        if (header_.freeHead >= limit) { header_.freeHead = 0; }
        return header_.freeHead++;
    }
    void* DataAt(size_t iNode) override
    {
        auto& seg = SegmentAt(iNode);
        return static_cast<std::byte*>(seg.data.get()) +
               header_.nodeSize * (iNode - seg.nodeBegin);
    }
    void* DeviceDataAt(size_t iNode) override
    {
        auto& seg = SegmentAt(iNode);
        if (seg.dataOnDevice == nullptr) { return nullptr; }
        return seg.dataOnDevice + header_.nodeSize * (iNode - seg.nodeBegin);
    }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_.get() + iNode; }

private:
    static size_t SegmentNodeCount(size_t nNode, size_t nodeSize, size_t segmentSize)
    {
        if (segmentSize == 0) { return nNode; }
        return std::max<size_t>(1, segmentSize / nodeSize);
    }
    Status SetupSegments(Trans::Buffer& buffer, int32_t deviceId, size_t nNode, size_t nodeSize)
    {
        segmentNodeCount_ = SegmentNodeCount(nNode, nodeSize, base_.segmentSize);
        const auto segmentCount = (nNode + segmentNodeCount_ - 1) / segmentNodeCount_;
        for (size_t idx = 0, nodeBegin = 0; nodeBegin < nNode;
             idx++, nodeBegin += segmentNodeCount_) {
            const auto nodeCount = std::min(segmentNodeCount_, nNode - nodeBegin);
            const auto dataSize = nodeSize * nodeCount;
            Segment seg{nodeBegin, nodeCount, dataSize};
            seg.data = ioDirect_ ? buffer.MakeHostBuffer4DirectIo(dataSize)
                                 : buffer.MakeHostBuffer(dataSize);
            if (!seg.data) [[unlikely]] {
                UC_ERROR("Failed to make pinned segment({}/{}) begin node({}) size({}) for "
                         "device({}).",
                         idx + 1, segmentCount, nodeBegin, dataSize, deviceId);
                LogReduceCapacityHint();
                return Status::OutOfMemory();
            }
            if (mapHostToDevice_) {
                void* deviceData = nullptr;
                auto s = Status::OK();
                if (ioDirect_) {
                    s = Trans::Buffer::GetHostDevicePointer(seg.data.get(), &deviceData);
                } else {
                    s = Trans::Buffer::RegisterHostBuffer(seg.data.get(), dataSize, &deviceData);
                    seg.registeredMappedHost = s.Success();
                }
                if (s.Failure()) [[unlikely]] {
                    UC_ERROR("Failed({}) to map pinned host segment({}/{}) begin node({}) "
                             "size({}) to device({}).",
                             s, idx + 1, segmentCount, nodeBegin, dataSize, deviceId);
                    LogReduceCapacityHint();
                    return s;
                }
                seg.dataOnDevice = static_cast<std::byte*>(deviceData);
            }
            if (ShouldLogSegmentProgress(idx, segmentCount)) {
                UC_INFO("Local cache segment ready: index({}/{}), begin node({}), node count({}), "
                        "size({}), device({}).",
                        idx + 1, segmentCount, seg.nodeBegin, seg.nodeCount, seg.dataSize,
                        deviceId);
            }
            segments_.push_back(std::move(seg));
        }
        return Status::OK();
    }
    Segment& SegmentAt(size_t iNode)
    {
        return segments_[iNode / segmentNodeCount_];
    }
};

class SharedBufferStrategy : public BufferStrategy {
protected:
    struct ShareMutex {
        pthread_mutex_t mutex;
        ~ShareMutex() = delete;
        void Init()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
            pthread_mutex_init(&mutex, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        void Lock() { pthread_mutex_lock(&mutex); }
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct ShareLock {
        pthread_spinlock_t lock;
        ~ShareLock() = delete;
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };
    static constexpr size_t sharedBufferMagic = (('S' << 16) | ('b' << 8) | 2);
    struct BufferHeader {
        std::atomic<size_t> magic;
        ShareLock lock;
        size_t nNode;
        size_t freeHead;
        size_t nodeSize;
        size_t segmentNodeCount;
        size_t segmentCount;
        size_t buckets[nHashTableBucket];
        ShareMutex bucketLocks[nHashTableBucket];
        ShareLock nodeLocks[0];
    };

    BufferHeader* header_{nullptr};
    BufferMetaNode* meta_{nullptr};
    const std::string& uuid_;
    std::string shmName_;
    size_t nodeSize_{0};
    size_t nNode_{0};
    size_t segmentNodeCount_{0};
    size_t segmentCount_{0};
    void* addrress_{nullptr};
    size_t totalSize_{0};
    bool owner_{false};
    struct Segment {
        std::string shmName;
        void* address{nullptr};
        std::byte* data{nullptr};
        std::byte* dataOnDevice{nullptr};
        size_t mapSize{0};
        size_t nodeBegin{0};
        size_t nodeCount{0};
        size_t dataSize{0};
        bool registered{false};
    };
    std::vector<Segment> segments_;

    size_t MetaOffset() const noexcept { return sizeof(BufferHeader) + sizeof(ShareLock) * nNode_; }
    size_t DataOffset() const noexcept
    {
        static const auto pageSize = sysconf(_SC_PAGESIZE);
        const auto size = MetaOffset() + sizeof(BufferMetaNode) * nNode_;
        return (size + pageSize - 1) & ~(pageSize - 1);
    }
    size_t DataSize() const noexcept { return nodeSize_ * nNode_; }
    size_t SegmentNodeCount() const noexcept
    {
        if (base_.segmentSize == 0) { return nNode_; }
        return std::max<size_t>(1, base_.segmentSize / nodeSize_);
    }
    std::string SegmentShmName(size_t idx) const
    {
        return shmName_ + "_" + std::to_string(idx);
    }
    void InitSegments()
    {
        segmentNodeCount_ = SegmentNodeCount();
        segmentCount_ = (nNode_ + segmentNodeCount_ - 1) / segmentNodeCount_;
        segments_.clear();
        segments_.reserve(segmentCount_);
        for (size_t idx = 0; idx < segmentCount_; idx++) {
            const auto nodeBegin = idx * segmentNodeCount_;
            const auto nodeCount = std::min(segmentNodeCount_, nNode_ - nodeBegin);
            const auto dataSize = nodeSize_ * nodeCount;
            const auto mapSize = idx == 0 ? DataOffset() + dataSize : dataSize;
            segments_.push_back(Segment{SegmentShmName(idx), nullptr, nullptr, nullptr,
                                        mapSize, nodeBegin, nodeCount, dataSize, false});
        }
    }
    Segment& SegmentAt(size_t iNode)
    {
        return segments_[iNode / segmentNodeCount_];
    }
    static const std::string& ShmPrefix() noexcept
    {
        static std::string prefix{"uc_shm_cache_"};
        return prefix;
    }
    static void CleanUpShmFileExceptMe(const std::string& me)
    {
        namespace fs = std::filesystem;
        std::string_view prefix = ShmPrefix();
        const auto segmentPrefix = me + "_";
        fs::path shmDir = "/dev/shm";
        if (!fs::exists(shmDir)) { return; }
        const auto now = fs::file_time_type::clock::now();
        const auto keepThreshold = std::chrono::minutes(10);
        for (const auto& entry : fs::directory_iterator(shmDir)) {
            const auto& path = entry.path();
            const auto& name = path.filename().string();
            if (!entry.is_regular_file() || name.compare(0, prefix.size(), prefix) != 0 ||
                name == me || name.compare(0, segmentPrefix.size(), segmentPrefix) == 0) {
                continue;
            }
            try {
                const auto lwt = fs::last_write_time(path);
                if (now - lwt <= keepThreshold) { continue; }
                fs::remove(path);
            } catch (...) {
            }
        }
    }
    static Status MmapShmFile(PosixShm& shmFile, const size_t size, void*& addr,
                              bool needTrunc = true)
    {
        auto s = Status::OK();
        if (needTrunc) {
            s = shmFile.Truncate(size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to trunc file({}) with size({}).", s, shmFile.ShmName(), size);
                return s;
            }
        }
        s = shmFile.MMap(addr, size, true, true, true);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to mmap file({}) with size({}).", s, shmFile.ShmName(), size);
            return s;
        }
        return Status::OK();
    }
    static Status WaitShmHeaderReady(BufferHeader* header)
    {
        constexpr auto retryInterval = std::chrono::milliseconds(100);
        constexpr auto maxTryTime = 100;
        auto tryTime = 0;
        do {
            if (header->magic == sharedBufferMagic) { break; }
            if (tryTime > maxTryTime) { return Status::Retry(); }
            std::this_thread::sleep_for(retryInterval);
            tryTime++;
        } while (true);
        return Status::OK();
    }
    Status MapSegment(size_t idx, bool create)
    {
        auto& seg = segments_[idx];
        PosixShm shmFile{seg.shmName};
        const auto flags = create
            ? (PosixShm::OpenFlag::CREATE | PosixShm::OpenFlag::EXCL |
               PosixShm::OpenFlag::READ_WRITE)
            : PosixShm::OpenFlag::READ_WRITE;
        auto s = shmFile.ShmOpen(flags);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open segment file({}) with flags({}).", s, seg.shmName,
                     flags);
            return s;
        }
        s = MmapShmFile(shmFile, seg.mapSize, seg.address, create);
        if (s.Failure()) [[unlikely]] { return s; }
        seg.data = idx == 0 ? static_cast<std::byte*>(seg.address) + DataOffset()
                            : static_cast<std::byte*>(seg.address);
        if (idx == 0) { addrress_ = seg.address; }
        return Status::OK();
    }
    Status MapDataSegments(bool create)
    {
        const auto segmentCount = segments_.size();
        for (size_t idx = 1; idx < segments_.size(); idx++) {
            auto s = MapSegment(idx, create);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to map shared cache segment({}/{}) begin node({}) size({}).",
                         s, idx + 1, segmentCount, segments_[idx].nodeBegin,
                         segments_[idx].dataSize);
                if (create) { LogReduceCapacityHint(); }
                return s;
            }
            if (ShouldLogSegmentProgress(idx, segmentCount)) {
                UC_INFO("Shared cache segment mapped: index({}/{}), begin node({}), node "
                        "count({}), size({}).",
                        idx + 1, segmentCount, segments_[idx].nodeBegin,
                        segments_[idx].nodeCount, segments_[idx].dataSize);
            }
        }
        return Status::OK();
    }
    Status InitShmBuffer(PosixShm& shmFile)
    {
        auto& seg0 = segments_[0];
        auto s = MmapShmFile(shmFile, seg0.mapSize, seg0.address);
        if (s.Failure()) [[unlikely]] { return s; }
        seg0.data = static_cast<std::byte*>(seg0.address) + DataOffset();
        addrress_ = seg0.address;
        header_ = static_cast<BufferHeader*>(addrress_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        header_->magic = 0;
        header_->lock.Init();
        header_->nNode = nNode_;
        header_->freeHead = 0;
        header_->nodeSize = nodeSize_;
        header_->segmentNodeCount = segmentNodeCount_;
        header_->segmentCount = segmentCount_;
        UC_INFO("Shared cache segment mapped: index(1/{}), begin node({}), node count({}), "
                "size({}).",
                segmentCount_, seg0.nodeBegin, seg0.nodeCount, seg0.dataSize);
        for (size_t i = 0; i < nHashTableBucket; i++) {
            header_->buckets[i] = invalidIndex;
            header_->bucketLocks[i].Init();
        }
        for (size_t i = 0; i < nNode_; i++) {
            header_->nodeLocks[i].Init();
            meta_[i].Init();
        }
        s = MapDataSegments(true);
        if (s.Failure()) [[unlikely]] { return s; }
        // Shared-memory readiness is independent of per-device registration.
        header_->magic = sharedBufferMagic;
        return Status::OK();
    }
    Status LoadShmBuffer(PosixShm& shmFile)
    {
        auto s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmFile.ShmName());
            return s;
        }
        auto& seg0 = segments_[0];
        s = MmapShmFile(shmFile, seg0.mapSize, seg0.address, false);
        if (s.Failure()) [[unlikely]] { return s; }
        seg0.data = static_cast<std::byte*>(seg0.address) + DataOffset();
        addrress_ = seg0.address;
        header_ = static_cast<BufferHeader*>(addrress_);
        s = WaitShmHeaderReady(header_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Shm file({}) not ready.", shmFile.ShmName());
            return s;
        }
        if (header_->nNode != nNode_ || header_->nodeSize != nodeSize_ ||
            header_->segmentNodeCount != segmentNodeCount_ || header_->segmentCount == 0 ||
            header_->segmentCount > segmentCount_) {
            return Status::InvalidParam("mismatched shm cache segment layout");
        }
        segmentCount_ = header_->segmentCount;
        segments_.resize(segmentCount_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        return MapDataSegments(false);
    }
    Status RegisterBuffer(int32_t deviceId)
    {
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        const auto segmentCount = segments_.size();
        for (size_t idx = 0; idx < segmentCount; idx++) {
            auto& seg = segments_[idx];
            s = Trans::Buffer::RegisterHostBuffer((void*)seg.data, seg.dataSize,
                                                  (void**)&seg.dataOnDevice);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to register shared cache segment({}/{}) begin node({}) "
                         "size({}) to device({}).",
                         s, idx + 1, segmentCount, seg.nodeBegin, seg.dataSize, deviceId);
                LogReduceCapacityHint();
                return s;
            }
            seg.registered = true;
            if (ShouldLogSegmentProgress(idx, segmentCount)) {
                UC_INFO("Shared cache segment registered: index({}/{}), begin node({}), node "
                        "count({}), size({}), device({}).",
                        idx + 1, segmentCount, seg.nodeBegin, seg.nodeCount, seg.dataSize,
                        deviceId);
            }
        }
        return Status::OK();
    }

public:
    SharedBufferStrategy(const std::string& uuid, int32_t deviceId, size_t nodeSize,
                         size_t totalSize, size_t reservedNumber, size_t segmentSize)
        : BufferStrategy(deviceId, nodeSize, totalSize, reservedNumber, segmentSize), uuid_(uuid)
    {
    }
    ~SharedBufferStrategy() override
    {
        for (auto& seg : segments_) {
            if (seg.registered) { Trans::Buffer::UnregisterHostBuffer(seg.data); }
            if (seg.address) { PosixShm::MUnmap(seg.address, seg.mapSize); }
            if (owner_) { PosixShm{seg.shmName}.ShmUnlink(); }
        }
    }
    Status Setup() override
    {
        const auto& uuid = uuid_;
        const auto deviceId = base_.deviceId;
        const auto nodeSize = base_.nodeSize;
        const auto totalSize = base_.totalSize;
        shmName_ = ShmPrefix() + uuid;
        nodeSize_ = nodeSize;
        nNode_ = totalSize / nodeSize;
        InitSegments();
        CleanUpShmFileExceptMe(shmName_);
        PosixShm shmFile{segments_[0].shmName};
        totalSize_ = segments_[0].mapSize;
        const auto flags =
            PosixShm::OpenFlag::CREATE | PosixShm::OpenFlag::EXCL | PosixShm::OpenFlag::READ_WRITE;
        auto s = shmFile.ShmOpen(flags);
        if (s.Success()) {
            owner_ = true;
            s = InitShmBuffer(shmFile);
        } else if (s == Status::DuplicateKey()) {
            s = LoadShmBuffer(shmFile);
        } else {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, segments_[0].shmName,
                     flags);
            return s;
        }
        if (s.Failure()) [[unlikely]] { return s; }
        return RegisterBuffer(deviceId);
    }
    void BucketLock(size_t iBucket) override { header_->bucketLocks[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return header_->bucketLocks[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { header_->bucketLocks[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { header_->nodeLocks[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { header_->nodeLocks[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_->buckets[iBucket]; }
    size_t FetchNode(bool allowReserved) override
    {
        const auto limit = header_->nNode - (allowReserved ? 0 : base_.reservedNumber);
        header_->lock.Lock();
        if (header_->freeHead >= limit) { header_->freeHead = 0; }
        const auto iNode = header_->freeHead++;
        header_->lock.Unlock();
        return iNode;
    }
    void* DataAt(size_t iNode) override
    {
        auto& seg = SegmentAt(iNode);
        return seg.data + nodeSize_ * (iNode - seg.nodeBegin);
    }
    void* DeviceDataAt(size_t iNode) override
    {
        auto& seg = SegmentAt(iNode);
        return seg.dataOnDevice + nodeSize_ * (iNode - seg.nodeBegin);
    }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_ + iNode; }
};

class SharedBufferWatcherStrategy : public SharedBufferStrategy {
public:
    explicit SharedBufferWatcherStrategy(const std::string& uuid)
        : SharedBufferStrategy(uuid, -1, 0, 0, 0, 0)
    {
    }
    Status Setup() override
    {
        shmName_ = ShmPrefix() + uuid_;
        CleanUpShmFileExceptMe(shmName_);
        PosixShm shmFile{SegmentShmName(0)};
        auto s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmFile.ShmName());
            return s;
        }
        void* addr = nullptr;
        auto size = sizeof(BufferHeader);
        s = MmapShmFile(shmFile, size, addr, false);
        if (s.Failure()) [[unlikely]] { return s; }
        auto header = static_cast<BufferHeader*>(addr);
        s = WaitShmHeaderReady(header);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Shm file({}) not ready.", shmFile.ShmName());
            return s;
        }
        nNode_ = header->nNode;
        nodeSize_ = header->nodeSize;
        segmentNodeCount_ = header->segmentNodeCount;
        segmentCount_ = header->segmentCount;
        shmFile.MUnmap(addr, size);
        totalSize_ = DataOffset();
        s = MmapShmFile(shmFile, totalSize_, addrress_, false);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        segments_.push_back(Segment{SegmentShmName(0), addrress_, nullptr, nullptr,
                                    totalSize_, 0, 0, 0, false});
        return Status::OK();
    }
    void* DataAt(size_t iNode) override { return nullptr; }
    void* DeviceDataAt(size_t iNode) override { return nullptr; }
};

Status TransBuffer::Setup(const Config& config)
{
    bypassHitOnLoad_ = config.cacheLoadBackendOnly;
    try {
        if (!config.shareBufferEnable) {
            strategy_ = std::make_shared<LocalBufferStrategy>(
                config.deviceId, config.shardSize, config.bufferCapacity,
                config.loadExclusiveBufferNumber, config.cacheSegmentSize, config.ioDirect,
                config.cacheSdmaDirect);
        } else if (config.deviceId >= 0) {
            strategy_ = std::make_shared<SharedBufferStrategy>(
                config.uniqueId, config.deviceId, config.shardSize, config.bufferCapacity,
                config.loadExclusiveBufferNumber, config.cacheSegmentSize);
        } else {
            strategy_ = std::make_shared<SharedBufferWatcherStrategy>(config.uniqueId);
        }
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed({}) to make buffer strategy", e.what()));
    }
    return strategy_->Setup();
}

TransBuffer::Handle TransBuffer::Get(const Detail::BlockId& blockId, size_t shardIdx,
                                     bool allowReserved, bool isLoad)
{
    auto iBucket = Hash(blockId, shardIdx);
    bool owner = false;
    strategy_->BucketLock(iBucket);
    auto iNode = FindAt(iBucket, blockId, shardIdx, owner);
    if (iNode != invalidIndex) {
        if (bypassHitOnLoad_ && isLoad && owner && Ready(iNode)) { MarkNotReady(iNode); }
        strategy_->BucketUnlock(iBucket);
        return Handle{this, iNode, owner};
    }
    iNode = Alloc(blockId, shardIdx, iBucket, allowReserved);
    strategy_->BucketUnlock(iBucket);
    return Handle(this, iNode, true);
}

bool TransBuffer::Exist(const Detail::BlockId& blockId, size_t shardIdx)
{
    auto iBucket = Hash(blockId, shardIdx);
    strategy_->BucketLock(iBucket);
    auto exist = ExistAt(iBucket, blockId, shardIdx);
    strategy_->BucketUnlock(iBucket);
    return exist;
}

bool TransBuffer::ExistAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx)
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->block == blockId && meta->shard == shardIdx) {
            strategy_->NodeUnlock(iNode);
            return true;
        }
        auto next = meta->next;
        strategy_->NodeUnlock(iNode);
        iNode = next;
    }
    return false;
}

size_t TransBuffer::FindAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx,
                           bool& owner)
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->block == blockId && meta->shard == shardIdx) {
            owner = meta->reference == 0;
            if (owner && meta->state == State::FAILED) {
                meta->state = State::LOADING;
                meta->errorCode = Status::OK().Underlying();
            }
            ++meta->reference;
            strategy_->NodeUnlock(iNode);
            break;
        }
        auto next = meta->next;
        strategy_->NodeUnlock(iNode);
        iNode = next;
    }
    return iNode;
}

size_t TransBuffer::Alloc(const Detail::BlockId& blockId, size_t shardIdx, size_t iBucket,
                          bool allowReserved)
{
    for (;;) {
        auto iNode = strategy_->FetchNode(allowReserved);
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->reference > 0) {
            strategy_->NodeUnlock(iNode);
            continue;
        }
        const auto oldBucket = meta->hash;
        if (oldBucket != iBucket) {
            if (oldBucket != invalidIndex) {
                if (!strategy_->BucketTryLock(oldBucket)) {
                    strategy_->NodeUnlock(iNode);
                    continue;
                }
                Remove(oldBucket, iNode);
                strategy_->BucketUnlock(oldBucket);
            }
            MoveTo(iBucket, iNode);
        }
        ++meta->reference;
        meta->block = blockId;
        meta->shard = shardIdx;
        meta->state = State::LOADING;
        meta->errorCode = Status::OK().Underlying();
        strategy_->NodeUnlock(iNode);
        return iNode;
    }
}

void TransBuffer::MoveTo(size_t iBucket, size_t iNode)
{
    auto meta = strategy_->MetaAt(iNode);
    auto& head = strategy_->FirstAt(iBucket);
    auto n = head;
    meta->next = n;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = iNode;
        strategy_->NodeUnlock(n);
    }
    meta->hash = iBucket;
    head = iNode;
}

void TransBuffer::Remove(size_t iBucket, size_t iNode)
{
    auto meta = strategy_->MetaAt(iNode);
    auto p = meta->prev;
    if (p != invalidIndex) {
        auto prev = strategy_->MetaAt(p);
        strategy_->NodeLock(p);
        prev->next = meta->next;
        strategy_->NodeUnlock(p);
    }
    auto n = meta->next;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = meta->prev;
        strategy_->NodeUnlock(n);
    }
    if (strategy_->FirstAt(iBucket) == iNode) { strategy_->FirstAt(iBucket) = n; }
    meta->prev = meta->next = invalidIndex;
    meta->hash = invalidIndex;
}

void* TransBuffer::DataAt(Index pos) { return strategy_->DataAt(pos); }

void* TransBuffer::DeviceDataAt(Index pos) { return strategy_->DeviceDataAt(pos); }

void TransBuffer::Acquire(Index pos)
{
    strategy_->NodeLock(pos);
    ++strategy_->MetaAt(pos)->reference;
    strategy_->NodeUnlock(pos);
}

void TransBuffer::Release(Index pos)
{
    strategy_->NodeLock(pos);
    --strategy_->MetaAt(pos)->reference;
    strategy_->NodeUnlock(pos);
}

bool TransBuffer::Ready(Index pos) { return GetState(pos) == State::READY; }

TransBuffer::State TransBuffer::GetState(Index pos)
{
    strategy_->NodeLock(pos);
    auto state = strategy_->MetaAt(pos)->state;
    strategy_->NodeUnlock(pos);
    return state;
}

Status TransBuffer::FailureStatus(Index pos)
{
    strategy_->NodeLock(pos);
    auto errorCode = strategy_->MetaAt(pos)->errorCode;
    strategy_->NodeUnlock(pos);
    if (errorCode == Status::OK().Underlying()) {
        return Status::Error("shared buffer failed without an error status");
    }
    return Status{errorCode, {}};
}

void TransBuffer::MarkReady(Index pos)
{
    strategy_->NodeLock(pos);
    auto meta = strategy_->MetaAt(pos);
    if (meta->state == State::LOADING) {
        meta->state = State::READY;
        meta->errorCode = Status::OK().Underlying();
    }
    strategy_->NodeUnlock(pos);
}

void TransBuffer::MarkFailed(Index pos, const Status& status)
{
    strategy_->NodeLock(pos);
    auto meta = strategy_->MetaAt(pos);
    if (meta->state == State::LOADING) {
        meta->state = State::FAILED;
        meta->errorCode = status.Underlying();
    }
    strategy_->NodeUnlock(pos);
}

void TransBuffer::MarkNotReady(Index pos)
{
    strategy_->NodeLock(pos);
    auto meta = strategy_->MetaAt(pos);
    meta->state = State::LOADING;
    meta->errorCode = Status::OK().Underlying();
    strategy_->NodeUnlock(pos);
}

}  // namespace UC::CacheStore
