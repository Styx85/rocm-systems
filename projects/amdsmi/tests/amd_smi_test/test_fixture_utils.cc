// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_fixture_utils.h"

#include <gtest/gtest.h>
#include <linux/limits.h>
#include <sys/inotify.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <system_error>
#include <thread>

#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_main.h"

namespace fs = std::filesystem;

FakeSysfsTree::FakeSysfsTree() {
  std::string tmpl = "/tmp/amdsmi_fake_sysfs_XXXXXX";
  const char* dir = mkdtemp(tmpl.data());
  if (dir == nullptr) {
    ADD_FAILURE() << "FakeSysfsTree: mkdtemp() failed";
    return;
  }
  root_ = dir;
}

FakeSysfsTree::~FakeSysfsTree() {
  if (root_.empty()) return;
  std::error_code ec;
  fs::remove_all(root_, ec);
}

void FakeSysfsTree::WriteFile(const std::string& relative_path, const std::string& content) {
  if (root_.empty()) return;  // mkdtemp() already failed; ADD_FAILURE()'d, avoid touching cwd.
  fs::path full_path = fs::path(root_) / relative_path;
  std::error_code ec;
  fs::create_directories(full_path.parent_path(), ec);
  if (ec) {
    ADD_FAILURE() << "FakeSysfsTree::WriteFile(" << relative_path
                  << "): create_directories() failed: " << ec.message();
    return;
  }
  std::ofstream out(full_path);
  out << content;
  out.close();
  if (!out) {
    ADD_FAILURE() << "FakeSysfsTree::WriteFile(" << relative_path << "): write failed";
  }
}

void FakeSysfsTree::RemoveFile(const std::string& relative_path) {
  if (root_.empty()) return;
  std::error_code ec;
  fs::remove(fs::path(root_) / relative_path, ec);
}

void FakeSysfsTree::SetPermissions(const std::string& relative_path, mode_t mode) {
  if (root_.empty()) return;
  if (chmod((fs::path(root_) / relative_path).c_str(), mode) != 0) {
    ADD_FAILURE() << "FakeSysfsTree::SetPermissions(" << relative_path << ", " << mode
                  << "): chmod() failed: " << strerror(errno);
  }
}

namespace amd::smi::testing {

ScopedProcessorRegistration::ScopedProcessorRegistration(amd::smi::AMDSmiProcessor* processor)
    : processor_(processor) {
  amd::smi::AMDSmiSystem::getInstance().register_processor_for_testing(processor_);
}

ScopedProcessorRegistration::~ScopedProcessorRegistration() {
  amd::smi::AMDSmiSystem::getInstance().unregister_processor_for_testing(processor_);
}

amdsmi_processor_handle ScopedProcessorRegistration::handle() const {
  return reinterpret_cast<amdsmi_processor_handle>(processor_);
}

ScopedFileWatch::ScopedFileWatch(const std::string& path, uint32_t event_mask) {
  inotify_fd_ = inotify_init1(IN_NONBLOCK);
  if (inotify_fd_ < 0) {
    return;  // IsAvailable() reports false; nothing else to clean up.
  }
  watch_fd_ = inotify_add_watch(inotify_fd_, path.c_str(), event_mask);
  if (watch_fd_ < 0) {
    close(inotify_fd_);
    inotify_fd_ = -1;
    return;
  }
}

ScopedFileWatch::~ScopedFileWatch() {
  if (watch_fd_ >= 0) {
    inotify_rm_watch(inotify_fd_, watch_fd_);
  }
  if (inotify_fd_ >= 0) {
    close(inotify_fd_);
  }
}

void ScopedFileWatch::Drain() {
  if (inotify_fd_ < 0) return;
  // Per man 7 inotify: struct inotify_event has a variable-length trailing
  // name, so the buffer must hold several whole entries, properly aligned.
  constexpr std::size_t kMaxBufferedEvents = 16;
  constexpr std::size_t kBufferSize =
      kMaxBufferedEvents * (sizeof(struct inotify_event) + NAME_MAX + 1);
  alignas(struct inotify_event) std::array<char, kBufferSize> buf;
  for (;;) {
    ssize_t len = read(inotify_fd_, buf.data(), buf.size());
    if (len < 0 && errno == EINTR) {
      continue;
    }
    if (len <= 0) {
      // len < 0 with EAGAIN/EWOULDBLOCK just means no more events are
      // pending right now -- not an error worth distinguishing here.
      break;
    }
    for (char* ptr = buf.data(); ptr < buf.data() + len;) {
      auto* event = reinterpret_cast<struct inotify_event*>(ptr);
      for (std::size_t bit = 0; bit < kBitsPerMask; ++bit) {
        if (event->mask & (1u << bit)) ++event_tally_[bit];
      }
      ptr += sizeof(struct inotify_event) + event->len;
    }
  }
}

uint32_t ScopedFileWatch::Count(uint32_t mask) {
  Drain();
  uint32_t total = 0;
  for (std::size_t bit = 0; bit < kBitsPerMask; ++bit) {
    if (mask & (1u << bit)) total += event_tally_[bit];
  }
  return total;
}

uint32_t ScopedFileWatch::OpenCount() { return Count(IN_OPEN); }

uint32_t ScopedFileWatch::WriteCount() { return Count(IN_CLOSE_WRITE); }

}  // namespace amd::smi::testing

ScopedRocmSmiDevice::ScopedRocmSmiDevice(const std::string& path) {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  index_ = static_cast<uint32_t>(devices.size());
  devices.push_back(std::make_shared<amd::smi::Device>(path, nullptr));
}

ScopedRocmSmiDevice::~ScopedRocmSmiDevice() {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  if (index_ < devices.size()) devices.resize(index_);
}

ScopedKfdNode::ScopedKfdNode(uint64_t gpu_id, uint32_t dev_index) : gpu_id_(gpu_id) {
  auto& nodes = amd::smi::RocmSMI::getInstance().kfd_node_map();
  if (const auto it = nodes.find(gpu_id_); it != nodes.end()) previous_ = it->second;
  auto node = std::make_shared<amd::smi::KFDNode>(0);
  node->set_amdgpu_dev_index(dev_index);
  nodes[gpu_id_] = std::move(node);
}

ScopedKfdNode::~ScopedKfdNode() {
  auto& nodes = amd::smi::RocmSMI::getInstance().kfd_node_map();
  if (previous_) {
    nodes[gpu_id_] = previous_;
  } else {
    nodes.erase(gpu_id_);
  }
}

void WaitOutProcessListCache() {
  const int ms = read_env_ms("AMDSMI_PROCESS_INFO_CACHE_MS", 1);  // As the library reads it
  std::this_thread::sleep_for(std::chrono::milliseconds(ms + 1));
}

ScopedAmdsmiInit::ScopedAmdsmiInit() {
  EXPECT_EQ(amdsmi_init(kNoHardwareDiscoveryInitFlags), AMDSMI_STATUS_SUCCESS);
}

ScopedAmdsmiInit::~ScopedAmdsmiInit() { amdsmi_shut_down(); }
