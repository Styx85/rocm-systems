// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/risc_v/hart_state.h"

namespace rocjitsu {
namespace risc_v {

static thread_local Memory *tl_memory = nullptr;

Memory *current_memory() { return tl_memory; }

void set_current_memory(Memory *m) { tl_memory = m; }

uint64_t HartState::read_csr(uint16_t addr) const {
  switch (addr) {
  case CSR_FFLAGS:
    return fcsr & 0x1f;
  case CSR_FRM:
    return fcsr >> 5;
  case CSR_FCSR:
    return fcsr;
  }
  auto it = csrs.find(addr);
  return (it != csrs.end()) ? it->second : 0;
}

void HartState::write_csr(uint16_t addr, uint64_t val) {
  switch (addr) {
  case CSR_FFLAGS:
    fcsr = (fcsr & 0xe0) | (val & 0x1f);
    break;
  case CSR_FRM:
    fcsr = (fcsr & 0x1f) | ((val & 7) << 5);
    break;
  case CSR_FCSR:
    fcsr = static_cast<uint8_t>(val);
    break;
  default:
    csrs[addr] = val;
  }
}

} // namespace risc_v
} // namespace rocjitsu
