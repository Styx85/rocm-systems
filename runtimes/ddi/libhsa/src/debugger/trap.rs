// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Runtime-owned second-level trap handlers matching debugger ABI version 11.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};

use rocddi::device::Device;
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocddi::session::{DebuggerRegistrationMode, Session};
use rocddi::topology::GpuInfo;

use crate::ffi::{ERROR, OUT_OF_RESOURCES, Status};
use crate::runtime::map_error;

/// Exact trap build target and target/revision fixup rules from `ROCr`.
pub(crate) fn trap_bytes(info: &GpuInfo) -> Option<&'static [u8]> {
    Some(match (info.gfx_major, info.gfx_minor, info.gfx_stepping) {
        (9, 4, _) => include_bytes!("traps/trap_942.bin"),
        (9, 5, _) => include_bytes!("traps/trap_950.bin"),
        (9, 0, 8 | 10) => include_bytes!("traps/trap_9.bin"),
        (10, 1, _) => include_bytes!("traps/trap_1010.bin"),
        (10, 3, _) => include_bytes!("traps/trap_10.bin"),
        (11, _, _) => include_bytes!("traps/trap_11.bin"),
        (12, 0, _) => include_bytes!("traps/trap_12.bin"),
        (12, 5, 0) if info.asic_revision == 0 => include_bytes!("traps/trap_1250_rev0.bin"),
        (12, 5, _) => include_bytes!("traps/trap_1250.bin"),
        _ => return None,
    })
}

/// Trap backing and exact GPU retained through checked bind/unbind operations.
pub(crate) struct GpuDebugger {
    /// Retains the exact activated VM and prevents session teardown while bound.
    device: Device,
    /// Executable system backing referenced by the native trap-handler register.
    allocation: Option<Arc<Allocation>>,
    /// True before binding starts, including after an ambiguous set failure.
    bound: bool,
    /// Ambiguous cache-command retirement prevents reclaiming its operand.
    publication_uncertain: bool,
}

impl GpuDebugger {
    /// Creates a passive owner that can be stored before installation begins.
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            allocation: None,
            bound: false,
            publication_uncertain: false,
        }
    }

    /// Publishes trap code before making its address reachable from KFD.
    fn install(&mut self, cancel: &AtomicBool) -> Result<(), Status> {
        let gpu = self.device.gpu().map_err(map_error)?;
        let code = trap_bytes(gpu.info()).ok_or(ERROR)?;
        let page = rocddi::memory::host::host_page_size().map_err(map_error)?;
        if page < 256 || !page.is_power_of_two() {
            return Err(ERROR);
        }
        let byte_length = u64::try_from(code.len()).map_err(|_| OUT_OF_RESOURCES)?;
        let size = byte_length.checked_add(page - 1).ok_or(OUT_OF_RESOURCES)? & !(page - 1);
        let allocation = self
            .device
            .allocate(
                MemoryKind::System,
                size,
                page,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .map_err(map_error)?;
        let info = allocation.info();
        let host = info.host_address.ok_or(ERROR)?;
        // SAFETY: The unpublished allocation owns the entire writable code range.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), host as *mut u8, code.len()) };
        fence(Ordering::SeqCst);
        self.allocation = Some(Arc::new(allocation));
        if let Err(failure) = gpu.invalidate_code_cache(info.device_address, byte_length, cancel) {
            self.publication_uncertain = failure.operands_may_be_live;
            return Err(map_error(failure.error));
        }
        self.bound = true;
        // SAFETY: This owner retains code through checked unbind; no TMA data is
        // needed by the exception-debugging trap ABI when sampling is disabled.
        unsafe { gpu.set_trap_handler(info.device_address, 0) }.map_err(map_error)
    }

    /// Unbinds before freeing code; a failed native operation retains this owner.
    pub(crate) fn shutdown(&mut self) -> Result<(), Status> {
        if self
            .allocation
            .as_ref()
            .is_some_and(|allocation| Arc::strong_count(allocation) != 1)
        {
            return Err(ERROR);
        }
        reset_binding(&mut self.bound, self.publication_uncertain, || {
            // SAFETY: Queue users are stopped by the caller; a successful reset
            // conclusively removes the native reference before backing is freed.
            unsafe { self.device.gpu().map_err(map_error)?.set_trap_handler(0, 0) }
                .map_err(map_error)
        })?;
        if let Some(allocation) = &mut self.allocation {
            Arc::get_mut(allocation)
                .ok_or(ERROR)?
                .free()
                .map_err(map_error)?;
        }
        self.allocation = None;
        Ok(())
    }
}

/// Retains the debugger trap while PC sampling temporarily replaces its binding.
pub(crate) struct TrapRestore {
    /// Prevents debugger teardown from unbinding/freeing a borrowed trap program.
    allocation: Arc<Allocation>,
}

impl TrapRestore {
    /// Resolves the saved trap in the exact GPU VM whose binding will be restored.
    pub(crate) fn address(&self, device: &Device) -> Result<u64, Status> {
        self.allocation.device_address(device).map_err(map_error)
    }
}

/// Retires a possibly accepted trap binding without losing retry/retention state.
fn reset_binding(
    bound: &mut bool,
    publication_uncertain: bool,
    reset: impl FnOnce() -> Result<(), Status>,
) -> Result<(), Status> {
    if publication_uncertain {
        return Err(ERROR);
    }
    if *bound {
        reset()?;
        *bound = false;
    }
    Ok(())
}

impl Drop for GpuDebugger {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            // Unwinding or failed shutdown must never recycle a native trap VA.
            // The allocation itself retains its exact device/driver dependency.
            std::mem::forget(allocation);
        }
    }
}

/// Debugger owner retained through partial initialization and active generations.
pub(crate) struct PendingDebugger {
    /// Each owner is inserted before any trap binding can become native-visible.
    gpus: Vec<GpuDebugger>,
}

impl PendingDebugger {
    /// Retains the existing trap binding before a sampling owner replaces it.
    pub(crate) fn restore_token(&self, device: &Device) -> Option<TrapRestore> {
        let owner = self.gpus.iter().find(|owner| {
            owner.device.endpoint().id == device.endpoint().id
                && owner.bound
                && !owner.publication_uncertain
        })?;
        Some(TrapRestore {
            allocation: owner.allocation.as_ref()?.clone(),
        })
    }

    /// Stages the stable rendezvous and TTMP setup before native activation.
    pub(crate) fn new(session: &mut Session) -> Result<Self, Status> {
        // SAFETY: The static record and breakpoint live until process exit. The
        // loader retains each linked object until its delete notification ends.
        #[cfg(target_os = "linux")]
        unsafe {
            session.request_debugger_registration(
                super::record_address(),
                true,
                DebuggerRegistrationMode::Prefer,
            )
        }
        .map_err(map_error)?;
        #[cfg(not(target_os = "linux"))]
        let _ = session;
        Ok(Self { gpus: Vec::new() })
    }

    /// Retains a GPU's trap owner before starting fallible native publication.
    pub(crate) fn install(&mut self, device: &Device, cancel: &AtomicBool) -> Result<(), Status> {
        // Another adapter may already own the shared primary context. Preserve
        // its registration and trap binding while keeping HSA execution usable.
        #[cfg(target_os = "linux")]
        if !device
            .gpu()
            .map_err(map_error)?
            .debugger_registration_matches(super::record_address(), true)
            .map_err(map_error)?
        {
            return Ok(());
        }
        self.gpus.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
        let index = self.gpus.len();
        self.gpus.push(GpuDebugger::new(device));
        self.gpus.get_mut(index).ok_or(ERROR)?.install(cancel)
    }

    /// Retires traps after all queue users stop and before native session teardown.
    /// An error keeps every unfinished owner available for retry or quarantine.
    pub(crate) fn shutdown(&mut self) -> Result<(), Status> {
        let mut first_error = None;
        self.gpus.retain_mut(|gpu| match gpu.shutdown() {
            Ok(()) => false,
            Err(status) => {
                first_error.get_or_insert(status);
                true
            }
        });
        first_error.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trap_fixups_follow_asic_revision_without_affecting_later_steppings() {
        let mut info = GpuInfo {
            gfx_major: 12,
            gfx_minor: 5,
            ..GpuInfo::default()
        };
        let rev0 = trap_bytes(&info);
        info.asic_revision = 1;
        let later = trap_bytes(&info);
        assert_ne!(rev0, later);
        info.gfx_stepping = 1;
        info.asic_revision = 0;
        assert_eq!(trap_bytes(&info), later);
        info.gfx_major = 13;
        assert!(trap_bytes(&info).is_none());
    }

    #[test]
    fn failed_unbind_retains_backing_obligation_and_uncertainty_prevents_reuse() {
        let mut bound = true;
        assert_eq!(
            reset_binding(&mut bound, false, || Err(OUT_OF_RESOURCES)),
            Err(OUT_OF_RESOURCES)
        );
        assert!(bound);
        let mut reset_called = false;
        assert_eq!(
            reset_binding(&mut bound, true, || {
                reset_called = true;
                Ok(())
            }),
            Err(ERROR)
        );
        assert!(!reset_called);
        assert!(bound);
        assert!(reset_binding(&mut bound, false, || Ok(())).is_ok());
        assert!(!bound);
        assert!(
            reset_binding(&mut bound, false, || {
                reset_called = true;
                Err(ERROR)
            })
            .is_ok()
        );
        assert!(!reset_called);
    }
}
