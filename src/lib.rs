//! ppc32-jit: a just-in-time translator from 32-bit PowerPC user-level code to x86-64.
//!
//! The guest's 4 GB address space is one flat reservation of host address space, so a guest
//! load is a host load with a base register and page protection is the host MMU's; blocks are
//! chained directly, indirect branches go through a jump cache, `lwarx`/`stwcx.` become a
//! host compare-and-swap, floating point runs on SSE2/FMA. The user supplies the memory map,
//! the registers and hooks for system calls (`sc`), exceptions and selected addresses, and
//! runs the CPU with [`Cpu::run`]. See the README for the design and its limits.
//!
//! The CPU state lives in a struct addressed by r15 in generated code, the guest memory
//! through r14, the lookup tables through r13. Whatever cannot be chained (a syscall, a hooked
//! address, a fault, the end of a time slice) leaves the generated code with an exit reason
//! that the dispatcher acts on.

#![allow(function_casts_as_integer)]
#![allow(dead_code)] // the state offsets are a complete table

mod translate;
pub mod x86;

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// A guest register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reg {
    Gpr(u32),
    Fpr(u32),
    Pc,
    Lr,
    Ctr,
    Xer,
    Cr,
    Fpscr,
    Msr,
}

/// What a memory or register operation can fail with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// bad alignment, size or register
    InvalidArgument,
    /// the range overlaps a mapping
    Overlap,
    /// the host would not give the memory
    OutOfMemory,
    /// the range is not (entirely) mapped
    Unmapped,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for Error {}

/// An event the guest raises, delivered to the exception hook (`Cpu::on_exception`) with the
/// CPU stopped: for a `Syscall` the PC is after the `sc`, for the others at the instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exception {
    /// `sc`
    Syscall,
    /// `tw`/`twi` with a true condition
    Trap,
    /// an instruction the translator does not know (the 32-bit user-level set is complete;
    /// 64-bit, Altivec and privileged instructions are not translated)
    Illegal,
    /// a misaligned `lwarx`/`stwcx.`
    Alignment,
}

/// Why [`Cpu::run`] returned an error. The PC register is at the faulting instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// a load from unmapped memory
    Read { addr: u32, pc: u32 },
    /// a store to unmapped memory
    Write { addr: u32, pc: u32 },
    /// the PC is not in mapped memory (or not word-aligned)
    Fetch { pc: u32 },
    /// a trap, illegal instruction or alignment exception that no hook handled (the hook
    /// neither moved the PC nor stopped the CPU)
    Unhandled { exception: Exception, pc: u32 },
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::Read { addr, pc } => write!(f, "invalid read at {:08x} (pc {:08x})", addr, pc),
            Fault::Write { addr, pc } => write!(f, "invalid write at {:08x} (pc {:08x})", addr, pc),
            Fault::Fetch { pc } => write!(f, "invalid fetch at {:08x}", pc),
            Fault::Unhandled { exception, pc } => write!(f, "unhandled {:?} at {:08x}", exception, pc),
        }
    }
}
impl std::error::Error for Fault {}

/// A mapped guest range, `[begin, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemRegion {
    pub begin: u32,
    pub end: u32,
}

pub const GUEST_SIZE: usize = 1 << 32;
/// Above the 4 GB: a wrapped address plus an access length lands here, not in host data.
const GUARD: usize = 1 << 20;
const CODE_SIZE: usize = 384 << 20;
const TABLES_SIZE: usize = 2 << 20;
/// Jump cache: 65536 entries of {guest pc, pad, host entry}.
pub const JC_ENTRIES: u32 = 65536;
pub const JC_OFF: i32 = 0;
/// Per-page flags (one byte per 4 KB page) after the jump cache.
pub const PF_OFF: i32 = 1 << 20;
pub const PF_CODE: u8 = 1;
pub const PF_WATCH: u8 = 2;
/// Blocks between two slice checks (host signals, thread pre-emption).
pub const SLICE: i32 = 2048;

// ---- exit reasons (eax when a block returns to the dispatcher)
pub const EXIT_CHAIN: u32 = 1; // a direct branch to a block not yet translated
pub const EXIT_INDIRECT: u32 = 2; // an indirect branch missed the jump cache
pub const EXIT_BUDGET: u32 = 3; // the slice is used up
pub const EXIT_SYSCALL: u32 = 4; // sc
pub const EXIT_HOOK: u32 = 5; // reached an address with code hooks
pub const EXIT_TRAP: u32 = 6; // a trap instruction with a true condition
pub const EXIT_FAULT: u32 = 7; // a memory fault (the signal handler sends us here)
pub const EXIT_STOP: u32 = 8; // a helper wants the dispatcher (self-modifying code)
pub const EXIT_ILLEGAL: u32 = 9; // an instruction the translator does not know
pub const EXIT_ALIGN: u32 = 10; // a misaligned lwarx/stwcx. (alignment exception)

/// The CPU state, addressed by the generated code through r15.
#[repr(C)]
pub struct State {
    pub gpr: [u32; 32],
    pub fpr: [u64; 32],
    pub pc: u32,
    pub lr: u32,
    pub ctr: u32,
    pub xer: u32,
    pub cr: u32,
    pub fpscr: u32,
    pub msr: u32,
    pub budget: i32,
    pub res_addr: u32,
    pub res_valid: u32,
    pub res_val: u32,
    pub pad0: u32,
    pub exit_site: u64,
    pub saved_rsp: u64,
    pub fault_addr: u64,
    pub fault_pc: u32,
    pub fault_write: u32,
    pub mxcsr: u32,
    pub pad1: u32,
    pub mxcsr_save: u32,
    pub pad2: u32,
    pub scratch: [u64; 4],
    pub tables: u64,
    pub membase: u64,
    pub cpu: u64,
}

pub const OFF_GPR: i32 = 0;
pub const OFF_FPR: i32 = 0x80;
pub const OFF_PC: i32 = 0x180;
pub const OFF_LR: i32 = 0x184;
pub const OFF_CTR: i32 = 0x188;
pub const OFF_XER: i32 = 0x18C;
pub const OFF_CR: i32 = 0x190;
pub const OFF_FPSCR: i32 = 0x194;
pub const OFF_MSR: i32 = 0x198;
pub const OFF_BUDGET: i32 = 0x19C;
pub const OFF_RES_ADDR: i32 = 0x1A0;
pub const OFF_RES_VALID: i32 = 0x1A4;
pub const OFF_RES_VAL: i32 = 0x1A8;
pub const OFF_EXIT_SITE: i32 = 0x1B0;
pub const OFF_SAVED_RSP: i32 = 0x1B8;
pub const OFF_FAULT_ADDR: i32 = 0x1C0;
pub const OFF_FAULT_PC: i32 = 0x1C8;
pub const OFF_FAULT_WRITE: i32 = 0x1CC;
pub const OFF_MXCSR: i32 = 0x1D0;
pub const OFF_MXCSR_SAVE: i32 = 0x1D8;
pub const OFF_SCRATCH: i32 = 0x1E0;
pub const OFF_TABLES: i32 = 0x200;
pub const OFF_MEMBASE: i32 = 0x208;
pub const OFF_CPU: i32 = 0x210;

pub fn gpr_off(r: u32) -> i32 {
    OFF_GPR + 4 * r as i32
}
pub fn fpr_off(r: u32) -> i32 {
    OFF_FPR + 8 * r as i32
}

/// A translated block.
pub struct Block {
    pub pc: u32,
    pub nohook: bool,
    pub guest_len: u32,
    pub host: usize,
    pub size: usize,
    /// (site, stub): direct-branch sites of this block and the stub each goes to while
    /// its target is not translated
    pub chains: Vec<(usize, usize)>,
    /// sites in other blocks patched to jump to this block, with their stubs
    pub incoming: Vec<(usize, usize)>,
    /// (host offset, guest pc) of every instruction that can fault
    pub pcmap: Vec<(u32, u32)>,
    pub valid: bool,
}

type CodeHookFn<'a> = Box<dyn FnMut(&mut Cpu<'a>, u32) + 'a>;
type BlockHookFn<'a> = Box<dyn FnMut(&mut Cpu<'a>, u32, u32) + 'a>;
type ExcHookFn<'a> = Box<dyn FnMut(&mut Cpu<'a>, Exception) + 'a>;
type MemHookFn<'a> = Box<dyn FnMut(&mut Cpu<'a>, u32, u32, u64) + 'a>;
type SliceFn<'a> = Box<dyn FnMut(u32) -> bool + 'a>;

struct CodeHook<'a> {
    begin: u32,
    end: u32,
    f: CodeHookFn<'a>,
}
struct MemHook<'a> {
    begin: u32,
    end: u32,
    f: MemHookFn<'a>,
}

pub struct Cpu<'a> {
    pub state: Box<State>,
    pub base: usize,
    tables: usize,
    code: usize,
    code_used: usize,
    enter: usize,
    epilogue: usize,
    fault_stub: usize,
    blocks: Vec<Block>,
    by_key: HashMap<u64, usize>,
    by_host: BTreeMap<usize, usize>,
    page_blocks: HashMap<u32, Vec<usize>>,
    /// [begin, end) guest ranges, sorted, disjoint
    regions: Vec<(u64, u64)>,
    /// ranges whose host memory belongs to the emulator's mmap layer (mem_map_ptr)
    hostbacked: Vec<(u64, u64)>,
    code_hooks: Vec<CodeHook<'a>>,
    exc_hooks: Vec<ExcHookFn<'a>>,
    block_hooks: Vec<BlockHookFn<'a>>,
    mem_hooks: Vec<MemHook<'a>>,
    stop: bool,
    running: u32,
    flushes: u64,
    pub translated: u64,
    exits: [u64; 12],
    slice_hook: Option<SliceFn<'a>>,
    pub features: x86::HostFeatures,
    pub trace: bool,
    _p: PhantomData<&'a ()>,
}

/// The one CPU of this process, for the fault handler and the helpers.
static CUR: AtomicPtr<Cpu<'static>> = AtomicPtr::new(std::ptr::null_mut());
static CUR_STATE: AtomicPtr<State> = AtomicPtr::new(std::ptr::null_mut());
/// The guest memory base, reserved once per process and reused by later CPUs.
static BASE: AtomicUsize = AtomicUsize::new(0);

fn cur() -> *mut Cpu<'static> {
    CUR.load(Ordering::Relaxed)
}

/// From a host signal handler (it is async-signal-safe): end the running slice at the next
/// block boundary, so the slice hook runs within microseconds.
pub fn interrupt() {
    let s = CUR_STATE.load(Ordering::Relaxed);
    if !s.is_null() {
        unsafe {
            std::ptr::write_volatile(&mut (*s).budget, 0);
        }
    }
}

unsafe fn reserve(at: *mut libc::c_void, size: usize) -> *mut libc::c_void {
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | if at.is_null() { 0 } else { libc::MAP_FIXED };
    libc::mmap(at, size, libc::PROT_NONE, flags, -1, 0)
}

impl<'a> Cpu<'a> {
    /// A CPU with an empty 4 GB address space. Only one may exist per process; creating a
    /// second one (after dropping the first, e.g. for exec) reuses and clears the guest region.
    pub fn new() -> Result<Self, String> {
        let mut base = BASE.load(Ordering::Relaxed);
        unsafe {
            if base == 0 {
                let p = reserve(std::ptr::null_mut(), GUEST_SIZE + GUARD);
                if p == libc::MAP_FAILED {
                    return Err("cannot reserve 4 GB of address space for the guest".into());
                }
                base = p as usize;
                BASE.store(base, Ordering::Relaxed);
            } else {
                // exec(): everything the old program had is thrown away
                if reserve(base as *mut _, GUEST_SIZE + GUARD) == libc::MAP_FAILED {
                    return Err("cannot reset the guest address space".into());
                }
            }
        }
        let tables = unsafe {
            let p = libc::mmap(std::ptr::null_mut(), TABLES_SIZE, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
            if p == libc::MAP_FAILED {
                return Err("cannot allocate the JIT tables".into());
            }
            p as usize
        };
        let code = unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                CODE_SIZE,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                return Err("cannot allocate the code cache (executable memory)".into());
            }
            p as usize
        };
        let mut st: Box<State> = Box::new(unsafe { std::mem::zeroed() });
        st.msr = 0x6000;
        st.tables = tables as u64;
        st.membase = base as u64;
        st.budget = SLICE;
        st.mxcsr = 0x1F80;
        st.scratch[3] = translate::CRTAB;
        let mut cpu = Cpu {
            state: st,
            base,
            tables,
            code,
            code_used: 0,
            enter: 0,
            epilogue: 0,
            fault_stub: 0,
            blocks: Vec::new(),
            by_key: HashMap::new(),
            by_host: BTreeMap::new(),
            page_blocks: HashMap::new(),
            regions: Vec::new(),
            hostbacked: Vec::new(),
            code_hooks: Vec::new(),
            exc_hooks: Vec::new(),
            block_hooks: Vec::new(),
            mem_hooks: Vec::new(),
            stop: false,
            running: 0,
            flushes: 0,
            translated: 0,
            exits: [0; 12],
            slice_hook: None,
            features: x86::host_features(),
            trace: std::env::var("PPC32_JIT_TRACE").is_ok(),
            _p: PhantomData,
        };
        if std::env::var("PPC32_JIT_NOMOVBE").is_ok() {
            cpu.features.movbe = false;
        }
        if std::env::var("PPC32_JIT_NOFMA").is_ok() {
            cpu.features.fma = false;
        }
        cpu.reset_tables();
        cpu.emit_trampolines();
        install_fault_handler();
        Ok(cpu)
    }

    /// Make this the process's CPU (the fault handler and helpers find it here).
    fn publish(&mut self) {
        let p = self as *mut Cpu<'a> as *mut Cpu<'static>;
        CUR.store(p, Ordering::Relaxed);
        CUR_STATE.store(&mut *self.state as *mut State, Ordering::Relaxed);
        self.state.cpu = p as u64;
    }

    fn reset_tables(&mut self) {
        unsafe {
            let jc = self.tables as *mut u32;
            for i in 0..JC_ENTRIES as usize {
                *jc.add(i * 4) = 0xFFFF_FFFF;
                *jc.add(i * 4 + 1) = 0;
                *(jc.add(i * 4 + 2) as *mut u64) = 0;
            }
            std::ptr::write_bytes((self.tables + PF_OFF as usize) as *mut u8, 0, 1 << 20);
        }
    }

    /// enter / epilogue / fault stub at the start of the code cache.
    fn emit_trampolines(&mut self) {
        use x86::*;
        let mut a = Asm::new();
        // enter(state: rdi, code: rsi) -> eax
        let enter = a.here();
        for r in [RBX, RBP, R12, R13, R14, R15] {
            a.push(r);
        }
        a.alu64_i(Alu::Sub, Rm::R(RSP), 8); // 16-byte alignment for helper calls
        a.mov64_rr(R15, RDI);
        a.mov64_rm(R14, Rm::M(mem(R15, OFF_MEMBASE)));
        a.mov64_rm(R13, Rm::M(mem(R15, OFF_TABLES)));
        a.mov64_mr(Rm::M(mem(R15, OFF_SAVED_RSP)), RSP);
        a.jmp_rm(Rm::R(RSI));
        let epilogue = a.here();
        a.alu64_i(Alu::Add, Rm::R(RSP), 8);
        for r in [R15, R14, R13, R12, RBP, RBX] {
            a.pop(r);
        }
        a.ret();
        let fault = a.here();
        a.mov_ri(RAX, EXIT_FAULT);
        a.jmp_back(epilogue);
        unsafe {
            std::ptr::copy_nonoverlapping(a.code.as_ptr(), self.code as *mut u8, a.code.len());
        }
        self.enter = self.code + enter;
        self.epilogue = self.code + epilogue;
        self.fault_stub = self.code + fault;
        self.code_used = (a.code.len() + 63) & !63;
    }

    pub fn epilogue_addr(&self) -> usize {
        self.epilogue
    }

    /// Throw every translation away (the code cache is full, or hooks changed).
    pub fn flush_code(&mut self) {
        self.blocks.clear();
        self.by_key.clear();
        self.by_host.clear();
        self.page_blocks.clear();
        self.reset_tables();
        self.emit_trampolines();
        self.flushes += 1;
        if self.trace {
            eprintln!("[jit] code cache flushed ({})", self.flushes);
        }
    }

    // ---- memory bookkeeping

    fn find_region(&self, a: u64) -> Option<usize> {
        // regions sorted by begin
        let i = self.regions.partition_point(|r| r.0 <= a);
        if i == 0 {
            return None;
        }
        let r = self.regions[i - 1];
        if a < r.1 { Some(i - 1) } else { None }
    }

    /// Is [a, a+n) mapped (possibly across adjacent regions)?
    pub fn mapped_range(&self, a: u64, n: u64) -> bool {
        if n == 0 {
            return true;
        }
        let end = match a.checked_add(n) {
            Some(e) if e <= GUEST_SIZE as u64 => e,
            _ => return false,
        };
        let mut p = a;
        while p < end {
            match self.find_region(p) {
                Some(i) => p = self.regions[i].1,
                None => return false,
            }
        }
        true
    }

    fn add_range(v: &mut Vec<(u64, u64)>, lo: u64, hi: u64) {
        v.push((lo, hi));
        v.sort_unstable();
    }

    fn remove_range(v: &mut Vec<(u64, u64)>, lo: u64, hi: u64) {
        let mut out = Vec::with_capacity(v.len() + 1);
        for &(b, e) in v.iter() {
            if e <= lo || b >= hi {
                out.push((b, e));
                continue;
            }
            if b < lo {
                out.push((b, lo));
            }
            if e > hi {
                out.push((hi, e));
            }
        }
        *v = out;
    }

    fn overlaps(v: &[(u64, u64)], lo: u64, hi: u64) -> bool {
        v.iter().any(|&(b, e)| b < hi && e > lo)
    }

    fn check_map_args(addr: u32, size: usize) -> Result<(u64, u64), Error> {
        let addr = addr as u64;
        if size == 0 || addr & 0xFFF != 0 || size & 0xFFF != 0 {
            return Err(Error::InvalidArgument);
        }
        let end = addr.checked_add(size as u64).ok_or(Error::InvalidArgument)?;
        if end > GUEST_SIZE as u64 {
            return Err(Error::InvalidArgument);
        }
        Ok((addr, end))
    }

    /// Map fresh zeroed memory at a page-aligned guest address (readable, writable,
    /// executable: the guest has no page protection).
    pub fn map(&mut self, addr: u32, size: usize) -> Result<(), Error> {
        let (lo, hi) = Self::check_map_args(addr, size)?;
        if Self::overlaps(&self.regions, lo, hi) {
            return Err(Error::Overlap);
        }
        unsafe {
            let p = libc::mmap(
                (self.base + lo as usize) as *mut libc::c_void,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                return Err(Error::OutOfMemory);
            }
        }
        Self::add_range(&mut self.regions, lo, hi);
        Ok(())
    }

    /// Register host memory the caller has mapped itself at `host_addr(addr)` with MAP_FIXED
    /// (a file, System V shared memory): the way to give the guest a mapping that is shared
    /// with other processes. `ptr` must be `host_addr(addr)`; the range replaces whatever
    /// records were there. Such memory is the caller's to release (`host_release`).
    pub unsafe fn map_host(&mut self, addr: u32, size: usize, ptr: *mut u8) -> Result<(), Error> {
        let (lo, hi) = Self::check_map_args(addr, size)?;
        if ptr as usize != self.base + lo as usize {
            return Err(Error::InvalidArgument);
        }
        // the host mapping has replaced whatever was there (MAP_FIXED): so does the record
        if Self::overlaps(&self.regions, lo, hi) {
            self.invalidate_range(lo as u32, (hi - lo) as u32);
            Self::remove_range(&mut self.regions, lo, hi);
            Self::remove_range(&mut self.hostbacked, lo, hi);
        }
        Self::add_range(&mut self.regions, lo, hi);
        Self::add_range(&mut self.hostbacked, lo, hi);
        Ok(())
    }

    /// Unmap a guest range. Pages from `map` go back to the reservation; memory registered
    /// with `map_host` stays mapped until the caller releases it (`host_release`), so that
    /// the caller can still write it back.
    pub fn unmap(&mut self, addr: u32, size: usize) -> Result<(), Error> {
        let (lo, hi) = Self::check_map_args(addr, size)?;
        if !self.mapped_range(lo, hi - lo) {
            return Err(Error::Unmapped);
        }
        self.invalidate_range(lo as u32, (hi - lo) as u32);
        // pieces that are not host-backed go back to PROT_NONE
        let mut own: Vec<(u64, u64)> = vec![(lo, hi)];
        for &(b, e) in &self.hostbacked {
            let mut next = Vec::new();
            for (ol, oh) in own {
                if e <= ol || b >= oh {
                    next.push((ol, oh));
                } else {
                    if b > ol {
                        next.push((ol, b));
                    }
                    if e < oh {
                        next.push((e, oh));
                    }
                }
            }
            own = next;
        }
        for (ol, oh) in own {
            unsafe {
                reserve((self.base + ol as usize) as *mut _, (oh - ol) as usize);
            }
        }
        Self::remove_range(&mut self.regions, lo, hi);
        Self::remove_range(&mut self.hostbacked, lo, hi);
        Ok(())
    }

    /// The host address of a guest address (where to `mmap` a file for `map_host`; also a
    /// direct pointer into mapped guest memory).
    pub fn host_addr(&self, a: u32) -> *mut u8 {
        (self.base + a as usize) as *mut u8
    }

    /// Release host memory that was mapped into the guest region with `map_host` (it is
    /// re-reserved so the host never places its own allocations there); memory outside the
    /// region is simply unmapped.
    pub fn host_release(ptr: *mut u8, len: usize) {
        let base = BASE.load(Ordering::Relaxed);
        let p = ptr as usize;
        if base != 0 && p >= base && p + len <= base + GUEST_SIZE {
            unsafe {
                reserve(ptr as *mut _, len);
            }
        } else {
            unsafe {
                libc::munmap(ptr as *mut _, len);
            }
        }
    }

    /// The mapped ranges, in address order.
    pub fn regions(&self) -> Vec<MemRegion> {
        self.regions.iter().map(|&(b, e)| MemRegion { begin: b as u32, end: e as u32 }).collect()
    }

    /// Is every byte of `[a, a+n)` mapped?
    pub fn is_mapped(&self, a: u32, n: usize) -> bool {
        self.mapped_range(a as u64, n as u64)
    }

    /// Read guest memory (the whole range must be mapped).
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Error> {
        let addr = addr as u64;
        if !self.mapped_range(addr, buf.len() as u64) {
            return Err(Error::Unmapped);
        }
        unsafe {
            std::ptr::copy_nonoverlapping((self.base + addr as usize) as *const u8, buf.as_mut_ptr(), buf.len());
        }
        Ok(())
    }

    pub fn read_vec(&self, addr: u32, n: usize) -> Result<Vec<u8>, Error> {
        let mut v = vec![0u8; n];
        self.read(addr, &mut v)?;
        Ok(v)
    }

    /// Write guest memory; translations of overwritten code are discarded.
    pub fn write(&mut self, addr: u32, data: &[u8]) -> Result<(), Error> {
        let addr = addr as u64;
        if !self.mapped_range(addr, data.len() as u64) {
            return Err(Error::Unmapped);
        }
        self.invalidate_range(addr as u32, data.len() as u32);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), (self.base + addr as usize) as *mut u8, data.len());
        }
        Ok(())
    }

    // ---- registers

    /// Read a register. FPRs are the raw 64 bits; CR, XER, FPSCR are the full 32-bit
    /// registers (CR0 in the top nibble of CR).
    pub fn reg(&self, r: Reg) -> u64 {
        let s = &self.state;
        match r {
            Reg::Gpr(n) => s.gpr[(n & 31) as usize] as u64,
            Reg::Fpr(n) => s.fpr[(n & 31) as usize],
            Reg::Pc => s.pc as u64,
            Reg::Lr => s.lr as u64,
            Reg::Ctr => s.ctr as u64,
            Reg::Xer => s.xer as u64,
            Reg::Cr => s.cr as u64,
            Reg::Fpscr => s.fpscr as u64,
            Reg::Msr => s.msr as u64,
        }
    }

    /// Write a register (while the CPU is stopped, or from a hook).
    pub fn set_reg(&mut self, r: Reg, val: u64) {
        let s = &mut self.state;
        match r {
            Reg::Gpr(n) => s.gpr[(n & 31) as usize] = val as u32,
            Reg::Fpr(n) => s.fpr[(n & 31) as usize] = val,
            Reg::Pc => s.pc = val as u32,
            Reg::Lr => s.lr = val as u32,
            Reg::Ctr => s.ctr = val as u32,
            Reg::Xer => s.xer = val as u32,
            Reg::Cr => s.cr = val as u32,
            Reg::Fpscr => {
                s.fpscr = val as u32;
                translate::fpscr_written(s);
            }
            Reg::Msr => s.msr = val as u32,
        }
    }

    // ---- hooks

    /// Call `f(cpu, pc)` before every instruction with `begin <= pc <= end` executes; the hook
    /// may change registers and memory, move the PC (the instruction is then skipped) or stop
    /// the CPU. Resolved at translation time: it costs nothing elsewhere.
    pub fn add_code_hook<F: FnMut(&mut Cpu<'a>, u32) + 'a>(&mut self, begin: u32, end: u32, f: F) {
        self.code_hooks.push(CodeHook { begin, end, f: Box::new(f) });
        // blocks translated before this hook existed do not stop at it
        if !self.blocks.is_empty() {
            self.flush_code();
        }
    }

    /// Call `f(cpu, pc, length)` at the start of every translated block (a call out of
    /// generated code for each block: for counting and tracing, not for production).
    pub fn add_block_hook<F: FnMut(&mut Cpu<'a>, u32, u32) + 'a>(&mut self, f: F) {
        self.block_hooks.push(Box::new(f));
        if !self.blocks.is_empty() {
            self.flush_code();
        }
    }

    /// Call `f(cpu, exception)` for `sc`, traps, illegal instructions and alignment
    /// exceptions. For a syscall the PC is already after the `sc`; the hook typically
    /// reads r0-r10, writes r3 and leaves the PC alone, or sets it. For the other exceptions
    /// the hook must move the PC or stop the CPU, or `run` returns `Fault::Unhandled`.
    pub fn on_exception<F: FnMut(&mut Cpu<'a>, Exception) + 'a>(&mut self, f: F) {
        self.exc_hooks.push(Box::new(f));
    }

    /// Call `f(cpu, addr, size, value)` after every guest store into `[begin, end]` (a
    /// watchpoint; the pages take a slow path, nothing else does).
    pub fn add_write_hook<F: FnMut(&mut Cpu<'a>, u32, u32, u64) + 'a>(&mut self, begin: u32, end: u32, f: F) {
        self.mem_hooks.push(MemHook { begin, end, f: Box::new(f) });
        let mut p = (begin & !0xFFF) as u64;
        while p <= end as u64 {
            unsafe {
                *((self.tables + PF_OFF as usize + (p >> 12) as usize) as *mut u8) |= PF_WATCH;
            }
            p += 0x1000;
        }
    }

    /// Call `f(blocks)` whenever a slice of `blocks` translated blocks has run (every
    /// [`SLICE`] blocks, or sooner after [`interrupt`]); `true` stops the CPU. This is where
    /// a host signal is noticed or a green thread pre-empted.
    pub fn set_slice_hook<F: FnMut(u32) -> bool + 'a>(&mut self, f: F) {
        self.slice_hook = Some(Box::new(f));
    }

    /// Is there a code hook at this address?
    pub fn hooked(&self, a: u32) -> bool {
        self.code_hooks.iter().any(|h| h.begin <= a && a <= h.end)
    }
    pub fn has_block_hooks(&self) -> bool {
        !self.block_hooks.is_empty()
    }

    fn run_code_hooks(&mut self, pc: u32) {
        // the hooks are taken out while they run (a hook may add hooks or run the CPU)
        let mut hooks = std::mem::take(&mut self.code_hooks);
        for h in hooks.iter_mut() {
            if h.begin <= pc && pc <= h.end {
                (h.f)(self, pc);
            }
        }
        hooks.append(&mut self.code_hooks);
        self.code_hooks = hooks;
    }

    fn run_exc_hooks(&mut self, e: Exception) {
        let mut hooks = std::mem::take(&mut self.exc_hooks);
        for h in hooks.iter_mut() {
            h(self, e);
        }
        hooks.append(&mut self.exc_hooks);
        self.exc_hooks = hooks;
    }

    fn run_block_hooks(&mut self, pc: u32, size: u32) {
        let mut hooks = std::mem::take(&mut self.block_hooks);
        for h in hooks.iter_mut() {
            h(self, pc, size);
        }
        hooks.append(&mut self.block_hooks);
        self.block_hooks = hooks;
    }

    fn run_mem_hooks(&mut self, addr: u32, size: u32, value: u64) {
        let mut hooks = std::mem::take(&mut self.mem_hooks);
        let lo = addr as u64;
        let hi = lo + size as u64 - 1;
        for h in hooks.iter_mut() {
            if h.begin as u64 <= hi && lo <= h.end as u64 {
                (h.f)(self, addr, size, value);
            }
        }
        hooks.append(&mut self.mem_hooks);
        self.mem_hooks = hooks;
    }

    /// From a hook: return from `run` once the hook is done.
    pub fn stop(&mut self) {
        self.stop = true;
    }

    // ---- the block cache

    fn get_block(&mut self, pc: u32, nohook: bool) -> Result<usize, Fault> {
        let key = pc as u64 | ((nohook as u64) << 32);
        if let Some(&i) = self.by_key.get(&key) {
            return Ok(self.blocks[i].host);
        }
        if !self.mapped_range(pc as u64, 4) || pc & 3 != 0 {
            return Err(Fault::Fetch { pc });
        }
        let mut dest = (self.code + self.code_used + 15) & !15;
        let mut t = translate::translate(self, pc, nohook, dest)?;
        if dest + t.code.len() > self.code + CODE_SIZE {
            self.flush_code();
            dest = (self.code + self.code_used + 15) & !15;
            t = translate::translate(self, pc, nohook, dest)?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(t.code.as_ptr(), dest as *mut u8, t.code.len());
        }
        self.code_used = dest + t.code.len() - self.code;
        self.translated += 1;
        let idx = self.blocks.len();
        let blk = Block {
            pc,
            nohook,
            guest_len: t.guest_len,
            host: dest,
            size: t.code.len(),
            chains: t.chains.iter().map(|&(s, st)| (dest + s as usize, dest + st as usize)).collect(),
            incoming: Vec::new(),
            pcmap: t.pcmap,
            valid: true,
        };
        self.blocks.push(blk);
        self.by_key.insert(key, idx);
        self.by_host.insert(dest, idx);
        let mut p = pc & !0xFFF;
        let end = pc + t.guest_len;
        while p < end {
            self.page_blocks.entry(p).or_default().push(idx);
            unsafe {
                *((self.tables + PF_OFF as usize + (p >> 12) as usize) as *mut u8) |= PF_CODE;
            }
            p += 0x1000;
        }
        if self.trace {
            eprintln!("[jit] block {:08x}+{:#x}{} -> {:#x} ({} bytes)", pc, t.guest_len, if nohook { " (nohook)" } else { "" }, dest, t.code.len());
        }
        Ok(dest)
    }

    /// Throw away the blocks that contain any byte of [a, a+n).
    pub fn invalidate_range(&mut self, a: u32, n: u32) {
        if n == 0 {
            return;
        }
        let end = a as u64 + n as u64;
        let mut p = (a & !0xFFF) as u64;
        while p < end {
            let flag = unsafe { *((self.tables + PF_OFF as usize + (p >> 12) as usize) as *const u8) };
            if flag & PF_CODE != 0 {
                let list = self.page_blocks.get(&(p as u32)).cloned().unwrap_or_default();
                for i in list {
                    let b = &self.blocks[i];
                    if b.valid && (b.pc as u64) < end && b.pc as u64 + b.guest_len as u64 > a as u64 {
                        self.invalidate_block(i);
                    }
                }
            }
            p += 0x1000;
        }
    }

    fn invalidate_block(&mut self, i: usize) {
        let (pc, nohook, host, incoming) = {
            let b = &mut self.blocks[i];
            b.valid = false;
            (b.pc, b.nohook, b.host, std::mem::take(&mut b.incoming))
        };
        self.by_key.remove(&(pc as u64 | ((nohook as u64) << 32)));
        self.by_host.remove(&host);
        for (site, stub) in incoming {
            patch_rel32(site, stub);
        }
        if !nohook {
            let e = self.jc_entry(pc);
            unsafe {
                if *(e as *const u32) == pc {
                    *(e as *mut u32) = 0xFFFF_FFFF;
                }
            }
        }
        // pages left without code lose their flag (stores to them take the fast path again)
        let (bpc, blen) = (self.blocks[i].pc, self.blocks[i].guest_len);
        let mut p = bpc & !0xFFF;
        while p < bpc.wrapping_add(blen) {
            let live = self.page_blocks.get(&p).map_or(false, |l| l.iter().any(|&k| self.blocks[k].valid));
            if !live {
                self.page_blocks.remove(&p);
                unsafe {
                    *((self.tables + PF_OFF as usize + (p >> 12) as usize) as *mut u8) &= !PF_CODE;
                }
            }
            p += 0x1000;
        }
        if self.trace {
            eprintln!("[jit] invalidated block {:08x}", pc);
        }
    }

    pub fn page_blocks_at(&self, page: u32) -> Option<&Vec<usize>> {
        self.page_blocks.get(&page)
    }
    pub fn block(&self, i: usize) -> Option<&Block> {
        self.blocks.get(i)
    }

    fn jc_entry(&self, pc: u32) -> usize {
        self.tables + JC_OFF as usize + (((pc >> 2) & (JC_ENTRIES - 1)) as usize) * 16
    }

    /// The guest pc of a host address inside a translated block.
    fn guest_pc_of(&self, rip: usize) -> Option<u32> {
        let (&start, &i) = self.by_host.range(..=rip).next_back()?;
        let b = &self.blocks[i];
        if rip >= start + b.size {
            return None;
        }
        let off = (rip - start) as u32;
        let k = b.pcmap.partition_point(|&(o, _)| o <= off);
        if k == 0 {
            return Some(b.pc);
        }
        Some(b.pcmap[k - 1].1)
    }

    // ---- running

    /// Run from `begin` until the PC reaches `until`, a hook stops the CPU (`Ok`), or a
    /// fault (`Err`, PC at the faulting instruction). Re-entrant: a hook may call `run` to
    /// execute a guest function (save and restore the registers it needs around the call).
    pub fn run(&mut self, begin: u32, until: Option<u32>) -> Result<(), Fault> {
        let until = until.map_or(u64::MAX, |u| u as u64);
        let saved_stop = self.stop;
        self.stop = false;
        self.publish();
        self.state.pc = begin;
        self.state.budget = SLICE;
        let enter: extern "C" fn(*mut State, usize) -> u32 = unsafe { std::mem::transmute(self.enter) };
        let mut skip_hook = false;
        let r = loop {
            if self.stop {
                break Ok(());
            }
            let pc = self.state.pc;
            if pc as u64 == until {
                break Ok(());
            }
            let host = match self.get_block(pc, skip_hook) {
                Ok(h) => h,
                Err(e) => break Err(e),
            };
            skip_hook = false;
            let flushes = self.flushes;
            self.running += 1;
            let reason = enter(&mut *self.state as *mut State, host);
            self.running -= 1;
            self.exits[(reason as usize).min(11)] += 1;
            match reason {
                EXIT_CHAIN => {
                    let site = self.state.exit_site as usize;
                    let target = self.state.pc;
                    match self.get_block(target, false) {
                        Ok(h) => {
                            if self.flushes == flushes {
                                patch_rel32(site, h);
                                // remember the site so that invalidating the target unlinks it
                                if let Some(&bi) = self.by_host.get(&h) {
                                    let stub = self.stub_of(site);
                                    self.blocks[bi].incoming.push((site, stub));
                                }
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
                EXIT_INDIRECT => {
                    let target = self.state.pc;
                    match self.get_block(target, false) {
                        Ok(h) => {
                            let e = self.jc_entry(target);
                            unsafe {
                                *(e as *mut u32) = target;
                                *((e + 8) as *mut u64) = h as u64;
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
                EXIT_BUDGET => {
                    self.state.budget = SLICE;
                    if let Some(mut f) = self.slice_hook.take() {
                        let stop = f(SLICE as u32);
                        self.slice_hook = Some(f);
                        if stop {
                            break Ok(());
                        }
                    }
                }
                EXIT_SYSCALL => {
                    self.run_exc_hooks(Exception::Syscall);
                }
                EXIT_HOOK => {
                    let hpc = self.state.pc;
                    self.run_code_hooks(hpc);
                    if !self.stop && self.state.pc == hpc {
                        skip_hook = true;
                    }
                }
                EXIT_TRAP | EXIT_ILLEGAL | EXIT_ALIGN => {
                    let tpc = self.state.pc;
                    if tpc as u64 == until {
                        break Ok(());
                    }
                    if reason == EXIT_ILLEGAL && self.trace {
                        let w = self.read_vec(tpc, 4).unwrap_or_default();
                        eprintln!("[jit] illegal instruction {:02x?} at {:08x}", w, tpc);
                    }
                    let e = match reason {
                        EXIT_TRAP => Exception::Trap,
                        EXIT_ILLEGAL => Exception::Illegal,
                        _ => Exception::Alignment,
                    };
                    self.run_exc_hooks(e);
                    if !self.stop && self.state.pc == tpc {
                        // nobody handled it: the program would loop on the trap
                        break Err(Fault::Unhandled { exception: e, pc: tpc });
                    }
                }
                EXIT_FAULT => {
                    let pc = self.state.fault_pc;
                    let addr = self.state.fault_addr as u32;
                    self.state.pc = pc;
                    break Err(if self.state.fault_write != 0 { Fault::Write { addr, pc } } else { Fault::Read { addr, pc } });
                }
                EXIT_STOP => {}
                other => {
                    panic!("ppc32-jit: bad exit reason {}", other);
                }
            }
        };
        self.stop = saved_stop;
        r
    }

    fn stub_of(&self, site: usize) -> usize {
        if let Some((_, &i)) = self.by_host.range(..=site).next_back() {
            for &(s, st) in &self.blocks[i].chains {
                if s == site {
                    return st;
                }
            }
        }
        site // should not happen; patching a site to itself would loop, so leave the target
    }

    /// (guest pc, host code) of every live block, for debugging and the fuzzer.
    pub fn dump_blocks(&self) -> Vec<(u32, Vec<u8>)> {
        self.blocks.iter().filter(|b| b.valid).map(|b| (b.pc, unsafe { std::slice::from_raw_parts(b.host as *const u8, b.size).to_vec() })).collect()
    }

    pub fn stats(&self) -> String {
        let names = ["?", "chain", "indirect", "budget", "syscall", "hook", "trap", "fault", "stop", "illegal", "align", "?"];
        let ex: Vec<String> = (0..12).filter(|&i| self.exits[i] != 0).map(|i| format!("{} {}", names[i], self.exits[i])).collect();
        format!("{} blocks translated, {} cache flushes, {} KB of code; exits: {}", self.translated, self.flushes, self.code_used / 1024, ex.join(", "))
    }
}

impl<'a> Drop for Cpu<'a> {
    fn drop(&mut self) {
        // the guest region stays reserved for the next CPU (exec); the caches go
        unsafe {
            libc::munmap(self.code as *mut libc::c_void, CODE_SIZE);
            libc::munmap(self.tables as *mut libc::c_void, TABLES_SIZE);
        }
        let me = self as *mut Cpu<'a> as *mut Cpu<'static>;
        if CUR.load(Ordering::Relaxed) == me {
            CUR.store(std::ptr::null_mut(), Ordering::Relaxed);
            CUR_STATE.store(std::ptr::null_mut(), Ordering::Relaxed);
        }
    }
}

fn patch_rel32(site: usize, target: usize) {
    let rel = target as i64 - (site as i64 + 4);
    unsafe {
        std::ptr::write_unaligned(site as *mut i32, rel as i32);
    }
}

// ---- helpers called from generated code (extern "C", state pointer first)

fn page_flag(st: &State, a: u32) -> u8 {
    unsafe { *((st.tables as usize + PF_OFF as usize + (a >> 12) as usize) as *const u8) }
}

/// A store to a page with translated code or a watch: perform it, invalidate what it
/// overwrote, run the watch hooks. `size` is 1, 2, 4, 8 or 32 (dcbz), with 0x100 added for
/// a byte-reversed store (stwbrx). Returns 1 when the running block must be left (it
/// overwrote itself, or a hook stopped the CPU).
pub extern "C" fn h_store_slow(st: *mut State, ea: u32, val: u64, size: u32, next_pc: u32) -> u32 {
    unsafe {
        let st = &mut *st;
        let cpu = &mut *(st.cpu as *mut Cpu<'static>);
        let raw = size & 0x100 != 0;
        let size = size & 0xFF;
        let cur = next_pc.wrapping_sub(4);
        st.pc = cur;
        let p = (st.membase as usize + ea as usize) as *mut u8;
        match (size, raw) {
            (1, _) => *p = val as u8,
            (2, false) => std::ptr::write_unaligned(p as *mut u16, (val as u16).to_be()),
            (2, true) => std::ptr::write_unaligned(p as *mut u16, val as u16),
            (4, false) => std::ptr::write_unaligned(p as *mut u32, (val as u32).to_be()),
            (4, true) => std::ptr::write_unaligned(p as *mut u32, val as u32),
            (8, _) => std::ptr::write_unaligned(p as *mut u64, val.to_be()),
            (32, _) => std::ptr::write_bytes(p, 0, 32),
            _ => {}
        }
        let mut r = translate::post_store(st, ea, size, cur);
        let flag = page_flag(st, ea) | page_flag(st, ea.wrapping_add(size - 1));
        if flag & PF_WATCH != 0 {
            cpu.run_mem_hooks(ea, size, val);
            if cpu.stop {
                st.pc = next_pc;
                r = 1;
            }
        }
        r
    }
}

// ---- the fault handler

fn install_fault_handler() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        for sig in [libc::SIGSEGV, libc::SIGBUS] {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_fault as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
    });
}

#[cfg(target_os = "linux")]
unsafe fn ctx_regs(ctx: *mut libc::c_void) -> (*mut i64, *mut i64, u64) {
    let uc = &mut *(ctx as *mut libc::ucontext_t);
    let g = &mut uc.uc_mcontext.gregs;
    let err = g[libc::REG_ERR as usize] as u64;
    (&mut g[libc::REG_RIP as usize] as *mut i64, &mut g[libc::REG_RSP as usize] as *mut i64, err)
}

#[cfg(target_os = "freebsd")]
unsafe fn ctx_regs(ctx: *mut libc::c_void) -> (*mut i64, *mut i64, u64) {
    let uc = &mut *(ctx as *mut libc::ucontext_t);
    let mc = &mut uc.uc_mcontext;
    (&mut mc.mc_rip as *mut i64, &mut mc.mc_rsp as *mut i64, mc.mc_err as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
unsafe fn ctx_regs(_ctx: *mut libc::c_void) -> (*mut i64, *mut i64, u64) {
    (std::ptr::null_mut(), std::ptr::null_mut(), 0)
}

unsafe extern "C" fn on_fault(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let cpu = cur();
    let base = BASE.load(Ordering::Relaxed);
    let addr = (*info).si_addr() as usize;
    let (rip_p, rsp_p, err) = ctx_regs(ctx);
    if !cpu.is_null() && !rip_p.is_null() && base != 0 && addr >= base && addr < base + GUEST_SIZE + GUARD {
        let cpu = &mut *cpu;
        if cpu.running > 0 {
            let rip = *rip_p as usize;
            let gpc = cpu.guest_pc_of(rip);
            let st = &mut *cpu.state;
            st.fault_addr = (addr - base) as u64;
            st.fault_write = ((err >> 1) & 1) as u32;
            st.fault_pc = gpc.unwrap_or(st.pc);
            *rip_p = cpu.fault_stub as i64;
            *rsp_p = st.saved_rsp as i64;
            return;
        }
    }
    // not a guest fault: let the default action happen (the instruction re-faults)
    let mut sa: libc::sigaction = std::mem::zeroed();
    sa.sa_sigaction = libc::SIG_DFL;
    libc::sigemptyset(&mut sa.sa_mask);
    libc::sigaction(sig, &sa, std::ptr::null_mut());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_layout() {
        assert_eq!(std::mem::offset_of!(State, fpr), OFF_FPR as usize);
        assert_eq!(std::mem::offset_of!(State, pc), OFF_PC as usize);
        assert_eq!(std::mem::offset_of!(State, lr), OFF_LR as usize);
        assert_eq!(std::mem::offset_of!(State, ctr), OFF_CTR as usize);
        assert_eq!(std::mem::offset_of!(State, xer), OFF_XER as usize);
        assert_eq!(std::mem::offset_of!(State, cr), OFF_CR as usize);
        assert_eq!(std::mem::offset_of!(State, fpscr), OFF_FPSCR as usize);
        assert_eq!(std::mem::offset_of!(State, budget), OFF_BUDGET as usize);
        assert_eq!(std::mem::offset_of!(State, res_addr), OFF_RES_ADDR as usize);
        assert_eq!(std::mem::offset_of!(State, res_valid), OFF_RES_VALID as usize);
        assert_eq!(std::mem::offset_of!(State, res_val), OFF_RES_VAL as usize);
        assert_eq!(std::mem::offset_of!(State, exit_site), OFF_EXIT_SITE as usize);
        assert_eq!(std::mem::offset_of!(State, saved_rsp), OFF_SAVED_RSP as usize);
        assert_eq!(std::mem::offset_of!(State, fault_addr), OFF_FAULT_ADDR as usize);
        assert_eq!(std::mem::offset_of!(State, fault_pc), OFF_FAULT_PC as usize);
        assert_eq!(std::mem::offset_of!(State, fault_write), OFF_FAULT_WRITE as usize);
        assert_eq!(std::mem::offset_of!(State, mxcsr), OFF_MXCSR as usize);
        assert_eq!(std::mem::offset_of!(State, mxcsr_save), OFF_MXCSR_SAVE as usize);
        assert_eq!(std::mem::offset_of!(State, scratch), OFF_SCRATCH as usize);
        assert_eq!(std::mem::offset_of!(State, tables), OFF_TABLES as usize);
        assert_eq!(std::mem::offset_of!(State, membase), OFF_MEMBASE as usize);
        assert_eq!(std::mem::offset_of!(State, cpu), OFF_CPU as usize);
    }
}
