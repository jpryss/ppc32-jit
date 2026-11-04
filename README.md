# ppc32-jit

A just-in-time translator from 32-bit PowerPC user-level code to x86-64, in Rust, with no
dependencies beyond `libc`. It is the CPU of [aixemu](https://github.com/jpryss/aixemu), a
user-mode emulator for AIX programs, where it replaced Unicorn (QEMU's TCG as a library) and
made compilers run 25 times faster. It is published separately because nothing in it is
AIX-specific: anything that runs 32-bit PowerPC user code - an old Linux/ppc, Mac OS X or
BeOS binary, firmware extracted from a device, a game console's CPU - can use it.

What it is not: a system emulator. There is no MMU, no privileged state, no interrupts,
no 64-bit mode, no Altivec. You supply the memory map, the registers and a system-call
hook; it runs the code.

## Using it

```rust
use ppc32_jit::{Cpu, Exception, Reg};

let mut cpu = Cpu::new()?;                 // one 4 GB guest address space, empty
cpu.map(0x1000_0000, 0x10000)?;            // text and data: zeroed, RWX
cpu.map(0x7FFF_0000, 0x10000)?;            // a stack
cpu.write(0x1000_0000, &program_bytes)?;   // big-endian PowerPC code
cpu.set_reg(Reg::Gpr(1), 0x7FFF_FFF0);     // r1 = stack pointer
cpu.on_exception(|cpu, e| match e {
    Exception::Syscall => {                // `sc`: r0 is the number, r3.. the arguments
        let n = cpu.reg(Reg::Gpr(0));
        /* ... do it, put the result in r3 ... */
        if n == 1 { cpu.stop(); }          // exit
    }
    other => { eprintln!("{:?} at {:08x}", other, cpu.reg(Reg::Pc)); cpu.stop(); }
});
cpu.run(0x1000_0000, None)?;               // Ok when a hook stopped it, Err(Fault) on a fault
```

The API in one paragraph: `map`/`unmap` build the address space (page-aligned, zeroed,
readable, writable and executable - the guest has no page protection); `map_host` registers
memory you mapped yourself with MAP_FIXED at `host_addr(guest_addr)`, which is how a file or
a System V shared-memory segment becomes guest memory shared with other processes;
`read`/`write` access guest memory (writes invalidate overwritten translations); `reg`/
`set_reg` access the registers (`Reg::Gpr`, `Fpr`, `Pc`, `Lr`, `Ctr`, `Xer`, `Cr`, `Fpscr`,
`Msr`); `run(begin, until)` runs until the PC reaches `until`, a hook calls `stop()`, or a
fault; `on_exception` gets `sc`, traps, illegal instructions and alignment exceptions;
`add_code_hook(lo, hi, f)` runs before any instruction in the range (free elsewhere: it is
resolved at translation time) and may change registers, move the PC or stop;
`add_write_hook` is a watchpoint; `set_slice_hook` runs every 2,048 translated blocks (or at
once after the async-signal-safe `interrupt()`) and can stop the CPU - this is where an
emulator notices a host signal or pre-empts a green thread. `run` is re-entrant: a hook may
run a guest function. `cargo doc` has the details.

Integer and floating-point semantics follow the PowerPC architecture; where it leaves a
result undefined, QEMU's choice is kept (`divw` by zero gives the dividend, `fctiw` of a NaN
or out-of-range value gives 0x80000000, ...), because that is what the programs this was
tested with ran on for months. `stfs` is the hardware's bit copy, not a rounding. A NaN
produced by an operation is PowerPC's positive default NaN. FPSCR's rounding mode is
honoured; its exception bits are approximate (folded in from the host's sticky flags when
`mffs` reads them) and FPRF is not computed.

## Design

**Memory.** The 4 GB guest address space is one reservation of host address space
(`PROT_NONE`, `MAP_NORESERVE`, a guard above). Guest address `a` is host address
`base + a`, so a guest load is one host load with a base register, and an access to an
unmapped page is a host SIGSEGV that the fault handler turns into `Fault::Read`/`Write`
with the exact guest PC (from a per-block table of host offsets). This is what user-mode
QEMU does and what Unicorn, being QEMU's *system* emulator, cannot: there every load and
store goes through a software TLB, which with byte-swapping and a dispatcher trip at
every indirect branch adds up to about 80 host cycles per guest instruction on compiler
code. Here it is about 4.

**Registers.** The CPU state is a struct in memory addressed by r15 in generated code (r14
is the guest memory base, r13 the lookup tables). Every guest register lives there; an
instruction loads its operands into eax/ecx/edx/ebx, computes and stores back. Flags are
never live across guest instructions. Simple and correct; a register allocator for the hot
GPRs is the obvious next step if more speed is ever needed.

**Blocks.** A block runs from its first instruction to the next branch, `sc`, trap, hooked
address or 200 instructions. Direct branches end in a `jmp rel32` patched to the target
block once that is translated, so hot paths never leave generated code; indirect branches
look the target up in a 65,536-entry jump cache inline. Every block end decrements a slice
budget; at zero the block exits and the slice hook runs. Hooks are only ever called from
the dispatcher, with the CPU state exact.

**Stores and code.** Translated code is not protected from the program's own stores. A
per-page byte marks pages with translated code (and watchpoints); every store tests it
(three instructions). A store into such a page performs the store, invalidates the blocks
it overlaps (incoming chained jumps are re-pointed to their stubs, the jump-cache entry
cleared) and, if the running block was among them, leaves to the dispatcher.

**Atomics.** `lwarx` records address and value; `stwcx.` does a `lock cmpxchg` on the
host memory with that value as the expected one - a real atomic between processes that
share the memory through `map_host`.

**Floating point.** SSE2 scalar doubles; `fmadd` & co. use FMA3 when the host has it
(`fnmadd`/`fnmsub` negate the result, not the product, so a zero result has PowerPC's sign;
under a directed rounding mode the negated-product forms are used, which round the right
way but can give an exact zero the other sign). `fctiw`, `fres`, `frsqrte`, `mffs`,
`mtfsf`, the string instructions and `stfs` denormals are small helpers.

**Hosts.** x86-64 Linux and FreeBSD (the fault handler reads the signal context; macOS
would need its own). `MOVBE` and `FMA` are used when present (`PPC32_JIT_NOMOVBE=1`,
`PPC32_JIT_NOFMA=1` turn them off). `PPC32_JIT_TRACE=1` logs translations and
invalidations; `Cpu::stats()` reports blocks, flushes and exit counts.

## Verification

`examples/fuzz.rs` is a differential test against Unicorn: random sequences of 2-7
instructions on random register and memory state run on both, every GPR, FPR, CR, XER, LR,
CTR and the data page compared (`cargo run --release --example fuzz -- 50000 SEED`; seven
seconds a run). A million sequences run clean but for the fnmadd/fnmsub zero sign above.
`cargo test` runs a short one, plus a self-modifying-code check. In aixemu the JIT runs
the Python 2.2 and Perl 5.8.2 test suites with the same results as Unicorn, compiles with
GCC 8 byte-identically, and runs Progress and Mimer database servers; see
[aixemu/docs/JIT.md](https://github.com/jpryss/aixemu/blob/main/docs/JIT.md).

Lessons from the fuzzer that may save you an evening: `setne dil` without a REX prefix is
`setne bh`; `rlwinm`'s target is the rA field; `stfs` does not round; Unicorn's
`reg_write(XER)` and `reg_write(FPSCR)` do not reach the fields its instructions use, and
its translations survive `mem_write` (`ctl_flush_tb`).

## Speed

On one core of a 2.7 GHz Xeon: a Progress 4GL interpreter loop at 1.6 billion PowerPC
instructions per second; GCC 8's `cc1` compiling a 2,000-line file in 0.8 s (-O0) and
4.2 s (-O2), against 21.8 s and 113 s on Unicorn 2.1.3; the Python and Perl test suites of
aixemu, which are system-call-bound, 7-10 times faster.

## Status and limits

- 32-bit user-level PowerPC (the 60x/G3/G4 integer and FP sets as compilers use them).
  No 64-bit instructions, no Altivec, no privileged state; those raise `Exception::Illegal`.
- One `Cpu` per process (the guest region is a process-wide reservation); dropping it and
  creating another reuses the region, which is what an emulator's `exec` wants. `fork` is
  fine: the child inherits region and code cache.
- The code cache is 384 MB of reserved address space, flushed when full.
- `run` has no instruction count or timeout; use the slice hook with a timer.
- Unaligned integer loads and stores work (as on the real CPUs); unaligned `lwarx`/
  `stwcx.` raise `Exception::Alignment`.

MIT licence. Written in a night with a fuzzer at hand; bug reports with a failing
`fuzz -x WORDS` sequence are the most useful kind.
