//! PowerPC32 -> x86-64 translation of one block.
//!
//! Register conventions in generated code: r15 = `State`, r14 = guest memory base, r13 =
//! tables (jump cache, page flags). Guest registers live in the state; each instruction
//! loads what it needs into eax/ecx/edx/ebx, computes, and stores back. Flags are never
//! live across guest instructions. Helpers are plain `extern "C"` functions with the state
//! pointer as first argument.

#![allow(function_casts_as_integer)]
#![allow(dead_code)]

use super::x86::*;
use super::*;

/// The result of translating a block.
pub struct Translated {
    pub code: Vec<u8>,
    pub guest_len: u32,
    /// (offset of a rel32 chain site, offset of its stub)
    pub chains: Vec<(u32, u32)>,
    /// (host offset, guest pc) for instructions that may fault
    pub pcmap: Vec<(u32, u32)>,
}

const MAX_INSNS: u32 = 200;
pub const XER_SO: u32 = 0x8000_0000;
pub const XER_OV: u32 = 0x4000_0000;
pub const XER_CA: u32 = 0x2000_0000;
/// CR0 nibble table indexed by LT*2+EQ (in State.scratch[3]).
pub const OFF_CRTAB: i32 = OFF_SCRATCH + 24;
pub const CRTAB: u64 = 0x0808_0204;

enum Cold {
    /// slice over: pc = target, exit
    Budget { site: usize, target: u32 },
    /// slice over, target in eax
    BudgetInd { site: usize },
    /// direct branch to an untranslated block
    Chain { site: usize, target: u32 },
    /// jump cache miss, target in eax
    Miss { site: usize },
    /// a store to a page with code or a watch: ecx = ea, rax = value
    StoreSlow { site: usize, back: usize, size: u32, next_pc: u32 },
    /// a trap condition came true
    Trap { site: usize, pc: u32 },
    /// a misaligned lwarx/stwcx.
    Align { site: usize, pc: u32 },
    /// stfs of a value with a small exponent: rax = the double, eax <- its single image
    ToSingle { site: usize, back: usize },
}

struct Tr<'c, 'a> {
    cpu: &'c Cpu<'a>,
    a: Asm,
    dest: usize,
    pc: u32,
    start: u32,
    cold: Vec<Cold>,
    chains: Vec<(u32, u32)>,
    pcmap: Vec<(u32, u32)>,
    done: bool,
    movbe: bool,
    fma: bool,
}

fn gm(r: u32) -> Mem {
    mem(R15, gpr_off(r))
}
fn fm(r: u32) -> Mem {
    mem(R15, fpr_off(r))
}
fn sm(off: i32) -> Mem {
    mem(R15, off)
}
fn gaddr() -> Mem {
    mem_idx(R14, RCX, 1, 0)
}
fn gaddr_off(d: i32) -> Mem {
    mem_idx(R14, RCX, 1, d)
}

/// The rotate-and-mask of rlwinm etc.: bits mb..me (IBM numbering, 0 = MSB).
pub fn rot_mask(mb: u32, me: u32) -> u32 {
    let hi = 0xFFFF_FFFFu32 >> mb; // bits mb..31
    let lo = 0xFFFF_FFFFu32 << (31 - me); // bits 0..me
    if mb <= me { hi & lo } else { hi | lo }
}

impl<'c, 'a> Tr<'c, 'a> {
    fn insn(&self) -> Option<u32> {
        if !self.cpu.mapped_range(self.pc as u64, 4) {
            return None;
        }
        let p = (self.cpu.base + self.pc as usize) as *const u32;
        Some(u32::from_be(unsafe { std::ptr::read_unaligned(p) }))
    }

    fn mark(&mut self) {
        self.pcmap.push((self.a.here() as u32, self.pc));
    }

    fn ld(&mut self, reg: u8, r: u32) {
        self.a.mov_rm(reg, Rm::M(gm(r)));
    }
    fn st(&mut self, r: u32, reg: u8) {
        self.a.mov_mr(Rm::M(gm(r)), reg);
    }

    /// jmp to an absolute host address
    fn jmp_abs(&mut self, target: usize) {
        let at = self.a.jmp(0);
        let rel = target as i64 - (self.dest as i64 + at as i64 + 4);
        self.a.code[at..at + 4].copy_from_slice(&(rel as i32).to_le_bytes());
    }

    fn call_helper(&mut self, f: usize) {
        self.a.mov64_ri(RAX, f as u64);
        self.a.call_rm(Rm::R(RAX));
    }

    /// Leave the generated code: state.pc = pc, eax = reason.
    fn exit(&mut self, reason: u32, pc: u32) {
        self.a.mov_mi(sm(OFF_PC), pc);
        self.a.mov_ri(RAX, reason);
        let ep = self.cpu.epilogue_addr();
        self.jmp_abs(ep);
    }

    /// End of block: direct branch to `target`.
    fn end_direct(&mut self, target: u32) {
        self.a.alu_i(Alu::Sub, Rm::M(sm(OFF_BUDGET)), 1);
        let j = self.a.jcc(CC_LE, 0);
        self.cold.push(Cold::Budget { site: j, target });
        let s = self.a.jmp(0);
        self.cold.push(Cold::Chain { site: s, target });
    }

    /// End of block: indirect branch, target in eax.
    fn end_indirect(&mut self) {
        self.a.alu_i(Alu::Sub, Rm::M(sm(OFF_BUDGET)), 1);
        let j = self.a.jcc(CC_LE, 0);
        self.cold.push(Cold::BudgetInd { site: j });
        self.a.mov_rr(RDX, RAX);
        self.a.shift_i(Shift::Shr, Rm::R(RDX), 2);
        self.a.alu_i(Alu::And, Rm::R(RDX), JC_ENTRIES - 1);
        self.a.shift_i(Shift::Shl, Rm::R(RDX), 4);
        self.a.alu_rm(Alu::Cmp, RAX, Rm::M(mem_idx(R13, RDX, 1, JC_OFF)));
        let m = self.a.jcc(CC_NE, 0);
        self.cold.push(Cold::Miss { site: m });
        self.a.jmp_rm(Rm::M(mem_idx(R13, RDX, 1, JC_OFF + 8)));
    }

    fn emit_cold(&mut self) {
        let cold = std::mem::take(&mut self.cold);
        let ep = self.cpu.epilogue_addr();
        for c in cold {
            match c {
                Cold::Budget { site, target } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    self.exit(EXIT_BUDGET, target);
                }
                Cold::BudgetInd { site } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    self.a.mov_mr(Rm::M(sm(OFF_PC)), RAX);
                    self.a.mov_ri(RAX, EXIT_BUDGET);
                    self.jmp_abs(ep);
                }
                Cold::Chain { site, target } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    self.chains.push((site as u32, h as u32));
                    self.a.mov_mi(sm(OFF_PC), target);
                    self.a.mov64_ri(RAX, (self.dest + site) as u64);
                    self.a.mov64_mr(Rm::M(sm(OFF_EXIT_SITE)), RAX);
                    self.a.mov_ri(RAX, EXIT_CHAIN);
                    self.jmp_abs(ep);
                }
                Cold::Miss { site } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    self.a.mov_mr(Rm::M(sm(OFF_PC)), RAX);
                    self.a.mov_ri(RAX, EXIT_INDIRECT);
                    self.jmp_abs(ep);
                }
                Cold::StoreSlow { site, back, size, next_pc } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    // ecx (the address) is still needed by update forms after the store
                    self.a.mov64_mr(Rm::M(sm(OFF_SCRATCH)), RCX);
                    self.a.mov64_rr(RDI, R15);
                    self.a.mov_rr(RSI, RCX);
                    self.a.mov64_rr(RDX, RAX);
                    self.a.mov_ri(RCX, size);
                    self.a.mov_ri(R8, next_pc);
                    self.call_helper(h_store_slow as usize);
                    self.a.mov64_rm(RCX, Rm::M(sm(OFF_SCRATCH)));
                    self.a.test_rr(RAX, RAX);
                    let j = self.a.jcc(CC_E, 0);
                    self.a.patch(j, back);
                    self.a.mov_ri(RAX, EXIT_STOP);
                    self.jmp_abs(ep);
                }
                Cold::Trap { site, pc } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    self.exit(EXIT_TRAP, pc);
                }
                Cold::Align { site, pc } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    self.exit(EXIT_ALIGN, pc);
                }
                Cold::ToSingle { site, back } => {
                    let h = self.a.here();
                    self.a.patch(site, h);
                    // ecx (the address) must survive the call
                    self.a.mov64_mr(Rm::M(sm(OFF_SCRATCH)), RCX);
                    self.a.mov64_rr(RDI, RAX);
                    self.call_helper(h_tosingle as usize);
                    self.a.mov64_rm(RCX, Rm::M(sm(OFF_SCRATCH)));
                    let j = self.a.jmp(0);
                    self.a.patch(j, back);
                }
            }
        }
    }

    // ---- condition register

    /// CR0 <- compare eax with 0 (signed), plus SO.
    fn set_cr0(&mut self) {
        self.a.alu_rr(Alu::Xor, RCX, RCX);
        self.a.alu_rr(Alu::Xor, RDX, RDX);
        self.a.test_rr(RAX, RAX);
        self.a.setcc(CC_S, RCX);
        self.a.setcc(CC_E, RDX);
        self.nibble_from_lt_eq();
        self.write_cr_field(0);
    }

    /// ecx = LT, edx = EQ (0/1) -> ecx = CR nibble without SO.
    fn nibble_from_lt_eq(&mut self) {
        self.a.lea(RCX, mem_idx(RDX, RCX, 2, 0));
        self.a.movzx8(RCX, Rm::M(mem_idx(R15, RCX, 1, OFF_CRTAB)));
    }

    /// ecx = nibble (LT GT EQ): add SO and put it into CR field `crf`.
    fn write_cr_field(&mut self, crf: u32) {
        self.a.mov_rm(RDX, Rm::M(sm(OFF_XER)));
        self.a.shift_i(Shift::Shr, Rm::R(RDX), 31);
        self.a.alu_rr(Alu::Or, RCX, RDX);
        self.write_cr_nibble(crf);
    }

    /// ecx = complete nibble -> CR field crf.
    fn write_cr_nibble(&mut self, crf: u32) {
        let sh = 28 - 4 * crf as u8;
        if sh != 0 {
            self.a.shift_i(Shift::Shl, Rm::R(RCX), sh);
        }
        self.a.mov_rm(RDX, Rm::M(sm(OFF_CR)));
        self.a.alu_i(Alu::And, Rm::R(RDX), !(0xFu32 << sh));
        self.a.alu_rr(Alu::Or, RDX, RCX);
        self.a.mov_mr(Rm::M(sm(OFF_CR)), RDX);
    }

    /// Compare eax with `rhs` into CR field crf.
    fn compare(&mut self, rhs: Rm, signed: bool, crf: u32) {
        self.a.alu_rr(Alu::Xor, RCX, RCX);
        self.a.alu_rr(Alu::Xor, RDX, RDX);
        self.a.alu_rm(Alu::Cmp, RAX, rhs);
        self.a.setcc(if signed { CC_L } else { CC_B }, RCX);
        self.a.setcc(CC_E, RDX);
        self.nibble_from_lt_eq();
        self.write_cr_field(crf);
    }
    fn compare_imm(&mut self, imm: u32, signed: bool, crf: u32) {
        self.a.alu_rr(Alu::Xor, RCX, RCX);
        self.a.alu_rr(Alu::Xor, RDX, RDX);
        self.a.alu_i(Alu::Cmp, Rm::R(RAX), imm);
        self.a.setcc(if signed { CC_L } else { CC_B }, RCX);
        self.a.setcc(CC_E, RDX);
        self.nibble_from_lt_eq();
        self.write_cr_field(crf);
    }

    // ---- XER

    /// Put XER.CA into the host carry flag.
    fn ca_to_cf(&mut self) {
        self.a.bt_i(Rm::M(sm(OFF_XER)), 29);
    }

    /// XER update after an arithmetic op: edx = CA (0/1) when `ca`, ebx = OV (0/1) when `ov`.
    fn xer_update(&mut self, ca: bool, ov: bool) {
        if !ca && !ov {
            return;
        }
        self.a.mov_rm(RCX, Rm::M(sm(OFF_XER)));
        let mut clear = 0;
        if ca {
            clear |= XER_CA;
        }
        if ov {
            clear |= XER_OV;
        }
        self.a.alu_i(Alu::And, Rm::R(RCX), !clear);
        if ca {
            self.a.shift_i(Shift::Shl, Rm::R(RDX), 29);
            self.a.alu_rr(Alu::Or, RCX, RDX);
        }
        if ov {
            self.a.mov_rr(RDI, RBX);
            self.a.shift_i(Shift::Shl, Rm::R(RDI), 30);
            self.a.alu_rr(Alu::Or, RCX, RDI);
            self.a.shift_i(Shift::Shl, Rm::R(RBX), 31);
            self.a.alu_rr(Alu::Or, RCX, RBX);
        }
        self.a.mov_mr(Rm::M(sm(OFF_XER)), RCX);
    }

    /// Finish an arithmetic instruction: result in eax -> rt; flags still those of the op
    /// when `ca`/`ov` are to be recorded (dl/bl are set from them first).
    fn finish_arith(&mut self, rt: u32, rc: bool, ca: bool, ov: bool) {
        if ca {
            self.a.setcc(CC_B, RDX);
        }
        if ov {
            self.a.setcc(CC_O, RBX);
        }
        self.st(rt, RAX);
        self.xer_update(ca, ov);
        if rc {
            self.set_cr0();
        }
    }

    /// Zero edx/ebx before an op whose CF/OF will be captured.
    fn prep_flags(&mut self, ca: bool, ov: bool) {
        if ca {
            self.a.alu_rr(Alu::Xor, RDX, RDX);
        }
        if ov {
            self.a.alu_rr(Alu::Xor, RBX, RBX);
        }
    }

    // ---- memory

    /// ecx = EA of a D-form access.
    fn ea_d(&mut self, ra: u32, d: i32) {
        if ra == 0 {
            self.a.mov_ri(RCX, d as u32);
        } else {
            self.ld(RCX, ra);
            if d != 0 {
                self.a.alu_i(Alu::Add, Rm::R(RCX), d as u32);
            }
        }
    }
    /// ecx = EA of an X-form access.
    fn ea_x(&mut self, ra: u32, rb: u32) {
        if ra == 0 {
            self.ld(RCX, rb);
        } else {
            self.ld(RCX, ra);
            self.a.alu_rm(Alu::Add, RCX, Rm::M(gm(rb)));
        }
    }

    /// eax = load of `size` bytes at [ecx], big-endian unless `raw`; sign-extended if `sx`.
    fn load(&mut self, size: u32, sx: bool, raw: bool) {
        self.mark();
        match size {
            1 => {
                if sx {
                    self.a.movsx8(RAX, Rm::M(gaddr()));
                } else {
                    self.a.movzx8(RAX, Rm::M(gaddr()));
                }
            }
            2 => {
                self.a.movzx16(RAX, Rm::M(gaddr()));
                if !raw {
                    self.a.shift16_i(Shift::Ror, Rm::R(RAX), 8);
                }
                if sx {
                    self.a.movsx16(RAX, Rm::R(RAX));
                }
            }
            4 => {
                if raw || !self.movbe {
                    self.a.mov_rm(RAX, Rm::M(gaddr()));
                    if !raw {
                        self.a.bswap(RAX);
                    }
                } else {
                    self.a.movbe_rm(RAX, gaddr());
                }
            }
            _ => {
                if self.movbe {
                    self.a.movbe64_rm(RAX, gaddr());
                } else {
                    self.a.mov64_rm(RAX, Rm::M(gaddr()));
                    self.a.bswap64(RAX);
                }
            }
        }
    }

    /// Store `size` bytes of eax/rax (guest value) at [ecx], big-endian unless `raw`.
    fn store(&mut self, size: u32, raw: bool) {
        // pages with translated code or a watch take the slow path
        self.a.mov_rr(RDX, RCX);
        self.a.shift_i(Shift::Shr, Rm::R(RDX), 12);
        self.a.cmp8_i(Rm::M(mem_idx(R13, RDX, 1, PF_OFF)), 0);
        let site = self.a.jcc(CC_NE, 0);
        self.mark();
        match size {
            1 => self.a.mov8_mr(gaddr(), RAX),
            2 => {
                if raw {
                    self.a.mov16_mr(gaddr(), RAX);
                } else if self.movbe {
                    self.a.movbe16_mr(gaddr(), RAX);
                } else {
                    self.a.shift16_i(Shift::Ror, Rm::R(RAX), 8);
                    self.a.mov16_mr(gaddr(), RAX);
                }
            }
            4 => {
                if raw {
                    self.a.mov_mr(Rm::M(gaddr()), RAX);
                } else if self.movbe {
                    self.a.movbe_mr(gaddr(), RAX);
                } else {
                    self.a.bswap(RAX);
                    self.a.mov_mr(Rm::M(gaddr()), RAX);
                }
            }
            _ => {
                if self.movbe {
                    self.a.movbe64_mr(gaddr(), RAX);
                } else {
                    self.a.bswap64(RAX);
                    self.a.mov64_mr(Rm::M(gaddr()), RAX);
                }
            }
        }
        let back = self.a.here();
        let sz = if raw { size | 0x100 } else { size };
        self.cold.push(Cold::StoreSlow { site, back, size: sz, next_pc: self.pc.wrapping_add(4) });
    }

    // ---- branches

    /// Evaluate BO/BI: emit jumps to `not_taken` (returned sites) for the failing cases.
    fn branch_cond(&mut self, bo: u32, bi: u32) -> Vec<usize> {
        let mut sites = Vec::new();
        if bo & 4 == 0 {
            // decrement CTR, test it
            self.a.alu_i(Alu::Sub, Rm::M(sm(OFF_CTR)), 1);
            // bo bit 3 (0x2): branch if CTR == 0
            sites.push(self.a.jcc(if bo & 2 != 0 { CC_NE } else { CC_E }, 0));
        }
        if bo & 0x10 == 0 {
            self.a.test_i(Rm::M(sm(OFF_CR)), 1u32 << (31 - bi));
            // bo bit 1 (0x8): branch if the bit is set
            sites.push(self.a.jcc(if bo & 8 != 0 { CC_E } else { CC_NE }, 0));
        }
        sites
    }

    // ---- floating point helpers

    fn fp_load(&mut self, x: u8, fr: u32) {
        self.a.movsd_rm(x, fm(fr));
    }
    fn fp_store(&mut self, fr: u32, x: u8) {
        self.a.movsd_mr(fm(fr), x);
    }
    /// xmm0 = an arithmetic result: a NaN the operation produced is the positive default
    /// NaN on PowerPC (0x7FF8...), the negative one on x86.
    fn fp_fixnan(&mut self) {
        self.a.ucomisd(0, Rm::R(0));
        let ok = self.a.jcc(CC_NP, 0);
        self.a.movq_rx(RAX, 0);
        self.a.mov64_ri(RDX, 0xFFF8_0000_0000_0000);
        self.a.alu64_rr(Alu::Cmp, RAX, RDX);
        let ok2 = self.a.jcc(CC_NE, 0);
        self.a.mov64_ri(RDX, 0x7FF8_0000_0000_0000);
        self.a.movq_xr(0, RDX);
        let h = self.a.here();
        self.a.patch(ok, h);
        self.a.patch(ok2, h);
    }

    /// round xmm0 to single precision
    fn fp_single(&mut self) {
        self.a.cvtsd2ss(0, Rm::R(0));
        self.a.cvtss2sd(0, Rm::R(0));
    }
    /// CR1 <- FPSCR[0:3] (the Rc form of FP instructions)
    fn fp_rc(&mut self) {
        self.a.mov_rm(RCX, Rm::M(sm(OFF_FPSCR)));
        self.a.shift_i(Shift::Shr, Rm::R(RCX), 28);
        self.write_cr_nibble(1);
    }
    /// After an FPSCR change: the host rounding mode follows it.
    fn fpscr_sync(&mut self) {
        self.a.mov64_rr(RDI, R15);
        self.call_helper(h_fpscr_sync as usize);
    }

    // ---- the instruction

    /// Translate one instruction; sets `done` when the block ends.
    fn translate_one(&mut self, i: u32) {
        let op = i >> 26;
        let rt = (i >> 21) & 31;
        let ra = (i >> 16) & 31;
        let rb = (i >> 11) & 31;
        let rc = i & 1 != 0;
        let simm = (i as u16 as i16) as i32;
        let uimm = i & 0xFFFF;
        let pc = self.pc;
        let next = pc.wrapping_add(4);
        match op {
            3 => {
                // twi TO, ra, simm
                self.ld(RAX, ra);
                self.a.alu_i(Alu::Cmp, Rm::R(RAX), simm as u32);
                self.trap_conds(rt);
            }
            7 => {
                // mulli
                self.a.imul_rmi(RAX, Rm::M(gm(ra)), simm);
                self.st(rt, RAX);
            }
            8 => {
                // subfic: rt = simm - ra, CA = no borrow
                self.a.alu_rr(Alu::Xor, RDX, RDX);
                self.a.mov_ri(RAX, simm as u32);
                self.a.alu_rm(Alu::Sub, RAX, Rm::M(gm(ra)));
                self.a.setcc(CC_AE, RDX);
                self.st(rt, RAX);
                self.xer_update(true, false);
            }
            10 => {
                // cmpli
                self.ld(RAX, ra);
                self.compare_imm(uimm, false, rt >> 2);
            }
            11 => {
                // cmpi
                self.ld(RAX, ra);
                self.compare_imm(simm as u32, true, rt >> 2);
            }
            12 | 13 => {
                // addic, addic.
                self.prep_flags(true, false);
                self.ld(RAX, ra);
                self.a.alu_i(Alu::Add, Rm::R(RAX), simm as u32);
                self.finish_arith(rt, op == 13, true, false);
            }
            14 => {
                // addi / li
                if ra == 0 {
                    self.a.mov_ri(RAX, simm as u32);
                } else {
                    self.ld(RAX, ra);
                    if simm != 0 {
                        self.a.alu_i(Alu::Add, Rm::R(RAX), simm as u32);
                    }
                }
                self.st(rt, RAX);
            }
            15 => {
                // addis / lis
                let v = (uimm << 16) as u32;
                if ra == 0 {
                    self.a.mov_ri(RAX, v);
                } else {
                    self.ld(RAX, ra);
                    if v != 0 {
                        self.a.alu_i(Alu::Add, Rm::R(RAX), v);
                    }
                }
                self.st(rt, RAX);
            }
            16 => {
                // bc BO, BI, BD
                let bo = rt;
                let bi = ra;
                let bd = (i & 0xFFFC) as i16 as i32;
                let target = if i & 2 != 0 { bd as u32 } else { pc.wrapping_add(bd as u32) };
                if i & 1 != 0 {
                    self.a.mov_mi(sm(OFF_LR), next);
                }
                let fails = self.branch_cond(bo, bi);
                self.end_direct(target);
                if !fails.is_empty() {
                    let h = self.a.here();
                    for s in fails {
                        self.a.patch(s, h);
                    }
                    self.end_direct(next);
                }
                self.done = true;
            }
            17 => {
                // sc
                self.exit(EXIT_SYSCALL, next);
                self.done = true;
            }
            18 => {
                // b / ba / bl / bla
                let li = ((i & 0x03FF_FFFC) << 6) as i32 >> 6;
                let target = if i & 2 != 0 { li as u32 } else { pc.wrapping_add(li as u32) };
                if i & 1 != 0 {
                    self.a.mov_mi(sm(OFF_LR), next);
                }
                self.end_direct(target);
                self.done = true;
            }
            19 => self.op19(i),
            20 => {
                // rlwimi ra, rs, sh, mb, me  (rs is the rt field)
                let sh = rb;
                let mask = rot_mask((i >> 6) & 31, (i >> 1) & 31);
                self.ld(RAX, rt);
                if sh != 0 {
                    self.a.shift_i(Shift::Rol, Rm::R(RAX), sh as u8);
                }
                self.a.alu_i(Alu::And, Rm::R(RAX), mask);
                self.ld(RDX, ra);
                self.a.alu_i(Alu::And, Rm::R(RDX), !mask);
                self.a.alu_rr(Alu::Or, RAX, RDX);
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            21 => {
                // rlwinm ra, rs, sh, mb, me
                let sh = rb;
                let mask = rot_mask((i >> 6) & 31, (i >> 1) & 31);
                self.ld(RAX, rt);
                if sh != 0 {
                    self.a.shift_i(Shift::Rol, Rm::R(RAX), sh as u8);
                }
                if mask != 0xFFFF_FFFF {
                    self.a.alu_i(Alu::And, Rm::R(RAX), mask);
                }
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            23 => {
                // rlwnm ra, rs, rb, mb, me
                let mask = rot_mask((i >> 6) & 31, (i >> 1) & 31);
                self.ld(RAX, rt);
                self.ld(RCX, rb);
                self.a.shift_cl(Shift::Rol, Rm::R(RAX));
                if mask != 0xFFFF_FFFF {
                    self.a.alu_i(Alu::And, Rm::R(RAX), mask);
                }
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            24 | 25 | 26 | 27 | 28 | 29 => {
                // ori oris xori xoris andi. andis.  (rt is the source rS, ra the target)
                let imm = if op & 1 != 0 { uimm << 16 } else { uimm };
                if op == 24 && imm == 0 && rt == ra {
                    return; // ori r,r,0: nop
                }
                let alu = match op {
                    24 | 25 => Alu::Or,
                    26 | 27 => Alu::Xor,
                    _ => Alu::And,
                };
                self.ld(RAX, rt);
                if !(imm == 0 && alu != Alu::And) {
                    self.a.alu_i(alu, Rm::R(RAX), imm);
                }
                self.st(ra, RAX);
                if op >= 28 {
                    self.set_cr0();
                }
            }
            31 => self.op31(i),
            32..=45 => {
                // lwz lwzu lbz lbzu stw stwu stb stbu lhz lhzu lha lhau sth sthu
                let (size, sx, is_store) = match op {
                    32 | 33 => (4, false, false),
                    34 | 35 => (1, false, false),
                    36 | 37 => (4, false, true),
                    38 | 39 => (1, false, true),
                    40 | 41 => (2, false, false),
                    42 | 43 => (2, true, false),
                    _ => (2, false, true),
                };
                let update = op & 1 != 0;
                self.ea_d(ra, simm);
                if is_store {
                    self.ld(RAX, rt);
                    self.store(size, false);
                } else {
                    self.load(size, sx, false);
                    self.st(rt, RAX);
                }
                if update {
                    self.st(ra, RCX);
                }
            }
            46 => {
                // lmw: inline loads (xlc's prologues use it constantly)
                self.ea_d(ra, simm);
                self.mark();
                for r in rt..32 {
                    let off = (4 * (r - rt)) as i32;
                    if self.movbe {
                        self.a.movbe_rm(RAX, gaddr_off(off));
                    } else {
                        self.a.mov_rm(RAX, Rm::M(gaddr_off(off)));
                        self.a.bswap(RAX);
                    }
                    self.st(r, RAX);
                }
            }
            47 => {
                // stmw: inline unless the range touches a page with code or a watch
                self.ea_d(ra, simm);
                let len = 4 * (32 - rt);
                self.a.mov_rr(RDX, RCX);
                self.a.shift_i(Shift::Shr, Rm::R(RDX), 12);
                self.a.movzx8(RDX, Rm::M(mem_idx(R13, RDX, 1, PF_OFF)));
                self.a.lea(RDI, mem(RCX, len as i32 - 1));
                self.a.shift_i(Shift::Shr, Rm::R(RDI), 12);
                self.a.alu_rm(Alu::Or, RDX, Rm::M(mem_idx(R13, RDI, 1, PF_OFF)));
                self.a.test8_i(Rm::R(RDX), 0xFF);
                let slow = self.a.jcc(CC_NE, 0);
                self.mark();
                for r in rt..32 {
                    let off = (4 * (r - rt)) as i32;
                    self.ld(RAX, r);
                    if self.movbe {
                        self.a.movbe_mr(gaddr_off(off), RAX);
                    } else {
                        self.a.bswap(RAX);
                        self.a.mov_mr(Rm::M(gaddr_off(off)), RAX);
                    }
                }
                let done = self.a.jmp(0);
                let h = self.a.here();
                self.a.patch(slow, h);
                self.a.mov64_rr(RDI, R15);
                self.a.mov_rr(RSI, RCX);
                self.a.mov_ri(RDX, rt);
                self.a.mov_ri(RCX, pc);
                self.call_helper(h_stmw as usize);
                self.store_helper_exit();
                let h2 = self.a.here();
                self.a.patch(done, h2);
                // (no update form: ecx is dead here)
            }
            48..=55 => {
                // lfs lfsu lfd lfdu stfs stfsu stfd stfdu
                let update = op & 1 != 0;
                self.ea_d(ra, simm);
                match op {
                    48 | 49 => {
                        self.load(4, false, false);
                        self.a.movd_xr(0, RAX);
                        self.a.cvtss2sd(0, Rm::R(0));
                        self.fp_store(rt, 0);
                    }
                    50 | 51 => {
                        self.load(8, false, false);
                        self.a.mov64_mr(Rm::M(fm(rt)), RAX);
                    }
                    52 | 53 => {
                        self.to_single(rt);
                        self.store(4, false);
                    }
                    _ => {
                        self.a.mov64_rm(RAX, Rm::M(fm(rt)));
                        self.store(8, false);
                    }
                }
                if update {
                    self.st(ra, RCX);
                }
            }
            59 => self.op59(i),
            63 => self.op63(i),
            _ => {
                self.exit(EXIT_ILLEGAL, pc);
                self.done = true;
            }
        }
    }

    /// After a helper that stores: eax != 0 means the block overwrote itself.
    fn store_helper_exit(&mut self) {
        self.a.test_rr(RAX, RAX);
        let j = self.a.jcc(CC_E, 0);
        self.a.mov_ri(RAX, EXIT_STOP);
        let ep = self.cpu.epilogue_addr();
        self.jmp_abs(ep);
        let h = self.a.here();
        self.a.patch(j, h);
    }

    /// eax = the single-precision image of FPR `fr` for stfs: not a rounding but the
    /// hardware's bit selection (sign, exponent bit 0 and the low 7 exponent bits, the top
    /// 23 mantissa bits); denormal results go through a helper.
    fn to_single(&mut self, fr: u32) {
        self.a.mov64_rm(RAX, Rm::M(fm(fr)));
        self.a.mov64_rr(RDX, RAX);
        self.a.shift64_i(Shift::Shr, Rm::R(RDX), 52);
        self.a.alu_i(Alu::And, Rm::R(RDX), 0x7FF);
        self.a.alu_i(Alu::Cmp, Rm::R(RDX), 896);
        let site = self.a.jcc(CC_BE, 0);
        self.a.mov64_rr(RDX, RAX);
        self.a.shift64_i(Shift::Shr, Rm::R(RDX), 32);
        self.a.alu_i(Alu::And, Rm::R(RDX), 0xC000_0000);
        self.a.shift64_i(Shift::Shr, Rm::R(RAX), 29);
        self.a.alu_i(Alu::And, Rm::R(RAX), 0x3FFF_FFFF);
        self.a.alu_rr(Alu::Or, RAX, RDX);
        let back = self.a.here();
        self.cold.push(Cold::ToSingle { site, back });
    }

    /// ecx = EA: an alignment exception unless the low bits are clear.
    fn align_check(&mut self, lowbits: u32) {
        self.a.test_i(Rm::R(RCX), lowbits);
        let j = self.a.jcc(CC_NE, 0);
        self.cold.push(Cold::Align { site: j, pc: self.pc });
    }

    /// After `cmp eax, x`: trap if any TO condition holds.
    fn trap_conds(&mut self, to: u32) {
        let pc = self.pc;
        if to & 31 == 31 {
            let j = self.a.jmp(0);
            self.cold.push(Cold::Trap { site: j, pc });
            self.done = true;
            return;
        }
        let conds: [(u32, u8); 5] = [(0x10, CC_L), (0x08, CC_G), (0x04, CC_E), (0x02, CC_B), (0x01, CC_A)];
        for (bit, cc) in conds {
            if to & bit != 0 {
                let j = self.a.jcc(cc, 0);
                self.cold.push(Cold::Trap { site: j, pc });
            }
        }
    }

    fn op19(&mut self, i: u32) {
        let xo = (i >> 1) & 0x3FF;
        let bt = (i >> 21) & 31;
        let ba = (i >> 16) & 31;
        let bb = (i >> 11) & 31;
        let pc = self.pc;
        let next = pc.wrapping_add(4);
        match xo {
            16 | 528 => {
                // bclr / bcctr
                let bo = bt;
                let bi = ba;
                if xo == 16 {
                    self.a.mov_rm(RAX, Rm::M(sm(OFF_LR)));
                } else {
                    self.a.mov_rm(RAX, Rm::M(sm(OFF_CTR)));
                }
                self.a.alu_i(Alu::And, Rm::R(RAX), !3u32);
                if i & 1 != 0 {
                    self.a.mov_mi(sm(OFF_LR), next);
                }
                let fails = self.branch_cond(bo, bi);
                self.end_indirect();
                if !fails.is_empty() {
                    let h = self.a.here();
                    for s in fails {
                        self.a.patch(s, h);
                    }
                    self.end_direct(next);
                }
                self.done = true;
            }
            150 => {} // isync
            0 => {
                // mcrf
                let (tf, sf) = (bt >> 2, ba >> 2);
                self.a.mov_rm(RCX, Rm::M(sm(OFF_CR)));
                self.a.shift_i(Shift::Shr, Rm::R(RCX), 28 - 4 * sf as u8);
                self.a.alu_i(Alu::And, Rm::R(RCX), 0xF);
                self.write_cr_nibble(tf);
            }
            33 | 129 | 193 | 225 | 257 | 289 | 417 | 449 => {
                // crnor crandc crxor crnand crand creqv crorc cror
                self.a.mov_rm(RAX, Rm::M(sm(OFF_CR)));
                self.a.mov_rr(RCX, RAX);
                self.a.shift_i(Shift::Shr, Rm::R(RAX), (31 - ba) as u8);
                self.a.shift_i(Shift::Shr, Rm::R(RCX), (31 - bb) as u8);
                match xo {
                    33 => {
                        self.a.alu_rr(Alu::Or, RAX, RCX);
                        self.a.not(Rm::R(RAX));
                    }
                    129 => {
                        self.a.not(Rm::R(RCX));
                        self.a.alu_rr(Alu::And, RAX, RCX);
                    }
                    193 => self.a.alu_rr(Alu::Xor, RAX, RCX),
                    225 => {
                        self.a.alu_rr(Alu::And, RAX, RCX);
                        self.a.not(Rm::R(RAX));
                    }
                    257 => self.a.alu_rr(Alu::And, RAX, RCX),
                    289 => {
                        self.a.alu_rr(Alu::Xor, RAX, RCX);
                        self.a.not(Rm::R(RAX));
                    }
                    417 => {
                        self.a.not(Rm::R(RCX));
                        self.a.alu_rr(Alu::Or, RAX, RCX);
                    }
                    _ => self.a.alu_rr(Alu::Or, RAX, RCX),
                }
                self.a.alu_i(Alu::And, Rm::R(RAX), 1);
                self.a.shift_i(Shift::Shl, Rm::R(RAX), (31 - bt) as u8);
                self.a.mov_rm(RDX, Rm::M(sm(OFF_CR)));
                self.a.alu_i(Alu::And, Rm::R(RDX), !(1u32 << (31 - bt)));
                self.a.alu_rr(Alu::Or, RDX, RAX);
                self.a.mov_mr(Rm::M(sm(OFF_CR)), RDX);
            }
            _ => {
                self.exit(EXIT_ILLEGAL, pc);
                self.done = true;
            }
        }
    }

    fn op31(&mut self, i: u32) {
        let xo = (i >> 1) & 0x3FF;
        let xo9 = (i >> 1) & 0x1FF;
        let oe = (i >> 10) & 1 != 0;
        let rt = (i >> 21) & 31;
        let ra = (i >> 16) & 31;
        let rb = (i >> 11) & 31;
        let rc = i & 1 != 0;
        let pc = self.pc;
        // arithmetic with OE (9-bit extended opcode)
        match xo9 {
            8 | 40 | 104 | 136 | 200 | 232 | 10 | 138 | 202 | 234 | 266 | 11 | 75 | 235 | 459 | 491 => {
                self.arith(xo9, rt, ra, rb, oe, rc);
                return;
            }
            _ => {}
        }
        match xo {
            0 => {
                self.ld(RAX, ra);
                self.compare(Rm::M(gm(rb)), true, rt >> 2);
            }
            32 => {
                self.ld(RAX, ra);
                self.compare(Rm::M(gm(rb)), false, rt >> 2);
            }
            4 => {
                // tw TO, ra, rb
                self.ld(RAX, ra);
                self.a.alu_rm(Alu::Cmp, RAX, Rm::M(gm(rb)));
                self.trap_conds(rt);
            }
            19 => {
                // mfcr / mfocrf
                self.a.mov_rm(RAX, Rm::M(sm(OFF_CR)));
                self.st(rt, RAX);
            }
            144 => {
                // mtcrf CRM, rs
                let crm = (i >> 12) & 0xFF;
                let mut mask = 0u32;
                for f in 0..8 {
                    if crm & (0x80 >> f) != 0 {
                        mask |= 0xF << (28 - 4 * f);
                    }
                }
                self.ld(RAX, rt);
                if mask != 0xFFFF_FFFF {
                    self.a.alu_i(Alu::And, Rm::R(RAX), mask);
                    self.a.mov_rm(RDX, Rm::M(sm(OFF_CR)));
                    self.a.alu_i(Alu::And, Rm::R(RDX), !mask);
                    self.a.alu_rr(Alu::Or, RAX, RDX);
                }
                self.a.mov_mr(Rm::M(sm(OFF_CR)), RAX);
            }
            512 => {
                // mcrxr: CR field <- SO OV CA 0, XER's SO OV CA <- 0 (bit 3 untouched, as QEMU)
                self.a.mov_rm(RCX, Rm::M(sm(OFF_XER)));
                self.a.mov_rr(RDX, RCX);
                self.a.alu_i(Alu::And, Rm::R(RDX), 0x1FFF_FFFF);
                self.a.mov_mr(Rm::M(sm(OFF_XER)), RDX);
                self.a.shift_i(Shift::Shr, Rm::R(RCX), 28);
                self.a.alu_i(Alu::And, Rm::R(RCX), 0xE);
                self.write_cr_nibble(rt >> 2);
            }
            339 | 371 => {
                // mfspr / mftb
                let spr = ((i >> 16) & 31) | (((i >> 11) & 31) << 5);
                match (xo, spr) {
                    (339, 1) => {
                        self.a.mov_rm(RAX, Rm::M(sm(OFF_XER)));
                        self.st(rt, RAX);
                    }
                    (339, 8) => {
                        self.a.mov_rm(RAX, Rm::M(sm(OFF_LR)));
                        self.st(rt, RAX);
                    }
                    (339, 9) => {
                        self.a.mov_rm(RAX, Rm::M(sm(OFF_CTR)));
                        self.st(rt, RAX);
                    }
                    (_, 268) | (_, 269) => {
                        self.a.mov64_rr(RDI, R15);
                        self.call_helper(h_timebase as usize);
                        if spr == 269 {
                            self.a.shift64_i(Shift::Shr, Rm::R(RAX), 32);
                        }
                        self.st(rt, RAX);
                    }
                    (339, 287) => {
                        // PVR: a 604e
                        self.a.mov_ri(RAX, 0x0009_0204);
                        self.st(rt, RAX);
                    }
                    _ => {
                        self.exit(EXIT_ILLEGAL, pc);
                        self.done = true;
                    }
                }
            }
            467 => {
                // mtspr
                let spr = ((i >> 16) & 31) | (((i >> 11) & 31) << 5);
                let off = match spr {
                    1 => OFF_XER,
                    8 => OFF_LR,
                    9 => OFF_CTR,
                    _ => {
                        self.exit(EXIT_ILLEGAL, pc);
                        self.done = true;
                        return;
                    }
                };
                self.ld(RAX, rt);
                if spr == 1 {
                    // bits 18-19 (OV32/CA32 in ISA 3) do not exist here, as in QEMU
                    self.a.alu_i(Alu::And, Rm::R(RAX), !0x000C_0000u32);
                }
                self.a.mov_mr(Rm::M(sm(off)), RAX);
            }
            26 => {
                // cntlzw
                self.ld(RAX, rt);
                self.a.mov_ri(RDX, 63);
                self.a.bsr(RCX, Rm::R(RAX));
                self.a.cmov(CC_E, RCX, Rm::R(RDX));
                self.a.alu_i(Alu::Xor, Rm::R(RCX), 31);
                self.a.mov_rr(RAX, RCX);
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            28 | 60 | 124 | 284 | 316 | 412 | 444 | 476 => {
                // and andc nor eqv xor orc or nand: ra = rs(rt) op rb
                self.ld(RAX, rt);
                match xo {
                    28 => self.a.alu_rm(Alu::And, RAX, Rm::M(gm(rb))),
                    60 => {
                        self.ld(RCX, rb);
                        self.a.not(Rm::R(RCX));
                        self.a.alu_rr(Alu::And, RAX, RCX);
                    }
                    124 => {
                        self.a.alu_rm(Alu::Or, RAX, Rm::M(gm(rb)));
                        self.a.not(Rm::R(RAX));
                    }
                    284 => {
                        self.a.alu_rm(Alu::Xor, RAX, Rm::M(gm(rb)));
                        self.a.not(Rm::R(RAX));
                    }
                    316 => self.a.alu_rm(Alu::Xor, RAX, Rm::M(gm(rb))),
                    412 => {
                        self.ld(RCX, rb);
                        self.a.not(Rm::R(RCX));
                        self.a.alu_rr(Alu::Or, RAX, RCX);
                    }
                    444 => {
                        if rb != rt {
                            self.a.alu_rm(Alu::Or, RAX, Rm::M(gm(rb)));
                        }
                    }
                    _ => {
                        self.a.alu_rm(Alu::And, RAX, Rm::M(gm(rb)));
                        self.a.not(Rm::R(RAX));
                    }
                }
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            922 | 954 => {
                // extsh / extsb
                if xo == 922 {
                    self.a.movsx16(RAX, Rm::M(gm(rt)));
                } else {
                    self.a.movsx8(RAX, Rm::M(gm(rt)));
                }
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            24 | 536 => {
                // slw / srw: ra = rs << rb[26:31] (0 when >= 32)
                self.ld(RAX, rt);
                self.ld(RCX, rb);
                self.a.shift_cl(if xo == 24 { Shift::Shl } else { Shift::Shr }, Rm::R(RAX));
                self.a.alu_rr(Alu::Xor, RDX, RDX);
                self.a.test_i(Rm::R(RCX), 32);
                self.a.cmov(CC_NE, RAX, Rm::R(RDX));
                self.st(ra, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            792 => {
                // sraw
                self.ld(RAX, rt);
                self.ld(RCX, rb);
                self.a.alu_i(Alu::And, Rm::R(RCX), 63);
                self.a.alu_rr(Alu::Xor, RBX, RBX);
                self.a.alu_i(Alu::Cmp, Rm::R(RCX), 32);
                let big = self.a.jcc(CC_AE, 0);
                // small shift: CA = sign && (bits shifted out != 0)
                self.a.mov_rr(RDX, RAX);
                self.a.mov_ri(RBX, 1);
                self.a.shift_cl(Shift::Shl, Rm::R(RBX));
                self.a.alu_i(Alu::Sub, Rm::R(RBX), 1);
                self.a.alu_rr(Alu::And, RBX, RDX);
                self.a.alu_rr(Alu::Xor, RDI, RDI);
                self.a.test_rr(RBX, RBX);
                self.a.setcc(CC_NE, RDI);
                self.a.mov_rr(RBX, RDI);
                self.a.shift_cl(Shift::Sar, Rm::R(RAX));
                let j = self.a.jmp(0);
                let h = self.a.here();
                self.a.patch(big, h);
                // shift >= 32: result is the sign, CA = sign
                self.a.shift_i(Shift::Sar, Rm::R(RAX), 31);
                self.a.mov_rr(RBX, RAX);
                self.a.alu_i(Alu::And, Rm::R(RBX), 1);
                let h2 = self.a.here();
                self.a.patch(j, h2);
                // CA = ebx & (rs >> 31)
                self.a.mov_rm(RDX, Rm::M(gm(rt)));
                self.a.shift_i(Shift::Shr, Rm::R(RDX), 31);
                self.a.alu_rr(Alu::And, RDX, RBX);
                self.st(ra, RAX);
                self.xer_update(true, false);
                if rc {
                    self.set_cr0();
                }
            }
            824 => {
                // srawi
                let sh = rb;
                self.ld(RAX, rt);
                self.a.alu_rr(Alu::Xor, RDX, RDX);
                if sh != 0 {
                    self.a.mov_rr(RCX, RAX);
                    self.a.shift_i(Shift::Sar, Rm::R(RAX), sh as u8);
                    self.a.test_i(Rm::R(RCX), (1u32 << sh) - 1);
                    self.a.setcc(CC_NE, RDX);
                    self.a.shift_i(Shift::Shr, Rm::R(RCX), 31);
                    self.a.alu_rr(Alu::And, RDX, RCX);
                }
                self.st(ra, RAX);
                self.xer_update(true, false);
                if rc {
                    self.set_cr0();
                }
            }
            20 => {
                // lwarx (an alignment exception when the address is not a word's)
                self.ea_x(ra, rb);
                self.align_check(3);
                self.mark();
                self.a.mov_rm(RAX, Rm::M(gaddr()));
                self.a.mov_mr(Rm::M(sm(OFF_RES_ADDR)), RCX);
                self.a.mov_mr(Rm::M(sm(OFF_RES_VAL)), RAX);
                self.a.mov_mi(sm(OFF_RES_VALID), 1);
                self.a.bswap(RAX);
                self.st(rt, RAX);
            }
            150 => {
                // stwcx. (the alignment check only where the store happens, as QEMU)
                self.ea_x(ra, rb);
                self.ld(RDX, rt);
                self.a.bswap(RDX);
                self.a.alu_rr(Alu::Xor, RBX, RBX);
                self.a.alu_rm(Alu::Cmp, RCX, Rm::M(sm(OFF_RES_ADDR)));
                let f1 = self.a.jcc(CC_NE, 0);
                self.a.alu_i(Alu::Cmp, Rm::M(sm(OFF_RES_VALID)), 0);
                let f2 = self.a.jcc(CC_E, 0);
                self.align_check(3);
                self.a.mov_rm(RAX, Rm::M(sm(OFF_RES_VAL)));
                self.mark();
                self.a.lock_cmpxchg(gaddr(), RDX);
                self.a.setcc(CC_E, RBX);
                let h = self.a.here();
                self.a.patch(f1, h);
                self.a.patch(f2, h);
                self.a.mov_mi(sm(OFF_RES_VALID), 0);
                // CR0 = 0b00 EQ SO
                self.a.mov_rm(RDX, Rm::M(sm(OFF_XER)));
                self.a.shift_i(Shift::Shr, Rm::R(RDX), 31);
                self.a.lea(RCX, mem_idx(RDX, RBX, 2, 0));
                self.write_cr_nibble(0);
                // a successful store may have hit code
                self.a.test_rr(RBX, RBX);
                let skip = self.a.jcc(CC_E, 0);
                self.a.mov_rm(RCX, Rm::M(sm(OFF_RES_ADDR)));
                self.a.mov_rr(RDX, RCX);
                self.a.shift_i(Shift::Shr, Rm::R(RDX), 12);
                self.a.cmp8_i(Rm::M(mem_idx(R13, RDX, 1, PF_OFF)), 0);
                let skip2 = self.a.jcc(CC_E, 0);
                self.a.mov64_rr(RDI, R15);
                self.a.mov_rr(RSI, RCX);
                self.a.mov_ri(RDX, 4);
                self.a.mov_ri(RCX, pc.wrapping_add(4));
                self.call_helper(h_code_touched as usize);
                self.store_helper_exit();
                let h = self.a.here();
                self.a.patch(skip, h);
                self.a.patch(skip2, h);
            }
            23 | 55 | 87 | 119 | 279 | 311 | 343 | 375 | 534 | 790 => {
                // lwzx lwzux lbzx lbzux lhzx lhzux lhax lhaux lwbrx lhbrx
                let (size, sx, raw) = match xo {
                    23 | 55 => (4, false, false),
                    87 | 119 => (1, false, false),
                    279 | 311 => (2, false, false),
                    343 | 375 => (2, true, false),
                    534 => (4, false, true),
                    _ => (2, false, true),
                };
                self.ea_x(ra, rb);
                self.load(size, sx, raw);
                self.st(rt, RAX);
                if matches!(xo, 55 | 119 | 311 | 375) {
                    self.st(ra, RCX);
                }
            }
            151 | 183 | 215 | 247 | 407 | 439 | 662 | 918 => {
                // stwx stwux stbx stbux sthx sthux stwbrx sthbrx
                let (size, raw) = match xo {
                    151 | 183 => (4, false),
                    215 | 247 => (1, false),
                    407 | 439 => (2, false),
                    662 => (4, true),
                    _ => (2, true),
                };
                self.ea_x(ra, rb);
                self.ld(RAX, rt);
                self.store(size, raw);
                if matches!(xo, 183 | 247 | 439) {
                    self.st(ra, RCX);
                }
            }
            535 | 567 | 599 | 631 => {
                // lfsx lfsux lfdx lfdux
                self.ea_x(ra, rb);
                if xo < 599 {
                    self.load(4, false, false);
                    self.a.movd_xr(0, RAX);
                    self.a.cvtss2sd(0, Rm::R(0));
                    self.fp_store(rt, 0);
                } else {
                    self.load(8, false, false);
                    self.a.mov64_mr(Rm::M(fm(rt)), RAX);
                }
                if xo == 567 || xo == 631 {
                    self.st(ra, RCX);
                }
            }
            663 | 695 | 727 | 759 | 983 => {
                // stfsx stfsux stfdx stfdux stfiwx
                self.ea_x(ra, rb);
                match xo {
                    663 | 695 => {
                        self.to_single(rt);
                        self.store(4, false);
                    }
                    983 => {
                        self.a.mov_rm(RAX, Rm::M(fm(rt)));
                        self.store(4, false);
                    }
                    _ => {
                        self.a.mov64_rm(RAX, Rm::M(fm(rt)));
                        self.store(8, false);
                    }
                }
                if xo == 695 || xo == 759 {
                    self.st(ra, RCX);
                }
            }
            533 | 597 => {
                // lswx / lswi
                self.ea_x_or_a(ra, rb, xo == 597);
                self.a.mov64_rr(RDI, R15);
                self.a.mov_rr(RSI, RCX);
                self.a.mov_ri(RDX, rt);
                if xo == 597 {
                    self.a.mov_ri(RCX, if rb == 0 { 32 } else { rb });
                } else {
                    self.a.mov_rm(RCX, Rm::M(sm(OFF_XER)));
                    self.a.alu_i(Alu::And, Rm::R(RCX), 0x7F);
                }
                self.a.mov_ri(R8, pc);
                self.call_helper(h_lsw as usize);
            }
            661 | 725 => {
                // stswx / stswi
                self.ea_x_or_a(ra, rb, xo == 725);
                self.a.mov64_rr(RDI, R15);
                self.a.mov_rr(RSI, RCX);
                self.a.mov_ri(RDX, rt);
                if xo == 725 {
                    self.a.mov_ri(RCX, if rb == 0 { 32 } else { rb });
                } else {
                    self.a.mov_rm(RCX, Rm::M(sm(OFF_XER)));
                    self.a.alu_i(Alu::And, Rm::R(RCX), 0x7F);
                }
                self.a.mov_ri(R8, pc);
                self.call_helper(h_stsw as usize);
                self.store_helper_exit();
            }
            598 | 854 | 278 | 246 | 86 | 54 | 982 | 470 => {} // sync eieio dcbt dcbtst dcbf dcbst icbi dcbi
            1014 => {
                // dcbz: zero the 32-byte line
                self.ea_x(ra, rb);
                self.a.alu_i(Alu::And, Rm::R(RCX), !31u32);
                self.a.mov_rr(RDX, RCX);
                self.a.shift_i(Shift::Shr, Rm::R(RDX), 12);
                self.a.cmp8_i(Rm::M(mem_idx(R13, RDX, 1, PF_OFF)), 0);
                let site = self.a.jcc(CC_NE, 0);
                self.mark();
                self.a.alu_rr(Alu::Xor, RAX, RAX);
                for k in 0..4 {
                    self.a.mov64_mr(Rm::M(gaddr_off(8 * k)), RAX);
                }
                let back = self.a.here();
                self.cold.push(Cold::StoreSlow { site, back, size: 32, next_pc: pc.wrapping_add(4) });
            }
            _ => {
                self.exit(EXIT_ILLEGAL, pc);
                self.done = true;
            }
        }
    }

    /// EA for the string instructions: X-form, or (ra|0) for the immediate forms.
    fn ea_x_or_a(&mut self, ra: u32, rb: u32, imm: bool) {
        if imm {
            if ra == 0 {
                self.a.alu_rr(Alu::Xor, RCX, RCX);
            } else {
                self.ld(RCX, ra);
            }
        } else {
            self.ea_x(ra, rb);
        }
    }

    /// The add/subtract/multiply/divide family (opcode 31, 9-bit XO).
    fn arith(&mut self, xo9: u32, rt: u32, ra: u32, rb: u32, oe: bool, rc: bool) {
        match xo9 {
            266 => {
                // add
                self.prep_flags(false, oe);
                self.ld(RAX, ra);
                self.a.alu_rm(Alu::Add, RAX, Rm::M(gm(rb)));
                self.finish_arith(rt, rc, false, oe);
            }
            10 => {
                // addc
                self.prep_flags(true, oe);
                self.ld(RAX, ra);
                self.a.alu_rm(Alu::Add, RAX, Rm::M(gm(rb)));
                self.finish_arith(rt, rc, true, oe);
            }
            138 => {
                // adde
                self.prep_flags(true, oe);
                self.ld(RAX, ra);
                self.ca_to_cf();
                self.a.alu_rm(Alu::Adc, RAX, Rm::M(gm(rb)));
                self.finish_arith(rt, rc, true, oe);
            }
            202 | 234 => {
                // addze / addme
                self.prep_flags(true, oe);
                self.ld(RAX, ra);
                self.ca_to_cf();
                self.a.alu_i(Alu::Adc, Rm::R(RAX), if xo9 == 202 { 0 } else { 0xFFFF_FFFF });
                self.finish_arith(rt, rc, true, oe);
            }
            40 => {
                // subf: rb - ra
                self.prep_flags(false, oe);
                self.ld(RAX, rb);
                self.a.alu_rm(Alu::Sub, RAX, Rm::M(gm(ra)));
                self.finish_arith(rt, rc, false, oe);
            }
            8 => {
                // subfc: CA = no borrow
                self.prep_flags(true, oe);
                self.ld(RAX, rb);
                self.a.alu_rm(Alu::Sub, RAX, Rm::M(gm(ra)));
                self.a.setcc(CC_AE, RDX);
                if oe {
                    self.a.setcc(CC_O, RBX);
                }
                self.st(rt, RAX);
                self.xer_update(true, oe);
                if rc {
                    self.set_cr0();
                }
            }
            136 => {
                // subfe: ~ra + rb + CA
                self.prep_flags(true, oe);
                self.ld(RAX, ra);
                self.a.not(Rm::R(RAX));
                self.ca_to_cf();
                self.a.alu_rm(Alu::Adc, RAX, Rm::M(gm(rb)));
                self.finish_arith(rt, rc, true, oe);
            }
            200 | 232 => {
                // subfze / subfme: ~ra + CA (+ -1)
                self.prep_flags(true, oe);
                self.ld(RAX, ra);
                self.a.not(Rm::R(RAX));
                self.ca_to_cf();
                self.a.alu_i(Alu::Adc, Rm::R(RAX), if xo9 == 200 { 0 } else { 0xFFFF_FFFF });
                self.finish_arith(rt, rc, true, oe);
            }
            104 => {
                // neg
                self.prep_flags(false, oe);
                self.ld(RAX, ra);
                self.a.neg(Rm::R(RAX));
                self.finish_arith(rt, rc, false, oe);
            }
            235 => {
                // mullw
                self.prep_flags(false, oe);
                self.ld(RAX, ra);
                self.a.imul_rm(RAX, Rm::M(gm(rb)));
                self.finish_arith(rt, rc, false, oe);
            }
            75 | 11 => {
                // mulhw / mulhwu
                self.ld(RAX, ra);
                if xo9 == 75 {
                    self.a.imul1(Rm::M(gm(rb)));
                } else {
                    self.a.mul(Rm::M(gm(rb)));
                }
                self.a.mov_rr(RAX, RDX);
                self.st(rt, RAX);
                if rc {
                    self.set_cr0();
                }
            }
            491 | 459 => {
                // divw / divwu: on overflow (division by zero, INT_MIN / -1) the result is
                // the dividend, as QEMU does it (the architecture leaves it undefined)
                self.ld(RAX, ra);
                self.ld(RCX, rb);
                self.a.alu_rr(Alu::Xor, RBX, RBX);
                self.a.test_rr(RCX, RCX);
                let z = self.a.jcc(CC_E, 0);
                let mut ovf_site = None;
                if xo9 == 491 {
                    self.a.alu_i(Alu::Cmp, Rm::R(RCX), 0xFFFF_FFFF);
                    let ok = self.a.jcc(CC_NE, 0);
                    self.a.alu_i(Alu::Cmp, Rm::R(RAX), 0x8000_0000);
                    ovf_site = Some(self.a.jcc(CC_E, 0));
                    let h = self.a.here();
                    self.a.patch(ok, h);
                    self.a.cdq();
                    self.a.idiv(Rm::R(RCX));
                } else {
                    self.a.alu_rr(Alu::Xor, RDX, RDX);
                    self.a.div(Rm::R(RCX));
                }
                let done = self.a.jmp(0);
                let h = self.a.here();
                self.a.patch(z, h);
                if let Some(s) = ovf_site {
                    self.a.patch(s, h);
                }
                self.a.mov_ri(RBX, 1);
                let h2 = self.a.here();
                self.a.patch(done, h2);
                self.st(rt, RAX);
                self.xer_update(false, oe);
                if rc {
                    self.set_cr0();
                }
            }
            _ => unreachable!(),
        }
    }

    fn op59(&mut self, i: u32) {
        let xo = (i >> 1) & 31;
        let frt = (i >> 21) & 31;
        let fra = (i >> 16) & 31;
        let frb = (i >> 11) & 31;
        let frc = (i >> 6) & 31;
        let rc = i & 1 != 0;
        match xo {
            18 | 20 | 21 | 25 => {
                self.fp_load(0, fra);
                match xo {
                    18 => self.a.divsd(0, Rm::M(fm(frb))),
                    20 => self.a.subsd(0, Rm::M(fm(frb))),
                    21 => self.a.addsd(0, Rm::M(fm(frb))),
                    _ => self.a.mulsd(0, Rm::M(fm(frc))),
                }
                self.fp_fixnan();
                self.fp_single();
                self.fp_store(frt, 0);
            }
            22 => {
                self.a.sqrtsd(0, Rm::M(fm(frb)));
                self.fp_fixnan();
                self.fp_single();
                self.fp_store(frt, 0);
            }
            24 => {
                // fres
                self.a.mov64_rr(RDI, R15);
                self.a.mov_ri(RSI, frt);
                self.a.mov_ri(RDX, frb);
                self.call_helper(h_fres as usize);
            }
            28..=31 => {
                self.fmadd(xo, frt, fra, frb, frc, true);
            }
            _ => {
                self.exit(EXIT_ILLEGAL, self.pc);
                self.done = true;
                return;
            }
        }
        if rc {
            self.fp_rc();
        }
    }

    /// fmsub(28) fmadd(29) fnmsub(30) fnmadd(31): frt = +-(fra*frc +- frb)
    fn fmadd(&mut self, xo: u32, frt: u32, fra: u32, frb: u32, frc: u32, single: bool) {
        self.fp_load(0, fra);
        self.fp_load(1, frc);
        self.fp_load(2, frb);
        // fnmadd/fnmsub negate the *result* of fmadd/fmsub: computing fmadd and flipping
        // the sign gives the right sign of a zero result, which the negated-product FMA
        // forms do not. Under a directed rounding mode that order rounds the wrong way, so
        // there the negated-product forms are used (and a zero's sign may differ).
        let mut neg_done = None;
        if self.fma {
            if xo >= 30 {
                self.a.test_i(Rm::M(sm(OFF_FPSCR)), 3);
                let nearest = self.a.jcc(CC_E, 0);
                if xo == 30 {
                    self.a.vfnmadd213sd(0, 1, 2); // -(a*c) + b
                } else {
                    self.a.vfnmsub213sd(0, 1, 2); // -(a*c) - b
                }
                let j = self.a.jmp(0);
                let h = self.a.here();
                self.a.patch(nearest, h);
                neg_done = Some(j);
            }
            if xo == 28 || xo == 30 {
                self.a.vfmsub213sd(0, 1, 2);
            } else {
                self.a.vfmadd213sd(0, 1, 2);
            }
        } else {
            self.a.mulsd(0, Rm::R(1));
            if xo == 28 || xo == 30 {
                self.a.subsd(0, Rm::R(2));
            } else {
                self.a.addsd(0, Rm::R(2));
            }
        }
        if xo >= 30 {
            self.a.mov64_ri(RAX, 0x8000_0000_0000_0000);
            self.a.movq_xr(3, RAX);
            self.a.xorpd(0, Rm::R(3));
        }
        if let Some(j) = neg_done {
            let h = self.a.here();
            self.a.patch(j, h);
        }
        self.fp_fixnan();
        if single {
            self.fp_single();
        }
        self.fp_store(frt, 0);
    }

    fn op63(&mut self, i: u32) {
        let xo5 = (i >> 1) & 31;
        let xo = (i >> 1) & 0x3FF;
        let frt = (i >> 21) & 31;
        let fra = (i >> 16) & 31;
        let frb = (i >> 11) & 31;
        let frc = (i >> 6) & 31;
        let rc = i & 1 != 0;
        let pc = self.pc;
        if matches!(xo5, 18 | 20 | 21 | 22 | 23 | 25 | 26 | 28 | 29 | 30 | 31) {
            match xo5 {
                18 | 20 | 21 | 25 => {
                    self.fp_load(0, fra);
                    match xo5 {
                        18 => self.a.divsd(0, Rm::M(fm(frb))),
                        20 => self.a.subsd(0, Rm::M(fm(frb))),
                        21 => self.a.addsd(0, Rm::M(fm(frb))),
                        _ => self.a.mulsd(0, Rm::M(fm(frc))),
                    }
                    self.fp_fixnan();
                    self.fp_store(frt, 0);
                }
                22 => {
                    self.a.sqrtsd(0, Rm::M(fm(frb)));
                    self.fp_fixnan();
                    self.fp_store(frt, 0);
                }
                23 => {
                    // fsel: frt = (fra >= 0) ? frc : frb
                    self.fp_load(0, fra);
                    self.a.xorpd(1, Rm::R(1));
                    self.a.mov64_rm(RAX, Rm::M(fm(frb)));
                    self.a.mov64_rm(RDX, Rm::M(fm(frc)));
                    self.a.comisd(0, Rm::R(1));
                    self.a.cmov64(CC_AE, RAX, Rm::R(RDX));
                    self.a.mov64_mr(Rm::M(fm(frt)), RAX);
                }
                26 => {
                    self.a.mov64_rr(RDI, R15);
                    self.a.mov_ri(RSI, frt);
                    self.a.mov_ri(RDX, frb);
                    self.call_helper(h_frsqrte as usize);
                }
                _ => self.fmadd(xo5, frt, fra, frb, frc, false),
            }
            if rc {
                self.fp_rc();
            }
            return;
        }
        match xo {
            0 | 32 => {
                // fcmpu / fcmpo
                let crf = frt >> 2;
                self.fp_load(0, fra);
                self.a.mov_ri(RCX, 2);
                self.a.ucomisd(0, Rm::M(fm(frb)));
                let j1 = self.a.jcc(CC_NP, 0);
                self.a.mov_ri(RCX, 1);
                let d1 = self.a.jmp(0);
                let l1 = self.a.here();
                self.a.patch(j1, l1);
                let j2 = self.a.jcc(CC_BE, 0);
                self.a.mov_ri(RCX, 4);
                let d2 = self.a.jmp(0);
                let l2 = self.a.here();
                self.a.patch(j2, l2);
                let j3 = self.a.jcc(CC_AE, 0);
                self.a.mov_ri(RCX, 8);
                let l3 = self.a.here();
                self.a.patch(j3, l3);
                self.a.patch(d1, l3);
                self.a.patch(d2, l3);
                // FPSCR.FPCC
                self.a.mov_rm(RDX, Rm::M(sm(OFF_FPSCR)));
                self.a.alu_i(Alu::And, Rm::R(RDX), !0xF000u32);
                self.a.mov_rr(RAX, RCX);
                self.a.shift_i(Shift::Shl, Rm::R(RAX), 12);
                self.a.alu_rr(Alu::Or, RDX, RAX);
                self.a.mov_mr(Rm::M(sm(OFF_FPSCR)), RDX);
                self.write_cr_nibble(crf);
            }
            12 => {
                // frsp
                self.fp_load(0, frb);
                self.fp_single();
                self.fp_store(frt, 0);
            }
            14 | 15 => {
                // fctiw / fctiwz
                self.a.mov64_rr(RDI, R15);
                self.a.mov_ri(RSI, frt);
                self.a.mov_ri(RDX, frb);
                self.a.mov_ri(RCX, (xo == 15) as u32);
                self.call_helper(h_fctiw as usize);
            }
            40 | 136 | 264 | 72 => {
                // fneg fnabs fabs fmr
                self.a.mov64_rm(RAX, Rm::M(fm(frb)));
                match xo {
                    40 => {
                        self.a.mov64_ri(RDX, 0x8000_0000_0000_0000);
                        self.a.alu64_rr(Alu::Xor, RAX, RDX);
                    }
                    136 => {
                        self.a.mov64_ri(RDX, 0x8000_0000_0000_0000);
                        self.a.alu64_rr(Alu::Or, RAX, RDX);
                    }
                    264 => {
                        self.a.mov64_ri(RDX, 0x7FFF_FFFF_FFFF_FFFF);
                        self.a.alu64_rr(Alu::And, RAX, RDX);
                    }
                    _ => {}
                }
                self.a.mov64_mr(Rm::M(fm(frt)), RAX);
            }
            583 => {
                // mffs
                self.a.mov64_rr(RDI, R15);
                self.call_helper(h_mffs as usize);
                self.a.mov64_mr(Rm::M(fm(frt)), RAX);
            }
            711 => {
                // mtfsf FM, frb
                let fmv = (i >> 17) & 0xFF;
                let mut mask = 0u32;
                for f in 0..8 {
                    if fmv & (0x80 >> f) != 0 {
                        mask |= 0xF << (28 - 4 * f);
                    }
                }
                self.a.mov_rm(RAX, Rm::M(fm(frb)));
                self.a.alu_i(Alu::And, Rm::R(RAX), mask);
                self.a.mov_rm(RDX, Rm::M(sm(OFF_FPSCR)));
                self.a.alu_i(Alu::And, Rm::R(RDX), !mask);
                self.a.alu_rr(Alu::Or, RDX, RAX);
                self.a.mov_mr(Rm::M(sm(OFF_FPSCR)), RDX);
                self.fpscr_sync();
            }
            134 => {
                // mtfsfi crf, imm
                let crf = frt >> 2;
                let imm = (i >> 12) & 0xF;
                let sh = 28 - 4 * crf;
                self.a.mov_rm(RDX, Rm::M(sm(OFF_FPSCR)));
                self.a.alu_i(Alu::And, Rm::R(RDX), !(0xF << sh));
                if imm != 0 {
                    self.a.alu_i(Alu::Or, Rm::R(RDX), imm << sh);
                }
                self.a.mov_mr(Rm::M(sm(OFF_FPSCR)), RDX);
                self.fpscr_sync();
            }
            38 | 70 => {
                // mtfsb1 / mtfsb0
                let bit = 1u32 << (31 - frt);
                self.a.mov_rm(RDX, Rm::M(sm(OFF_FPSCR)));
                if xo == 38 {
                    self.a.alu_i(Alu::Or, Rm::R(RDX), bit);
                } else {
                    self.a.alu_i(Alu::And, Rm::R(RDX), !bit);
                }
                self.a.mov_mr(Rm::M(sm(OFF_FPSCR)), RDX);
                self.fpscr_sync();
            }
            64 => {
                // mcrfs: CR field <- FPSCR field (exception bits in it are cleared)
                let (tf, sf) = (frt >> 2, fra >> 2);
                let sh = 28 - 4 * sf;
                self.a.mov_rm(RCX, Rm::M(sm(OFF_FPSCR)));
                self.a.shift_i(Shift::Shr, Rm::R(RCX), sh as u8);
                self.a.alu_i(Alu::And, Rm::R(RCX), 0xF);
                let clear = (0xF << sh) & 0xF91E_0000u32; // FX OX UX ZX XX VXSNAN VXISI ... (not FEX VX)
                if clear != 0 {
                    self.a.mov_rm(RDX, Rm::M(sm(OFF_FPSCR)));
                    self.a.alu_i(Alu::And, Rm::R(RDX), !clear);
                    self.a.mov_mr(Rm::M(sm(OFF_FPSCR)), RDX);
                }
                self.write_cr_nibble(tf);
            }
            _ => {
                self.exit(EXIT_ILLEGAL, pc);
                self.done = true;
                return;
            }
        }
        if rc && xo != 0 && xo != 32 && xo != 64 {
            self.fp_rc();
        }
    }
}

/// Translate the block at `pc` into code that will live at host address `dest`.
pub fn translate(cpu: &Cpu, pc: u32, nohook: bool, dest: usize) -> Result<Translated, Fault> {
    let mut t = Tr {
        cpu,
        a: Asm::new(),
        dest,
        pc,
        start: pc,
        cold: Vec::new(),
        chains: Vec::new(),
        pcmap: Vec::new(),
        done: false,
        movbe: cpu.features.movbe,
        fma: cpu.features.fma,
    };
    if !nohook && cpu.hooked(pc) {
        t.exit(EXIT_HOOK, pc);
        return Ok(Translated { code: t.a.code, guest_len: 4, chains: Vec::new(), pcmap: Vec::new() });
    }
    let mut size_patch = None;
    if cpu.has_block_hooks() {
        t.a.mov64_rr(RDI, R15);
        t.a.mov_ri(RSI, pc);
        t.a.mov_ri(RDX, 0);
        size_patch = Some(t.a.here() - 4);
        t.call_helper(h_block as usize);
    }
    let mut n = 0;
    while !t.done {
        let insn = match t.insn() {
            Some(w) => w,
            None => {
                // the next instruction is not mapped: let the dispatcher report the fetch fault
                if n == 0 {
                    return Err(Fault::Fetch { pc: t.pc });
                }
                t.end_direct(t.pc);
                break;
            }
        };
        if n > 0 && cpu.hooked(t.pc) {
            t.end_direct(t.pc);
            break;
        }
        t.translate_one(insn);
        t.pc = t.pc.wrapping_add(4);
        n += 1;
        if !t.done && n >= MAX_INSNS {
            t.end_direct(t.pc);
            break;
        }
    }
    let guest_len = t.pc.wrapping_sub(t.start);
    if let Some(p) = size_patch {
        t.a.code[p..p + 4].copy_from_slice(&guest_len.to_le_bytes());
    }
    t.emit_cold();
    Ok(Translated { code: t.a.code, guest_len, chains: t.chains, pcmap: t.pcmap })
}

// ---- helpers called from generated code

fn gptr(st: &State, a: u32) -> *mut u8 {
    (st.membase as usize + a as usize) as *mut u8
}

pub extern "C" fn h_block(st: *mut State, pc: u32, size: u32) {
    unsafe {
        let cpu = &mut *((*st).cpu as *mut Cpu<'static>);
        cpu.run_block_hooks(pc, size);
    }
}

pub extern "C" fn h_lmw(st: *mut State, ea: u32, rt: u32, pc: u32) {
    unsafe {
        let st = &mut *st;
        st.pc = pc;
        let mut a = ea;
        for r in rt..32 {
            st.gpr[r as usize] = u32::from_be(std::ptr::read_unaligned(gptr(st, a) as *const u32));
            a = a.wrapping_add(4);
        }
    }
}

pub extern "C" fn h_stmw(st: *mut State, ea: u32, rs: u32, pc: u32) -> u32 {
    unsafe {
        let st = &mut *st;
        st.pc = pc;
        let mut a = ea;
        for r in rs..32 {
            std::ptr::write_unaligned(gptr(st, a) as *mut u32, st.gpr[r as usize].to_be());
            a = a.wrapping_add(4);
        }
        post_store(st, ea, 4 * (32 - rs), pc)
    }
}

/// lswi / lswx: n bytes into registers from rt on (wrapping to r0), left-justified.
pub extern "C" fn h_lsw(st: *mut State, ea: u32, rt: u32, n: u32, pc: u32) {
    unsafe {
        let st = &mut *st;
        st.pc = pc;
        let mut r = rt;
        let mut i = 0;
        let mut word = 0u32;
        let mut shift = 24i32;
        while i < n {
            let b = *gptr(st, ea.wrapping_add(i)) as u32;
            word |= b << shift;
            shift -= 8;
            i += 1;
            if shift < 0 || i == n {
                st.gpr[r as usize] = word;
                r = (r + 1) & 31;
                word = 0;
                shift = 24;
            }
        }
    }
}

pub extern "C" fn h_stsw(st: *mut State, ea: u32, rs: u32, n: u32, pc: u32) -> u32 {
    unsafe {
        let st = &mut *st;
        st.pc = pc;
        let mut r = rs;
        let mut shift = 24i32;
        for i in 0..n {
            let b = (st.gpr[r as usize] >> shift) as u8;
            *gptr(st, ea.wrapping_add(i)) = b;
            shift -= 8;
            if shift < 0 {
                shift = 24;
                r = (r + 1) & 31;
            }
        }
        if n == 0 {
            return 0;
        }
        post_store(st, ea, n, pc)
    }
}

/// stwcx. succeeded on a page with code: invalidate.
pub extern "C" fn h_code_touched(st: *mut State, ea: u32, size: u32, next_pc: u32) -> u32 {
    unsafe { post_store(&mut *st, ea, size, next_pc.wrapping_sub(4)) }
}

/// After a store by a helper: invalidate overwritten code; 1 if the running block was hit.
pub unsafe fn post_store(st: &mut State, ea: u32, len: u32, cur_pc: u32) -> u32 {
    let cpu = &mut *(st.cpu as *mut Cpu<'static>);
    let end = ea as u64 + len as u64;
    let mut p = (ea & !0xFFF) as u64;
    let mut must_exit = false;
    let mut any = false;
    while p < end {
        let flag = *((st.tables as usize + PF_OFF as usize + (p >> 12) as usize) as *const u8);
        if flag & PF_CODE != 0 {
            any = true;
            if let Some(list) = cpu.page_blocks_at(p as u32) {
                for &i in list {
                    if let Some(b) = cpu.block(i) {
                        if b.valid && (b.pc as u64) < end && b.pc as u64 + b.guest_len as u64 > ea as u64 && b.pc <= cur_pc && cur_pc < b.pc + b.guest_len {
                            must_exit = true;
                        }
                    }
                }
            }
        }
        p += 0x1000;
    }
    if any {
        cpu.invalidate_range(ea, len);
    }
    if must_exit {
        st.pc = cur_pc.wrapping_add(4);
        1
    } else {
        0
    }
}

/// stfs of a double with exponent <= 896: a single denormal (truncated) or zero.
pub extern "C" fn h_tosingle(bits: u64) -> u32 {
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let mut ret = ((bits >> 63) as u32) << 31;
    if exp >= 874 {
        let top23 = ((bits >> 29) & 0x7F_FFFF) as u32;
        ret |= (0x0080_0000 | top23) >> (897 - exp);
    }
    ret
}

pub extern "C" fn h_timebase(_st: *mut State) -> u64 {
    // TBCFG says 1 GHz: nanoseconds
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn fpscr_rounding(fpscr: u32) -> u32 {
    // PPC RN: 0 nearest, 1 zero, 2 +inf, 3 -inf ; MXCSR RC: 0 nearest, 1 -inf, 2 +inf, 3 zero
    match fpscr & 3 {
        0 => 0,
        1 => 3,
        2 => 2,
        _ => 1,
    }
}

#[cfg(target_arch = "x86_64")]
fn set_mxcsr(v: u32) {
    #[allow(deprecated)]
    unsafe {
        std::arch::x86_64::_mm_setcsr(v)
    }
}
#[cfg(target_arch = "x86_64")]
fn get_mxcsr() -> u32 {
    #[allow(deprecated)]
    unsafe {
        std::arch::x86_64::_mm_getcsr()
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn set_mxcsr(_v: u32) {}
#[cfg(not(target_arch = "x86_64"))]
fn get_mxcsr() -> u32 {
    0x1F80
}

/// FPSCR was written: the host rounding mode follows RN.
pub extern "C" fn h_fpscr_sync(st: *mut State) {
    unsafe {
        let st = &mut *st;
        set_mxcsr((get_mxcsr() & !0x603F) | (fpscr_rounding(st.fpscr) << 13));
    }
}

/// FPSCR set from outside (reg_write): same.
pub fn fpscr_written(st: &mut State) {
    set_mxcsr((get_mxcsr() & !0x603F) | (fpscr_rounding(st.fpscr) << 13));
}

/// mffs: fold the host's sticky exception flags into FPSCR.
pub extern "C" fn h_mffs(st: *mut State) -> u64 {
    unsafe {
        let st = &mut *st;
        let m = get_mxcsr();
        let mut f = st.fpscr;
        let before = f;
        if m & 0x01 != 0 {
            f |= 0x2000_0000 | 0x0100_0000; // VX, VXSNAN (some invalid operation)
        }
        if m & 0x04 != 0 {
            f |= 0x0400_0000; // ZX
        }
        if m & 0x08 != 0 {
            f |= 0x1000_0000; // OX
        }
        if m & 0x10 != 0 {
            f |= 0x0800_0000; // UX
        }
        if m & 0x20 != 0 {
            f |= 0x0200_0000; // XX
        }
        if f != before {
            f |= 0x8000_0000; // FX
        }
        st.fpscr = f;
        set_mxcsr(m & !0x3F);
        f as u64
    }
}

/// fctiw / fctiwz: double to int32. As QEMU (which the Unicorn build ran everything on):
/// the result is the sign-extended integer; NaN, infinities and values out of range give
/// 0x80000000 with a zero upper word. (The architecture leaves the upper word undefined and
/// saturates positive overflow to 0x7FFFFFFF.)
pub extern "C" fn h_fctiw(st: *mut State, frt: u32, frb: u32, truncate: u32) {
    unsafe {
        let st = &mut *st;
        let x = f64::from_bits(st.fpr[frb as usize]);
        let bad = 0x8000_0000u64;
        let r: u64 = if x.is_nan() || x.is_infinite() {
            bad
        } else {
            let y = if truncate != 0 {
                x.trunc()
            } else {
                match st.fpscr & 3 {
                    0 => {
                        // round to nearest even
                        let r = x.round();
                        if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
                    }
                    1 => x.trunc(),
                    2 => x.ceil(),
                    _ => x.floor(),
                }
            };
            if !(-2147483648.0..2147483648.0).contains(&y) {
                bad
            } else {
                (y as i32) as i64 as u64
            }
        };
        st.fpr[frt as usize] = r;
    }
}

/// The host's default NaN is negative, PowerPC's positive.
fn fixnan(bits: u64) -> u64 {
    if bits == 0xFFF8_0000_0000_0000 { 0x7FF8_0000_0000_0000 } else { bits }
}

pub extern "C" fn h_fres(st: *mut State, frt: u32, frb: u32) {
    unsafe {
        let st = &mut *st;
        let x = f64::from_bits(st.fpr[frb as usize]);
        st.fpr[frt as usize] = fixnan(((1.0 / x) as f32 as f64).to_bits());
    }
}

/// frsqrte: the exact 1/sqrt(x) in double precision (an estimate is allowed; QEMU does this).
pub extern "C" fn h_frsqrte(st: *mut State, frt: u32, frb: u32) {
    unsafe {
        let st = &mut *st;
        let x = f64::from_bits(st.fpr[frb as usize]);
        st.fpr[frt as usize] = fixnan((1.0 / x.sqrt()).to_bits());
    }
}

#[cfg(test)]
mod tests {
    use super::rot_mask;
    #[test]
    fn masks() {
        assert_eq!(rot_mask(0, 31), 0xFFFF_FFFF);
        assert_eq!(rot_mask(24, 31), 0xFF);
        assert_eq!(rot_mask(0, 7), 0xFF00_0000);
        assert_eq!(rot_mask(28, 3), 0xF000_000F);
        assert_eq!(rot_mask(31, 0), 0x8000_0001);
    }
}
