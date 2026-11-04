//! A small x86-64 instruction encoder: just the forms the translator emits.
//!
//! Operands are 32-bit unless the name says otherwise (`mov64`, `lea64` ...). Memory operands
//! are `[base + index*scale + disp32]`; registers are numbered 0-15 (RAX ... R15), XMM
//! registers 0-15 as well.

#![allow(dead_code)]

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

/// Condition codes (the low nibble of Jcc / SETcc / CMOVcc).
pub const CC_O: u8 = 0;
pub const CC_NO: u8 = 1;
pub const CC_B: u8 = 2; // below / carry
pub const CC_AE: u8 = 3; // above or equal / no carry
pub const CC_E: u8 = 4;
pub const CC_NE: u8 = 5;
pub const CC_BE: u8 = 6;
pub const CC_A: u8 = 7;
pub const CC_S: u8 = 8;
pub const CC_NS: u8 = 9;
pub const CC_P: u8 = 10;
pub const CC_NP: u8 = 11;
pub const CC_L: u8 = 12;
pub const CC_GE: u8 = 13;
pub const CC_LE: u8 = 14;
pub const CC_G: u8 = 15;

#[derive(Clone, Copy, Debug)]
pub struct Mem {
    pub base: u8,
    pub index: Option<u8>,
    pub scale: u8,
    pub disp: i32,
}

pub fn mem(base: u8, disp: i32) -> Mem {
    Mem { base, index: None, scale: 1, disp }
}
pub fn mem_idx(base: u8, index: u8, scale: u8, disp: i32) -> Mem {
    Mem { base, index: Some(index), scale, disp }
}

#[derive(Clone, Copy, Debug)]
pub enum Rm {
    R(u8),
    M(Mem),
}

/// ALU operations (the /digit of the 81 group and the base of the two-operand opcodes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alu {
    Add = 0,
    Or = 1,
    Adc = 2,
    Sbb = 3,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

/// Shift / rotate operations (the /digit of the C1 / D3 group).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shift {
    Rol = 0,
    Ror = 1,
    Shl = 4,
    Shr = 5,
    Sar = 7,
}

pub struct Asm {
    pub code: Vec<u8>,
}

impl Default for Asm {
    fn default() -> Self {
        Self::new()
    }
}

impl Asm {
    pub fn new() -> Self {
        Asm { code: Vec::with_capacity(4096) }
    }
    pub fn len(&self) -> usize {
        self.code.len()
    }
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }
    fn b(&mut self, x: u8) {
        self.code.push(x);
    }
    fn d32(&mut self, x: i32) {
        self.code.extend_from_slice(&x.to_le_bytes());
    }
    fn d64(&mut self, x: u64) {
        self.code.extend_from_slice(&x.to_le_bytes());
    }

    /// REX prefix for a reg/rm pair (emitted when any high register or REX.W is involved).
    fn rex(&mut self, w: bool, reg: u8, rm: &Rm, force: bool) {
        let (x, b) = match rm {
            Rm::R(r) => (0u8, (r >> 3) & 1),
            Rm::M(m) => ((m.index.unwrap_or(0) >> 3) & 1, (m.base >> 3) & 1),
        };
        let r = (reg >> 3) & 1;
        let v = 0x40 | ((w as u8) << 3) | (r << 2) | (x << 1) | b;
        if v != 0x40 || force {
            self.b(v);
        }
    }

    /// ModRM (+ SIB + displacement) for `reg` and `rm`.
    fn modrm(&mut self, reg: u8, rm: &Rm) {
        let reg = (reg & 7) << 3;
        match rm {
            Rm::R(r) => self.b(0xC0 | reg | (r & 7)),
            Rm::M(m) => {
                let base = m.base & 7;
                let need_sib = m.index.is_some() || base == 4;
                let (md, disp8) = if m.disp == 0 && base != 5 {
                    (0u8, false)
                } else if (-128..=127).contains(&m.disp) {
                    (0x40u8, true)
                } else {
                    (0x80u8, false)
                };
                if need_sib {
                    self.b(md | reg | 4);
                    let scale = match m.scale {
                        1 => 0,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => panic!("bad scale"),
                    };
                    let idx = match m.index {
                        Some(i) => {
                            assert!(i != RSP, "rsp cannot be an index");
                            i & 7
                        }
                        None => 4,
                    };
                    self.b((scale << 6) | (idx << 3) | base);
                } else {
                    self.b(md | reg | base);
                }
                if md == 0x40 {
                    self.b(m.disp as i8 as u8);
                } else if md == 0x80 {
                    self.d32(m.disp);
                }
                let _ = disp8;
            }
        }
    }

    /// `[prefix] [REX] opcode... modrm`
    fn op(&mut self, prefix: Option<u8>, w: bool, opcode: &[u8], reg: u8, rm: Rm) {
        if let Some(p) = prefix {
            self.b(p);
        }
        self.rex(w, reg, &rm, false);
        for &o in opcode {
            self.b(o);
        }
        self.modrm(reg, &rm);
    }

    // ---- moves
    /// mov r32, r/m32
    pub fn mov_rm(&mut self, dst: u8, src: Rm) {
        self.op(None, false, &[0x8B], dst, src);
    }
    /// mov r/m32, r32
    pub fn mov_mr(&mut self, dst: Rm, src: u8) {
        self.op(None, false, &[0x89], src, dst);
    }
    pub fn mov_rr(&mut self, dst: u8, src: u8) {
        self.mov_rm(dst, Rm::R(src));
    }
    /// mov r32, imm32
    pub fn mov_ri(&mut self, dst: u8, imm: u32) {
        if dst >= 8 {
            self.b(0x41);
        }
        self.b(0xB8 + (dst & 7));
        self.d32(imm as i32);
    }
    /// mov r/m32, imm32
    pub fn mov_mi(&mut self, dst: Mem, imm: u32) {
        self.op(None, false, &[0xC7], 0, Rm::M(dst));
        self.d32(imm as i32);
    }
    /// mov r64, r/m64
    pub fn mov64_rm(&mut self, dst: u8, src: Rm) {
        self.op(None, true, &[0x8B], dst, src);
    }
    /// mov r/m64, r64
    pub fn mov64_mr(&mut self, dst: Rm, src: u8) {
        self.op(None, true, &[0x89], src, dst);
    }
    pub fn mov64_rr(&mut self, dst: u8, src: u8) {
        self.mov64_rm(dst, Rm::R(src));
    }
    /// mov r64, imm64
    pub fn mov64_ri(&mut self, dst: u8, imm: u64) {
        self.b(0x48 | ((dst >> 3) & 1));
        self.b(0xB8 + (dst & 7));
        self.d64(imm);
    }
    /// mov m8, r8
    pub fn mov8_mr(&mut self, dst: Mem, src: u8) {
        if (4..8).contains(&src) && dst.base < 8 && dst.index.unwrap_or(0) < 8 {
            self.b(0x40);
        }
        self.op(None, false, &[0x88], src, Rm::M(dst));
    }
    /// mov r/m16, r16
    pub fn mov16_mr(&mut self, dst: Mem, src: u8) {
        self.op(Some(0x66), false, &[0x89], src, Rm::M(dst));
    }
    /// movzx r32, r/m8
    pub fn movzx8(&mut self, dst: u8, src: Rm) {
        if let Rm::R(r) = src {
            if (4..8).contains(&r) && dst < 8 {
                self.b(0x40);
            }
        }
        self.op(None, false, &[0x0F, 0xB6], dst, src);
    }
    /// movzx r32, r/m16
    pub fn movzx16(&mut self, dst: u8, src: Rm) {
        self.op(None, false, &[0x0F, 0xB7], dst, src);
    }
    /// movsx r32, r/m8
    pub fn movsx8(&mut self, dst: u8, src: Rm) {
        if let Rm::R(r) = src {
            if (4..8).contains(&r) && dst < 8 {
                self.b(0x40);
            }
        }
        self.op(None, false, &[0x0F, 0xBE], dst, src);
    }
    /// movsx r32, r/m16
    pub fn movsx16(&mut self, dst: u8, src: Rm) {
        self.op(None, false, &[0x0F, 0xBF], dst, src);
    }
    /// lea r32, [mem] (zero-extends into the 64-bit register)
    pub fn lea(&mut self, dst: u8, m: Mem) {
        self.op(None, false, &[0x8D], dst, Rm::M(m));
    }
    pub fn lea64(&mut self, dst: u8, m: Mem) {
        self.op(None, true, &[0x8D], dst, Rm::M(m));
    }

    // ---- arithmetic
    /// op r32, r/m32
    pub fn alu_rm(&mut self, op: Alu, dst: u8, src: Rm) {
        self.op(None, false, &[0x03 + 8 * (op as u8)], dst, src);
    }
    /// op r/m32, r32
    pub fn alu_mr(&mut self, op: Alu, dst: Rm, src: u8) {
        self.op(None, false, &[0x01 + 8 * (op as u8)], src, dst);
    }
    pub fn alu_rr(&mut self, op: Alu, dst: u8, src: u8) {
        self.alu_rm(op, dst, Rm::R(src));
    }
    /// op r/m32, imm32 (imm8 form when it fits)
    pub fn alu_i(&mut self, op: Alu, dst: Rm, imm: u32) {
        let s = imm as i32;
        if (-128..=127).contains(&s) {
            self.op(None, false, &[0x83], op as u8, dst);
            self.b(s as i8 as u8);
        } else {
            self.op(None, false, &[0x81], op as u8, dst);
            self.d32(s);
        }
    }
    /// op r/m64, imm32 (sign-extended)
    pub fn alu64_i(&mut self, op: Alu, dst: Rm, imm: i32) {
        if (-128..=127).contains(&imm) {
            self.op(None, true, &[0x83], op as u8, dst);
            self.b(imm as i8 as u8);
        } else {
            self.op(None, true, &[0x81], op as u8, dst);
            self.d32(imm);
        }
    }
    pub fn alu64_rr(&mut self, op: Alu, dst: u8, src: u8) {
        self.op(None, true, &[0x03 + 8 * (op as u8)], dst, Rm::R(src));
    }
    /// test r/m32, r32
    pub fn test_rr(&mut self, a: u8, b: u8) {
        self.op(None, false, &[0x85], b, Rm::R(a));
    }
    /// test r/m32, imm32
    pub fn test_i(&mut self, dst: Rm, imm: u32) {
        self.op(None, false, &[0xF7], 0, dst);
        self.d32(imm as i32);
    }
    /// test r/m8, imm8 (RAX..RBX low bytes, or memory)
    pub fn test8_i(&mut self, dst: Rm, imm: u8) {
        self.op(None, false, &[0xF6], 0, dst);
        self.b(imm);
    }
    pub fn not(&mut self, dst: Rm) {
        self.op(None, false, &[0xF7], 2, dst);
    }
    pub fn neg(&mut self, dst: Rm) {
        self.op(None, false, &[0xF7], 3, dst);
    }
    /// mul r/m32: edx:eax = eax * src (unsigned)
    pub fn mul(&mut self, src: Rm) {
        self.op(None, false, &[0xF7], 4, src);
    }
    /// imul r/m32: edx:eax = eax * src (signed)
    pub fn imul1(&mut self, src: Rm) {
        self.op(None, false, &[0xF7], 5, src);
    }
    /// div r/m32: eax = edx:eax / src, edx = remainder (unsigned)
    pub fn div(&mut self, src: Rm) {
        self.op(None, false, &[0xF7], 6, src);
    }
    pub fn idiv(&mut self, src: Rm) {
        self.op(None, false, &[0xF7], 7, src);
    }
    /// imul r32, r/m32
    pub fn imul_rm(&mut self, dst: u8, src: Rm) {
        self.op(None, false, &[0x0F, 0xAF], dst, src);
    }
    /// imul r32, r/m32, imm32
    pub fn imul_rmi(&mut self, dst: u8, src: Rm, imm: i32) {
        self.op(None, false, &[0x69], dst, src);
        self.d32(imm);
    }
    /// shift/rotate r/m32, imm8
    pub fn shift_i(&mut self, kind: Shift, dst: Rm, n: u8) {
        if n == 1 {
            self.op(None, false, &[0xD1], kind as u8, dst);
        } else {
            self.op(None, false, &[0xC1], kind as u8, dst);
            self.b(n);
        }
    }
    /// shift/rotate r/m32, cl
    pub fn shift_cl(&mut self, kind: Shift, dst: Rm) {
        self.op(None, false, &[0xD3], kind as u8, dst);
    }
    /// shift/rotate r/m64, imm8
    pub fn shift64_i(&mut self, kind: Shift, dst: Rm, n: u8) {
        self.op(None, true, &[0xC1], kind as u8, dst);
        self.b(n);
    }
    /// rotate r/m16, imm8
    pub fn shift16_i(&mut self, kind: Shift, dst: Rm, n: u8) {
        self.op(Some(0x66), false, &[0xC1], kind as u8, dst);
        self.b(n);
    }
    pub fn bswap(&mut self, r: u8) {
        if r >= 8 {
            self.b(0x41);
        }
        self.b(0x0F);
        self.b(0xC8 + (r & 7));
    }
    pub fn bswap64(&mut self, r: u8) {
        self.b(0x48 | ((r >> 3) & 1));
        self.b(0x0F);
        self.b(0xC8 + (r & 7));
    }
    /// movbe r64, m64
    pub fn movbe64_rm(&mut self, dst: u8, src: Mem) {
        self.op(None, true, &[0x0F, 0x38, 0xF0], dst, Rm::M(src));
    }
    /// movbe m64, r64
    pub fn movbe64_mr(&mut self, dst: Mem, src: u8) {
        self.op(None, true, &[0x0F, 0x38, 0xF1], src, Rm::M(dst));
    }
    /// cmp r/m8, imm8
    pub fn cmp8_i(&mut self, dst: Rm, imm: u8) {
        self.op(None, false, &[0x80], 7, dst);
        self.b(imm);
    }
    /// cmovcc r64, r/m64
    pub fn cmov64(&mut self, cc: u8, dst: u8, src: Rm) {
        self.op(None, true, &[0x0F, 0x40 + cc], dst, src);
    }
    /// test r/m64, r64
    pub fn test64_rr(&mut self, a: u8, b: u8) {
        self.op(None, true, &[0x85], b, Rm::R(a));
    }
    /// movbe r32, m32
    pub fn movbe_rm(&mut self, dst: u8, src: Mem) {
        self.op(None, false, &[0x0F, 0x38, 0xF0], dst, Rm::M(src));
    }
    /// movbe m32, r32
    pub fn movbe_mr(&mut self, dst: Mem, src: u8) {
        self.op(None, false, &[0x0F, 0x38, 0xF1], src, Rm::M(dst));
    }
    /// movbe r16, m16
    pub fn movbe16_rm(&mut self, dst: u8, src: Mem) {
        self.op(Some(0x66), false, &[0x0F, 0x38, 0xF0], dst, Rm::M(src));
    }
    /// movbe m16, r16
    pub fn movbe16_mr(&mut self, dst: Mem, src: u8) {
        self.op(Some(0x66), false, &[0x0F, 0x38, 0xF1], src, Rm::M(dst));
    }
    /// setcc r8 (a REX prefix selects spl/bpl/sil/dil instead of ah/ch/dh/bh)
    pub fn setcc(&mut self, cc: u8, dst: u8) {
        if (4..8).contains(&dst) {
            self.b(0x40);
        }
        self.op(None, false, &[0x0F, 0x90 + cc], 0, Rm::R(dst));
    }
    /// cmovcc r32, r/m32
    pub fn cmov(&mut self, cc: u8, dst: u8, src: Rm) {
        self.op(None, false, &[0x0F, 0x40 + cc], dst, src);
    }
    /// bt r/m32, imm8
    pub fn bt_i(&mut self, dst: Rm, bit: u8) {
        self.op(None, false, &[0x0F, 0xBA], 4, dst);
        self.b(bit);
    }
    /// bt r/m32, r32
    pub fn bt_r(&mut self, dst: Rm, bit: u8) {
        self.op(None, false, &[0x0F, 0xA3], bit, dst);
    }
    /// bsr r32, r/m32
    pub fn bsr(&mut self, dst: u8, src: Rm) {
        self.op(None, false, &[0x0F, 0xBD], dst, src);
    }
    pub fn cdq(&mut self) {
        self.b(0x99);
    }
    pub fn push(&mut self, r: u8) {
        if r >= 8 {
            self.b(0x41);
        }
        self.b(0x50 + (r & 7));
    }
    pub fn pop(&mut self, r: u8) {
        if r >= 8 {
            self.b(0x41);
        }
        self.b(0x58 + (r & 7));
    }
    /// lock cmpxchg m32, r32 (eax = expected; ZF set on success)
    pub fn lock_cmpxchg(&mut self, dst: Mem, src: u8) {
        self.b(0xF0);
        self.op(None, false, &[0x0F, 0xB1], src, Rm::M(dst));
    }
    pub fn mfence(&mut self) {
        self.code.extend_from_slice(&[0x0F, 0xAE, 0xF0]);
    }
    pub fn stmxcsr(&mut self, m: Mem) {
        self.op(None, false, &[0x0F, 0xAE], 3, Rm::M(m));
    }
    pub fn ldmxcsr(&mut self, m: Mem) {
        self.op(None, false, &[0x0F, 0xAE], 2, Rm::M(m));
    }

    // ---- control flow
    /// jcc rel32; returns the offset of the rel32 field for `patch`.
    pub fn jcc(&mut self, cc: u8, target: i32) -> usize {
        self.b(0x0F);
        self.b(0x80 + cc);
        let at = self.code.len();
        self.d32(target);
        at
    }
    /// jmp rel32; returns the offset of the rel32 field.
    pub fn jmp(&mut self, target: i32) -> usize {
        self.b(0xE9);
        let at = self.code.len();
        self.d32(target);
        at
    }
    /// call rel32
    pub fn call(&mut self, target: i32) -> usize {
        self.b(0xE8);
        let at = self.code.len();
        self.d32(target);
        at
    }
    /// jmp r/m64
    pub fn jmp_rm(&mut self, t: Rm) {
        self.op(None, false, &[0xFF], 4, t);
    }
    /// call r/m64
    pub fn call_rm(&mut self, t: Rm) {
        self.op(None, false, &[0xFF], 2, t);
    }
    pub fn ret(&mut self) {
        self.b(0xC3);
    }
    pub fn ud2(&mut self) {
        self.b(0x0F);
        self.b(0x0B);
    }
    /// Set a rel32 at `at` so that it jumps to `target` (an offset in this buffer).
    pub fn patch(&mut self, at: usize, target: usize) {
        let rel = target as i64 - (at as i64 + 4);
        self.code[at..at + 4].copy_from_slice(&(rel as i32).to_le_bytes());
    }
    /// A label: the current offset.
    pub fn here(&self) -> usize {
        self.code.len()
    }
    /// jcc to a label already emitted.
    pub fn jcc_back(&mut self, cc: u8, label: usize) {
        let at = self.jcc(cc, 0);
        self.patch(at, label);
    }
    pub fn jmp_back(&mut self, label: usize) {
        let at = self.jmp(0);
        self.patch(at, label);
    }

    // ---- SSE2 scalar double
    fn sse(&mut self, prefix: Option<u8>, w: bool, opcode: u8, reg: u8, rm: Rm) {
        self.op(prefix, w, &[0x0F, opcode], reg, rm);
    }
    /// movsd xmm, m64
    pub fn movsd_rm(&mut self, x: u8, m: Mem) {
        self.sse(Some(0xF2), false, 0x10, x, Rm::M(m));
    }
    /// movsd m64, xmm
    pub fn movsd_mr(&mut self, m: Mem, x: u8) {
        self.sse(Some(0xF2), false, 0x11, x, Rm::M(m));
    }
    /// movss xmm, m32
    pub fn movss_rm(&mut self, x: u8, m: Mem) {
        self.sse(Some(0xF3), false, 0x10, x, Rm::M(m));
    }
    pub fn movss_mr(&mut self, m: Mem, x: u8) {
        self.sse(Some(0xF3), false, 0x11, x, Rm::M(m));
    }
    /// movapd xmm, xmm
    pub fn movapd(&mut self, dst: u8, src: u8) {
        self.sse(Some(0x66), false, 0x28, dst, Rm::R(src));
    }
    pub fn addsd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x58, dst, src);
    }
    pub fn mulsd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x59, dst, src);
    }
    pub fn subsd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x5C, dst, src);
    }
    pub fn divsd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x5E, dst, src);
    }
    pub fn sqrtsd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x51, dst, src);
    }
    pub fn ucomisd(&mut self, a: u8, b: Rm) {
        self.sse(Some(0x66), false, 0x2E, a, b);
    }
    pub fn comisd(&mut self, a: u8, b: Rm) {
        self.sse(Some(0x66), false, 0x2F, a, b);
    }
    /// cvtsd2ss xmm, xmm/m64
    pub fn cvtsd2ss(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x5A, dst, src);
    }
    /// cvtss2sd xmm, xmm/m32
    pub fn cvtss2sd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF3), false, 0x5A, dst, src);
    }
    /// cvttsd2si r32, xmm/m64 (truncate)
    pub fn cvttsd2si(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x2C, dst, src);
    }
    /// cvtsd2si r32, xmm/m64 (current rounding mode)
    pub fn cvtsd2si(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x2D, dst, src);
    }
    /// cvttsd2si r64, xmm/m64
    pub fn cvttsd2si64(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), true, 0x2C, dst, src);
    }
    pub fn cvtsd2si64(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), true, 0x2D, dst, src);
    }
    /// cvtsi2sd xmm, r/m32
    pub fn cvtsi2sd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), false, 0x2A, dst, src);
    }
    /// cvtsi2sd xmm, r/m64
    pub fn cvtsi2sd64(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0xF2), true, 0x2A, dst, src);
    }
    pub fn xorpd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0x66), false, 0x57, dst, src);
    }
    pub fn andpd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0x66), false, 0x54, dst, src);
    }
    pub fn andnpd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0x66), false, 0x55, dst, src);
    }
    pub fn orpd(&mut self, dst: u8, src: Rm) {
        self.sse(Some(0x66), false, 0x56, dst, src);
    }
    /// movd xmm, r32
    pub fn movd_xr(&mut self, x: u8, r: u8) {
        self.sse(Some(0x66), false, 0x6E, x, Rm::R(r));
    }
    /// movd r32, xmm
    pub fn movd_rx(&mut self, r: u8, x: u8) {
        self.sse(Some(0x66), false, 0x7E, x, Rm::R(r));
    }
    /// movq xmm, r64
    pub fn movq_xr(&mut self, x: u8, r: u8) {
        self.sse(Some(0x66), true, 0x6E, x, Rm::R(r));
    }
    /// movq r64, xmm
    pub fn movq_rx(&mut self, r: u8, x: u8) {
        self.sse(Some(0x66), true, 0x7E, x, Rm::R(r));
    }
    /// VEX-encoded FMA: op xmm1, xmm2, xmm3 (213 forms: xmm1 = xmm2*xmm1 +/- xmm3 etc.)
    fn fma(&mut self, opcode: u8, dst: u8, src2: u8, src3: u8) {
        // C4 [R X B m-mmmm] [W vvvv L pp] op modrm ; map 0F38, W1, pp=66
        self.b(0xC4);
        let r = ((dst >> 3) & 1) ^ 1;
        let b = ((src3 >> 3) & 1) ^ 1;
        self.b((r << 7) | (1 << 6) | (b << 5) | 0b00010);
        self.b((1 << 7) | (((!src2) & 0xF) << 3) | 0b01);
        self.b(opcode);
        self.b(0xC0 | ((dst & 7) << 3) | (src3 & 7));
    }
    /// vfmadd213sd dst, a, b: dst = a*dst + b
    pub fn vfmadd213sd(&mut self, dst: u8, a: u8, b: u8) {
        self.fma(0xA9, dst, a, b);
    }
    /// vfmsub213sd dst, a, b: dst = a*dst - b
    pub fn vfmsub213sd(&mut self, dst: u8, a: u8, b: u8) {
        self.fma(0xAB, dst, a, b);
    }
    /// vfnmadd213sd dst, a, b: dst = -(a*dst) + b
    pub fn vfnmadd213sd(&mut self, dst: u8, a: u8, b: u8) {
        self.fma(0xAD, dst, a, b);
    }
    /// vfnmsub213sd dst, a, b: dst = -(a*dst) - b
    pub fn vfnmsub213sd(&mut self, dst: u8, a: u8, b: u8) {
        self.fma(0xAF, dst, a, b);
    }
}

/// Does this CPU have MOVBE / FMA / LZCNT?
pub struct HostFeatures {
    pub movbe: bool,
    pub fma: bool,
}

pub fn host_features() -> HostFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        let r = std::arch::x86_64::__cpuid(1);
        HostFeatures { movbe: r.ecx & (1 << 22) != 0, fma: r.ecx & (1 << 12) != 0 }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        HostFeatures { movbe: false, fma: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(f: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut a = Asm::new();
        f(&mut a);
        a.code
    }

    #[test]
    fn encodings() {
        assert_eq!(enc(|a| a.mov_rm(RAX, Rm::M(mem(R15, 8)))), vec![0x41, 0x8B, 0x47, 0x08]);
        assert_eq!(enc(|a| a.mov_mr(Rm::M(mem(R15, 0x100)), RCX)), vec![0x41, 0x89, 0x8F, 0x00, 0x01, 0x00, 0x00]);
        assert_eq!(enc(|a| a.mov_rm(RAX, Rm::M(mem_idx(R14, RCX, 1, 0)))), vec![0x41, 0x8B, 0x04, 0x0E]);
        assert_eq!(enc(|a| a.mov_rm(RDX, Rm::M(mem(R13, 0)))), vec![0x41, 0x8B, 0x55, 0x00]);
        assert_eq!(enc(|a| a.mov_rm(RDX, Rm::M(mem(RSP, 0)))), vec![0x8B, 0x14, 0x24]);
        assert_eq!(enc(|a| a.mov_rm(RDX, Rm::M(mem(R12, 4)))), vec![0x41, 0x8B, 0x54, 0x24, 0x04]);
        assert_eq!(enc(|a| a.alu_i(Alu::Add, Rm::R(RAX), 5)), vec![0x83, 0xC0, 0x05]);
        assert_eq!(enc(|a| a.alu_i(Alu::And, Rm::R(RCX), 0x0FFF_FFFF)), vec![0x81, 0xE1, 0xFF, 0xFF, 0xFF, 0x0F]);
        assert_eq!(enc(|a| a.bswap(RAX)), vec![0x0F, 0xC8]);
        assert_eq!(enc(|a| a.bswap(R8)), vec![0x41, 0x0F, 0xC8]);
        assert_eq!(enc(|a| a.movbe_rm(RAX, mem_idx(R14, RCX, 1, 0))), vec![0x41, 0x0F, 0x38, 0xF0, 0x04, 0x0E]);
        assert_eq!(enc(|a| a.setcc(CC_E, RDX)), vec![0x0F, 0x94, 0xC2]);
        assert_eq!(enc(|a| a.setcc(CC_NE, RDI)), vec![0x40, 0x0F, 0x95, 0xC7]);
        assert_eq!(enc(|a| a.setcc(CC_NE, R8)), vec![0x41, 0x0F, 0x95, 0xC0]);
        assert_eq!(enc(|a| a.shift_i(Shift::Rol, Rm::R(RAX), 5)), vec![0xC1, 0xC0, 0x05]);
        assert_eq!(enc(|a| a.mov64_ri(RAX, 0x1122334455667788)), vec![0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!(enc(|a| a.call_rm(Rm::R(RAX))), vec![0xFF, 0xD0]);
        assert_eq!(enc(|a| a.jmp_rm(Rm::M(mem_idx(R13, RDX, 1, 8)))), vec![0x41, 0xFF, 0x64, 0x15, 0x08]);
        assert_eq!(enc(|a| a.movsd_rm(0, mem(R15, 0x80))), vec![0xF2, 0x41, 0x0F, 0x10, 0x87, 0x80, 0x00, 0x00, 0x00]);
        assert_eq!(enc(|a| a.addsd(0, Rm::R(1))), vec![0xF2, 0x0F, 0x58, 0xC1]);
        assert_eq!(enc(|a| a.vfmadd213sd(0, 1, 2)), vec![0xC4, 0xE2, 0xF1, 0xA9, 0xC2]);
        assert_eq!(enc(|a| a.lock_cmpxchg(mem_idx(R14, RCX, 1, 0), RDX)), vec![0xF0, 0x41, 0x0F, 0xB1, 0x14, 0x0E]);
        assert_eq!(enc(|a| a.cvttsd2si(RAX, Rm::R(0))), vec![0xF2, 0x0F, 0x2C, 0xC0]);
        assert_eq!(enc(|a| a.movq_rx(RAX, 0)), vec![0x66, 0x48, 0x0F, 0x7E, 0xC0]);
        assert_eq!(enc(|a| a.movzx16(RAX, Rm::M(mem_idx(R14, RCX, 1, 0)))), vec![0x41, 0x0F, 0xB7, 0x04, 0x0E]);
        assert_eq!(enc(|a| a.shift16_i(Shift::Ror, Rm::R(RAX), 8)), vec![0x66, 0xC1, 0xC8, 0x08]);
        assert_eq!(enc(|a| a.mov16_mr(mem_idx(R14, RCX, 1, 0), RAX)), vec![0x66, 0x41, 0x89, 0x04, 0x0E]);
        assert_eq!(enc(|a| a.bt_i(Rm::M(mem(R15, 0x18c)), 29)), vec![0x41, 0x0F, 0xBA, 0xA7, 0x8C, 0x01, 0x00, 0x00, 0x1D]);
    }
}
