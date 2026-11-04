//! Differential test of the JIT against Unicorn (QEMU's TCG): random instruction sequences
//! on random register and memory state, run on both CPUs, every register and the data page
//! compared.
//!
//!     cargo run --release --example fuzz -- [ITERATIONS] [SEED] [-v]
//!
//! Prints each mismatch with the code (hex words for a disassembler) and the registers
//! that differ; exits 1 if there were any.

use ppc32_jit::{self as jit, Reg};
use unicorn_engine::{Arch, Mode, Permission, RegisterPPC, Unicorn};

const CODE: u32 = 0x10000;
const DATA: u32 = 0x18000;
const DATA_SIZE: u32 = 0x1000;
const END_TRAP: u32 = 0x7FE0_0008; // tw 31,0,0

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn chance(&mut self, pct: u32) -> bool {
        self.below(100) < pct
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u32) as usize]
    }
    fn value(&mut self) -> u32 {
        // a mix of interesting integers
        match self.below(8) {
            0 => 0,
            1 => 0xFFFF_FFFF,
            2 => 0x8000_0000,
            3 => 0x7FFF_FFFF,
            4 => self.below(256),
            5 => (self.below(256) as i32 - 128) as u32,
            _ => self.next(),
        }
    }
    fn double(&mut self) -> u64 {
        match self.below(12) {
            0 => 0.0f64.to_bits(),
            1 => (-0.0f64).to_bits(),
            2 => f64::INFINITY.to_bits(),
            3 => f64::NEG_INFINITY.to_bits(),
            4 => f64::NAN.to_bits(),
            5 => 1.0f64.to_bits(),
            6 => (self.next() as f64 / 1000.0).to_bits(),
            7 => ((self.next() as i32) as f64).to_bits(),
            8 => (self.next() as f64 * 1e300).to_bits(),
            9 => (self.next() as f64 * 1e-310).to_bits(),
            10 => ((self.next() as f64 - 2e9) / 7.0).to_bits(),
            _ => (self.next() as u64) << 32 | self.next() as u64,
        }
    }
}

fn d(op: u32, rt: u32, ra: u32, imm: u32) -> u32 {
    (op << 26) | (rt << 21) | (ra << 16) | (imm & 0xFFFF)
}
fn x(rt: u32, ra: u32, rb: u32, xo: u32, rc: u32) -> u32 {
    (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc
}
fn xo(rt: u32, ra: u32, rb: u32, oe: u32, xo9: u32, rc: u32) -> u32 {
    (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (oe << 10) | (xo9 << 1) | rc
}
fn m(op: u32, rs: u32, ra: u32, sh: u32, mb: u32, me: u32, rc: u32) -> u32 {
    (op << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1) | rc
}
fn a(op: u32, frt: u32, fra: u32, frb: u32, frc: u32, xo5: u32, rc: u32) -> u32 {
    (op << 26) | (frt << 21) | (fra << 16) | (frb << 11) | (frc << 6) | (xo5 << 1) | rc
}
fn fx(op: u32, frt: u32, fra: u32, frb: u32, xo10: u32, rc: u32) -> u32 {
    (op << 26) | (frt << 21) | (fra << 16) | (frb << 11) | (xo10 << 1) | rc
}

/// Registers 1-5 hold data addresses, 6-7 small offsets, 8 a shift count.
fn addr_reg(r: &mut Rng) -> u32 {
    1 + r.below(5)
}
fn off_reg(r: &mut Rng) -> u32 {
    6 + r.below(2)
}
fn any_reg(r: &mut Rng) -> u32 {
    r.below(32)
}
/// A target register that is not an address register (so later accesses stay in range).
fn dst_reg(r: &mut Rng) -> u32 {
    let v = r.below(24);
    if v == 0 { 0 } else { v + 8 }
}
fn disp(r: &mut Rng) -> u32 {
    ((r.below(400) as i32) - 200) as u32
}

/// One random instruction; `end` is the address of the final trap (branch target).
fn gen(r: &mut Rng, pc: u32, end: u32, out: &mut Vec<u32>) {
    let rc = r.below(2);
    match r.below(16) {
        0 | 1 => {
            // XO arithmetic
            let xo9 = r.pick(&[266, 10, 138, 202, 234, 40, 8, 136, 200, 232, 104, 235, 75, 11, 491, 459]);
            // OE is an invalid form on mulhw/mulhwu; rb must be 0 on the one-operand forms
            let oe = if r.chance(30) && xo9 != 75 && xo9 != 11 { 1 } else { 0 };
            let rb = if matches!(xo9, 202 | 234 | 200 | 232 | 104) { 0 } else { any_reg(r) };
            out.push(xo(dst_reg(r), any_reg(r), rb, oe, xo9, rc));
        }
        2 => {
            // D-form arithmetic / logical immediates / compares
            let op = r.pick(&[14, 15, 12, 13, 7, 8, 10, 11, 24, 25, 26, 27, 28, 29]);
            let imm = r.value() & 0xFFFF;
            match op {
                10 | 11 => out.push(d(op, r.below(8) << 2, any_reg(r), imm)),
                24..=29 => out.push(d(op, any_reg(r), dst_reg(r), imm)),
                _ => out.push(d(op, dst_reg(r), any_reg(r), imm)),
            }
        }
        3 => {
            // X logical / shifts / compares
            let xov = r.pick(&[28, 60, 124, 284, 316, 412, 444, 476, 922, 954, 26, 24, 536, 792, 824, 0, 32]);
            match xov {
                0 | 32 => out.push(x(r.below(8) << 2, any_reg(r), any_reg(r), xov, 0)),
                824 => out.push(x(any_reg(r), dst_reg(r), r.below(32), xov, rc)),
                24 | 536 | 792 => out.push(x(any_reg(r), dst_reg(r), if r.chance(50) { 8 } else { any_reg(r) }, xov, rc)),
                922 | 954 | 26 => out.push(x(any_reg(r), dst_reg(r), 0, xov, rc)),
                _ => out.push(x(any_reg(r), dst_reg(r), any_reg(r), xov, rc)),
            }
        }
        4 => {
            // rotates
            let op = r.pick(&[20, 21, 23]);
            let sh = if op == 23 { if r.chance(50) { 8 } else { any_reg(r) } } else { r.below(32) };
            out.push(m(op, any_reg(r), dst_reg(r), sh, r.below(32), r.below(32), rc));
        }
        5 | 6 => {
            // D-form loads and stores
            let op = r.pick(&[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55]);
            let ra = addr_reg(r);
            let mut rt = dst_reg(r);
            if op & 1 != 0 && rt == 0 {
                rt = 9; // no update forms with ra = 0 / rt = ra
            }
            if op >= 48 {
                rt = r.below(32);
            }
            if op & 1 != 0 && op < 48 {
                // update forms change the address register: keep it in the window
                out.push(d(op, rt, ra, (r.below(32) * 4) as u32));
            } else {
                out.push(d(op, rt, ra, disp(r)));
            }
        }
        7 => {
            // X-form loads and stores (rb is a small offset register)
            let xov = r.pick(&[23, 87, 279, 343, 534, 790, 151, 215, 407, 662, 918, 535, 599, 663, 727, 983, 55, 119, 311, 375, 183, 247, 439, 567, 631, 695, 759]);
            let ra = addr_reg(r);
            let rt = if xov >= 535 && xov != 662 && xov != 790 && xov != 918 { r.below(32) } else { dst_reg(r).max(9) };
            out.push(x(rt, ra, off_reg(r), xov, 0));
        }
        8 => {
            // multiple / string / dcbz / lwarx-stwcx.
            match r.below(6) {
                0 => out.push(d(46, 20 + r.below(10), addr_reg(r), (r.below(16) * 4) as u32)),
                1 => out.push(d(47, 20 + r.below(10), addr_reg(r), (r.below(16) * 4) as u32)),
                2 => out.push(x(20 + r.below(6), addr_reg(r), 1 + r.below(16), 597, 0)),
                3 => out.push(x(20 + r.below(6), addr_reg(r), 1 + r.below(16), 725, 0)),
                4 => out.push(x(0, addr_reg(r), off_reg(r), 1014, 0)),
                _ => {
                    let ra = addr_reg(r);
                    let rb = off_reg(r);
                    out.push(x(dst_reg(r).max(9), ra, rb, 20, 0));
                    if r.chance(70) {
                        out.push(x(any_reg(r), ra, rb, 150, 1));
                    } else {
                        out.push(x(any_reg(r), ra, 6 + (rb & 1) ^ 1, 150, 1));
                    }
                }
            }
        }
        9 => {
            // CR and SPR moves
            match r.below(8) {
                0 => {
                    let xov = r.pick(&[33, 129, 193, 225, 257, 289, 417, 449]);
                    out.push((19 << 26) | (r.below(32) << 21) | (r.below(32) << 16) | (r.below(32) << 11) | (xov << 1));
                }
                1 => out.push((19 << 26) | ((r.below(8) << 2) << 21) | ((r.below(8) << 2) << 16)), // mcrf
                2 => out.push(x(dst_reg(r), 0, 0, 19, 0)),                                           // mfcr
                3 => out.push((31 << 26) | (any_reg(r) << 21) | (r.below(256) << 12) | (144 << 1)),  // mtcrf
                4 => out.push(x(r.below(8) << 2, 0, 0, 512, 0)),                                     // mcrxr
                5 => {
                    let spr = r.pick(&[1u32, 8, 9]);
                    out.push(x(dst_reg(r), spr & 31, spr >> 5, 339, 0)); // mfspr
                }
                // (no mtctr/mtlr with a random value: a later bcctr/bclr into the code
                // page's random data could loop for ever on the JIT, which has no timeout)
                _ => out.push(x(any_reg(r), 1, 0, 467, 0)), // mtxer
            }
        }
        10 => {
            // branches: forward to the next instruction / the end, or indirect to the end
            let bo = r.pick(&[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
            let bi = r.below(32);
            let lk = r.below(2);
            match r.below(4) {
                0 => {
                    // (no LK either: a not-taken bcl leaves an LR a later bclr loops on)
                    let bd = end.wrapping_sub(pc);
                    out.push((16 << 26) | (bo << 21) | (bi << 16) | (bd & 0xFFFC));
                }
                1 => {
                    // (no LK here or on the indirect forms: a later bclr to the LR this
                    // sets would loop for ever)
                    out.push((16 << 26) | (bo << 21) | (bi << 16) | 8);
                    out.push(d(14, dst_reg(r).max(9), 0, r.value() & 0xFFFF)); // skipped or not
                }
                2 => {
                    let li = end.wrapping_sub(pc);
                    out.push((18 << 26) | (li & 0x03FF_FFFC) | lk);
                }
                _ => {
                    // bclr (LR = end) or bcctr always (CTR = end)
                    if r.chance(50) {
                        out.push((19 << 26) | (bo << 21) | (bi << 16) | (16 << 1));
                    } else {
                        out.push((19 << 26) | (20 << 21) | (528 << 1));
                    }
                }
            }
        }
        11 => {
            // traps
            if r.chance(50) {
                out.push(d(3, r.below(32), any_reg(r), r.value() & 0xFFFF));
            } else {
                out.push(x(r.below(32), any_reg(r), any_reg(r), 4, 0));
            }
        }
        12 | 13 | 14 => {
            // floating point
            match r.below(5) {
                0 | 1 => {
                    let op = r.pick(&[59, 63]);
                    let xo5 = if op == 59 { r.pick(&[18, 20, 21, 22, 24, 25, 28, 29, 30, 31]) } else { r.pick(&[18, 20, 21, 22, 23, 25, 26, 28, 29, 30, 31]) };
                    // the unused operand fields must be 0 (invalid forms are illegal in QEMU)
                    let (fra, frb, frc) = match xo5 {
                        18 | 20 | 21 => (r.below(32), r.below(32), 0),
                        25 => (r.below(32), 0, r.below(32)),
                        22 | 24 | 26 => (0, r.below(32), 0),
                        _ => (r.below(32), r.below(32), r.below(32)),
                    };
                    out.push(a(op, r.below(32), fra, frb, frc, xo5, 0));
                }
                2 => {
                    let xov = r.pick(&[0, 32, 12, 14, 15, 40, 72, 136, 264]);
                    let frt = if xov == 0 || xov == 32 { r.below(8) << 2 } else { r.below(32) };
                    let fra = if xov == 0 || xov == 32 { r.below(32) } else { 0 };
                    out.push(fx(63, frt, fra, r.below(32), xov, 0));
                }
                3 => {
                    // mffs / mtfsf / mtfsfi / mtfsb0 / mtfsb1 / mcrfs
                    // (no mffs / mcrfs: they expose the exception bits QEMU keeps per operation)
                    match r.below(4) {
                        0 => out.push((63 << 26) | (r.below(256) << 17) | (r.below(32) << 11) | (711 << 1)),
                        1 => out.push((63 << 26) | ((r.below(8) << 2) << 21) | (r.below(16) << 12) | (134 << 1)),
                        2 => out.push(fx(63, r.below(32), 0, 0, 70, 0)),
                        _ => out.push(fx(63, r.below(32), 0, 0, 38, 0)),
                    }
                }
                _ => {
                    let op = r.pick(&[48, 50, 52, 54]);
                    out.push(d(op, r.below(32), addr_reg(r), disp(r) & !7));
                }
            }
        }
        _ => {
            // a plain mr / li so sequences have data flow
            if r.chance(50) {
                out.push(x(any_reg(r), dst_reg(r), any_reg(r), 444, rc));
            } else {
                out.push(d(14, dst_reg(r), 0, r.value() & 0xFFFF));
            }
        }
    }
}

struct Init {
    gpr: [u32; 32],
    fpr: [u64; 32],
    cr: u32,
    xer: u32,
    fpscr: u32,
    data: Vec<u8>,
}

fn init_state(r: &mut Rng, end: u32) -> Init {
    let mut gpr = [0u32; 32];
    for g in gpr.iter_mut() {
        *g = r.value();
    }
    let xer = r.below(17) | if r.chance(30) { r.next() & 0xE000_0000 } else { 0 };
    gpr[0] = xer;
    // a quarter of the sequences store into the upper part of the code page: it carries
    // the "has translated code" flag, so stores take the JIT's slow path
    let in_code = r.chance(25);
    for i in 1..6 {
        gpr[i] = if in_code { CODE + 0x800 + r.below(0x600 / 4) * 4 } else { DATA + 256 + r.below((DATA_SIZE - 512) / 4) * 4 };
        if r.chance(20) {
            gpr[i] += r.below(4); // misaligned too
        }
    }
    gpr[6] = r.below(200);
    gpr[7] = r.below(200);
    gpr[8] = r.below(64);
    let mut fpr = [0u64; 32];
    for f in fpr.iter_mut() {
        *f = r.double();
    }
    // no SO/OV/CA: Unicorn's reg_write(XER) does not split them into the fields its
    // instructions use, so a preset bit would be seen by the JIT only
    // RN is 0: Unicorn's reg_write(FPSCR) does not set its rounding mode from it (the
    // sequences change RN with mtfsf etc., which both honour)
    let fpscr = 0;
    let mut data = vec![0u8; DATA_SIZE as usize];
    for b in data.iter_mut() {
        *b = r.next() as u8;
    }
    let _ = end;
    Init { gpr, fpr, cr: r.next(), xer, fpscr, data }
}

fn main() {
    std::process::exit(run(std::env::args().collect()));
}

/// The fuzzer proper: exit status 0 when every sequence agreed.
#[allow(dead_code)]
pub fn run(args: Vec<String>) -> i32 {
    let iters: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let verbose = args.iter().any(|a| a == "-v");
    // -x WORD...: run this one sequence (hex instruction words) instead of random ones
    let fixed: Option<Vec<u32>> = args.iter().position(|a| a == "-x").map(|p| args[p + 1..].iter().take_while(|a| *a != "-f").filter_map(|w| u32::from_str_radix(w, 16).ok()).collect());
    // -f N=HEX ...: preset FPRs (with -x)
    // -c HEX: preset CR (with -x)
    let cr_preset: Option<u32> = args.iter().position(|a| a == "-c").and_then(|p| args.get(p + 1)).and_then(|v| u32::from_str_radix(v, 16).ok());
    let fpr_presets: Vec<(usize, u64)> = args.iter().position(|a| a == "-f").map(|p| args[p + 1..].iter().filter_map(|w| { let (n, v) = w.split_once('=')?; Some((n.parse().ok()?, u64::from_str_radix(v, 16).ok()?)) }).collect()).unwrap_or_default();
    // -i N: replay only iteration N (with -x: its register state, the given code)
    let only: Option<u64> = args.iter().position(|a| a == "-i").and_then(|p| args.get(p + 1)).and_then(|v| v.parse().ok());
    let iters = if let Some(n) = only { n + 1 } else if fixed.is_some() { 1 } else { iters };
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

    let mut uc = Unicorn::new(Arch::PPC, Mode::PPC32 | Mode::BIG_ENDIAN).expect("unicorn");
    uc.mem_map(CODE as u64, 0x1000, Permission::ALL).unwrap();
    uc.mem_map(DATA as u64, DATA_SIZE as usize, Permission::ALL).unwrap();
    uc.add_intr_hook(|uc, _intno| {
        let _ = uc.emu_stop();
    })
    .unwrap();
    let mut jc = jit::Cpu::new().expect("jit");
    jc.map(CODE, 0x1000).unwrap();
    jc.map(DATA, DATA_SIZE as usize).unwrap();
    jc.on_exception(|c, _e| c.stop());

    let gpr = |n: u32| -> RegisterPPC { unsafe { std::mem::transmute(2 + n as i32) } };
    let jreg = |r: RegisterPPC| -> Reg {
        match r as i32 {
            2..=33 => Reg::Gpr(r as u32 - 2),
            42..=73 => Reg::Fpr(r as u32 - 42),
            1 => Reg::Pc,
            74 => Reg::Lr,
            75 => Reg::Xer,
            76 => Reg::Ctr,
            77 => Reg::Msr,
            78 => Reg::Fpscr,
            _ => Reg::Cr,
        }
    };
    let fpr = |n: u32| -> RegisterPPC { unsafe { std::mem::transmute(42 + n as i32) } };
    let crf = |n: u32| -> RegisterPPC { unsafe { std::mem::transmute(34 + n as i32) } };

    // Self-modifying code: a block that overwrites an instruction ahead of itself, and one
    // that overwrites its own next instruction (the running block must be left).
    {
        let prog: Vec<u32> = vec![
            d(14, 3, 0, 1),            // li r3, 1
            d(15, 4, 0, CODE >> 16),   // lis r4, CODE>>16
            d(24, 4, 4, CODE & 0xFFFF),// ori r4, r4, lo        -> r4 = CODE
            d(15, 5, 0, 0x3860),       // lis r5, 0x3860        (li r3, 7 = 0x38600007)
            d(24, 5, 5, 7),            // ori r5, r5, 7
            d(36, 5, 4, 7 * 4),        // stw r5, 28(r4)        overwrite word 7 (li r3,2 -> li r3,7)
            d(14, 6, 0, 0),            // li r6, 0
            d(14, 3, 0, 2),            // [7] li r3, 2          (becomes li r3, 7)
            d(36, 5, 4, 10 * 4),       // stw r5, 40(r4)        overwrite word 10 ahead in this block
            d(14, 6, 0, 1),            // li r6, 1
            d(14, 3, 0, 3),            // [10] li r3, 3         (becomes li r3, 7)
            END_TRAP,
        ];
        let mut bytes = Vec::new();
        for w in &prog {
            bytes.extend_from_slice(&w.to_be_bytes());
        }
        jc.write(CODE, &bytes).unwrap();
        jc.set_reg(jreg(gpr(3)), 0);
        let r = jc.run(CODE, None);
        let r3 = jc.reg(jreg(gpr(3)));
        let r6 = jc.reg(jreg(gpr(6)));
        if r.is_err() || r3 != 7 || r6 != 1 {
            println!("self-modifying code: {:?} r3={} (expected 7) r6={} (expected 1)", r, r3, r6);
            return 1;
        }
        // the same program on Unicorn, for the record
        uc.mem_write(CODE as u64, &bytes).unwrap();
        uc.ctl_flush_tb().unwrap();
        uc.reg_write(gpr(3), 0).unwrap();
        let _ = uc.emu_start(CODE as u64, 0, 1_000_000, 0);
        if verbose {
            println!("self-modifying code: unicorn r3={} jit r3={}", uc.reg_read(gpr(3)).unwrap(), r3);
        }
    }

    let mut bad = 0u64;
    for it in 0..iters {
        let mut code: Vec<u32> = Vec::new();
        let n = match &fixed {
            Some(f) => {
                code = f.clone();
                f.len() as u32
            }
            None => 2 + r.below(6),
        };
        let mut pc = CODE;
        let end = CODE + 4 * (n + 8); // leave room: branches target this trap
        // mtxer r0 first: Unicorn's reg_write(XER) leaves SO/OV/CA as the last instruction
        // that wrote them set them, so the sequence itself must load XER (r0 = the value)
        if fixed.is_none() {
            code.push(x(0, 1, 0, 467, 0));
        }
        while code.len() < n as usize {
            gen(&mut r, pc, end, &mut code);
            pc = CODE + 4 * code.len() as u32;
        }
        while code.len() < (n + 8) as usize {
            code.push(0x6000_0000); // nop
        }
        code.push(END_TRAP);
        code.push(END_TRAP);
        let mut init = init_state(&mut r, end);
        for &(n, v) in &fpr_presets {
            init.fpr[n] = v;
        }
        if let Some(c) = cr_preset {
            init.cr = c;
        }
        if let Some(n) = only {
            if it < n {
                continue;
            }
        }
        let fixed = if only.is_some() || fixed.is_some() { Some(code.clone()) } else { None };
        let mut bytes = Vec::new();
        for w in &code {
            bytes.extend_from_slice(&w.to_be_bytes());
        }
        // Unicorn
        uc.mem_write(CODE as u64, &bytes).unwrap();
        uc.mem_write(CODE as u64 + 0x800, &init.data[..0x800]).unwrap();
        uc.ctl_flush_tb().unwrap(); // Unicorn keeps translations across mem_write
        uc.mem_write(DATA as u64, &init.data).unwrap();
        for i in 0..32 {
            uc.reg_write(gpr(i), init.gpr[i as usize] as u64).unwrap();
            uc.reg_write(fpr(i), init.fpr[i as usize]).unwrap();
        }
        for i in 0..8 {
            uc.reg_write(crf(i), ((init.cr >> (28 - 4 * i)) & 0xF) as u64).unwrap();
        }
        uc.reg_write(RegisterPPC::XER, init.xer as u64).unwrap();
        uc.reg_write(RegisterPPC::LR, end as u64).unwrap();
        uc.reg_write(RegisterPPC::CTR, end as u64).unwrap();
        uc.reg_write(RegisterPPC::FPSCR, init.fpscr as u64).unwrap();
        uc.reg_write(RegisterPPC::MSR, 0x6000).unwrap();
        let ur = uc.emu_start(CODE as u64, 0, 1_000_000, 0);
        // JIT
        jc.write(CODE, &bytes).unwrap();
        jc.write(CODE + 0x800, &init.data[..0x800]).unwrap();
        jc.write(DATA, &init.data).unwrap();
        for i in 0..32 {
            jc.set_reg(jreg(gpr(i)), init.gpr[i as usize] as u64);
            jc.set_reg(jreg(fpr(i)), init.fpr[i as usize]);
        }
        jc.set_reg(jreg(RegisterPPC::CR), init.cr as u64);
        jc.set_reg(jreg(RegisterPPC::XER), init.xer as u64);
        jc.set_reg(jreg(RegisterPPC::LR), end as u64);
        jc.set_reg(jreg(RegisterPPC::CTR), end as u64);
        jc.set_reg(jreg(RegisterPPC::FPSCR), init.fpscr as u64);
        jc.set_reg(jreg(RegisterPPC::MSR), 0x6000);
        let jr = jc.run(CODE, None);

        let mut diffs: Vec<String> = Vec::new();
        if ur.is_err() != jr.is_err() {
            diffs.push(format!("result unicorn {:?} jit {:?} (pc {:08x} / {:08x})", ur, jr, uc.reg_read(RegisterPPC::PC).unwrap(), jc.reg(jreg(RegisterPPC::PC))));
        }
        for i in 0..32 {
            let (u, j) = (uc.reg_read(gpr(i)).unwrap() as u32, jc.reg(jreg(gpr(i))) as u32);
            if u != j {
                diffs.push(format!("r{}: unicorn {:08x} jit {:08x}", i, u, j));
            }
        }
        for i in 0..32 {
            let (u, j) = (uc.reg_read(fpr(i)).unwrap(), jc.reg(jreg(fpr(i))));
            if u != j && !(f64::from_bits(u).is_nan() && f64::from_bits(j).is_nan()) {
                diffs.push(format!("f{}: unicorn {:016x} jit {:016x}", i, u, j));
            }
        }
        let mut ucr = 0u32;
        for i in 0..8 {
            ucr |= ((uc.reg_read(crf(i)).unwrap() as u32) & 0xF) << (28 - 4 * i);
        }
        let jcr = jc.reg(jreg(RegisterPPC::CR)) as u32;
        if ucr != jcr {
            diffs.push(format!("cr: unicorn {:08x} jit {:08x}", ucr, jcr));
        }
        for (name, reg) in [("xer", RegisterPPC::XER), ("lr", RegisterPPC::LR), ("ctr", RegisterPPC::CTR)] {
            let (mut u, mut j) = (uc.reg_read(reg).unwrap() as u32, jc.reg(jreg(reg)) as u32);
            if name == "xer" {
                // Unicorn's reg_read(XER) leaves out SO/OV/CA (they are checked through mfxer)
                u &= 0x1FFF_FFFF;
                j &= 0x1FFF_FFFF;
            }
            if u != j {
                diffs.push(format!("{}: unicorn {:08x} jit {:08x}", name, u, j));
            }
        }
        let (uf, jf) = (uc.reg_read(RegisterPPC::FPSCR).unwrap() as u32, jc.reg(jreg(RegisterPPC::FPSCR)) as u32);
        // only the rounding mode: Unicorn keeps FPRF and the exception bits per operation,
        // the JIT does not (fcmp's CR field is compared through CR)
        if (uf ^ jf) & 0x0000_0003 != 0 {
            diffs.push(format!("fpscr: unicorn {:08x} jit {:08x}", uf, jf));
        }
        let mut ud = uc.mem_read_as_vec(DATA as u64, DATA_SIZE as usize).unwrap();
        let mut jd = jc.read_vec(DATA, DATA_SIZE as usize).unwrap();
        ud.extend(uc.mem_read_as_vec(CODE as u64 + 0x800, 0x800).unwrap());
        jd.extend(jc.read_vec(CODE + 0x800, 0x800).unwrap());
        // memory: a stored NaN may differ in sign/payload (register NaNs are accepted too)
        let mut first_diff = None;
        let mut i = 0;
        while i < ud.len() {
            if ud[i] == jd[i] {
                i += 1;
                continue;
            }
            // a stored NaN at any alignment: an 8-byte (or 4-byte) window covering this
            // byte that is a NaN on both sides
            let mut skip_to = None;
            for k in 0..8usize {
                if i < k || i - k + 8 > ud.len() {
                    continue;
                }
                let b = i - k;
                if f64::from_be_bytes(ud[b..b + 8].try_into().unwrap()).is_nan() && f64::from_be_bytes(jd[b..b + 8].try_into().unwrap()).is_nan() {
                    skip_to = Some(b + 8);
                    break;
                }
            }
            if skip_to.is_none() {
                for k in 0..4usize {
                    if i < k || i - k + 4 > ud.len() {
                        continue;
                    }
                    let b = i - k;
                    if f32::from_be_bytes(ud[b..b + 4].try_into().unwrap()).is_nan() && f32::from_be_bytes(jd[b..b + 4].try_into().unwrap()).is_nan() {
                        skip_to = Some(b + 4);
                        break;
                    }
                }
            }
            match skip_to {
                Some(n) => i = n,
                None => {
                    first_diff = Some(i);
                    break;
                }
            }
        }
        if let Some(first) = first_diff {
            let lo = first.saturating_sub(12) & !3;
            let hi = (first + 12).min(ud.len());
            let base = if first < DATA_SIZE as usize { DATA } else { CODE + 0x800 - DATA_SIZE };
            diffs.push(format!("memory differs at {:08x} (from {:08x}): unicorn {:02x?} jit {:02x?}", base + first as u32, base + lo as u32, &ud[lo..hi], &jd[lo..hi]));
        }
        if fixed.is_some() {
            println!("unicorn: {:?}  jit: {:?}", ur, jr);
            for (bpc, code) in jc.dump_blocks() {
                let hex: Vec<String> = code.iter().map(|b| format!("{:02x}", b)).collect();
                println!("  block {:08x}: {}", bpc, hex.join(""));
            }
            for i in 0..32 {
                println!("  r{:<2} {:08x} {:08x}   f{:<2} {:016x} {:016x}", i, uc.reg_read(gpr(i)).unwrap() as u32, jc.reg(jreg(gpr(i))) as u32, i, uc.reg_read(fpr(i)).unwrap(), jc.reg(jreg(fpr(i))));
            }
            println!("  cr {:08x} {:08x}  xer {:08x} {:08x}  fpscr {:08x} {:08x}", ucr, jcr, uc.reg_read(RegisterPPC::XER).unwrap() as u32, jc.reg(jreg(RegisterPPC::XER)) as u32, uf, jf);
        }
        if !diffs.is_empty() || verbose {
            let shown = code.iter().rposition(|&w| w != 0x6000_0000 && w != END_TRAP).map(|p| p + 1).unwrap_or(0);
            let words: Vec<String> = code[..shown].iter().map(|w| format!("{:08x}", w)).collect();
            println!("--- iteration {} ({}): {}", it, if diffs.is_empty() { "ok" } else { "MISMATCH" }, words.join(" "));
            if !diffs.is_empty() {
                let regs: Vec<String> = (0..32).map(|i| format!("r{}={:08x}", i, init.gpr[i])).collect();
                println!("    init: cr={:08x} xer={:08x} fpscr={:08x} {}", init.cr, init.xer, init.fpscr, regs.join(" "));
                let fregs: Vec<String> = (0..32).map(|i| format!("f{}={:016x}", i, init.fpr[i])).collect();
                println!("    init fpr: {}", fregs.join(" "));
                for dd in &diffs {
                    println!("    {}", dd);
                }
                bad += 1;
            }
        }
        if bad >= 20 {
            println!("(stopping after 20 mismatches)");
            break;
        }
    }
    println!("{} iterations, {} mismatches", iters, bad);
    if bad == 0 { 0 } else { 1 }
}
