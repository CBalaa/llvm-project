//===- RISCVSchedTensixBaby.td - Tensix baby RISC-V sched ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Scheduling model for the Tenstorrent "baby" RISC-V cores that sit in every
// Tensix tile (tt-1xx / Blackhole). CPU name: -mcpu=tt-bh, which is also the
// name the tt-metal toolchain's own GCC uses for this core
// (riscv-tt-elf-g++ -mcpu=tt-bh => -march=rv32im_zmmul_zaamo_zba_zbb).
//
// Scope: the scalar pipeline of RISCV B / T0 / T1 / NC. It does NOT model the
// Tensix coprocessor (the .ttinsn stream), and it does not model RISCV T2's
// vector unit.
//
// Every number below is traceable to one of:
//
//   [ISA-R] tt-isa-documentation/BlackholeA0/TensixTile/BabyRISCV/README.md
//           (pipeline, EX1/EX2 occupancy, load-latency table, retire unit,
//            store coalescing)
//   [ISA-I] tt-isa-documentation/BlackholeA0/TensixTile/BabyRISCV/
//           InstructionSet.md (fence semantics, Zba/Zbb, Zaamo, div/mul)
//   [MEAS]  measurements recorded in thirdparty/I-know-tt/
//           (fence = 9 cycles; see isa/l0-data-cache-and-fence.md)
//
// Anything that is documented-but-unquantified, or chosen because the ISA
// gives a *range*, is called out as [GAP] or [CHOICE] so that nobody mistakes
// it for a documented number.
//
//===----------------------------------------------------------------------===//
//
// Shape of the core [ISA-R]:
//
//   Frontend (1 instr/cycle, predicts all control flow)
//      -> EX1   (>= 1 cycle for every instruction; in-order; single issue)
//      -> EX2   (mul / FP arithmetic only)
//      -> Load/Store Unit (memory ops; >= 1 cycle)
//      -> Retire Unit (8-entry retire-order queue, 1 pop/cycle)
//
// "Each RISCV core is intended to execute one RISCV instruction per cycle"
// and "the Frontend is capable of sending one instruction per cycle to EX1".
// That is MicroOpBufferSize = 0 (in-order) with IssueWidth = 1.
//
//===----------------------------------------------------------------------===//

def TensixBabyModel : SchedMachineModel {
  // In-order: "The frontend is in-order, as is EX1". Results are buffered in
  // the Retire Unit and committed in program order.
  let MicroOpBufferSize = 0;

  // "the Frontend is capable of sending one instruction per cycle to EX1".
  let IssueWidth = 1;

  // Codegen-only; llvm-mca never reads these two.
  // LoadLatency: local data RAM / L0-data-cache hit on L1 = 2 cycles [ISA-R].
  let LoadLatency = 2;
  // 1 cycle in EX1 plus a "four cycle pipeline bubble" on a mispredict [ISA-R].
  let MispredictPenalty = 5;

  // Not every instruction in the ISA has an entry here (the vector unit, D, Q
  // and the crypto extensions are deliberately absent).
  let CompleteModel = false;

  // Extensions the baby cores do not implement [ISA-I]. Zbkb is NOT in this
  // list: the babies do implement `pack` and `brev8` (but not packh/zip/unzip).
  let UnsupportedFeatures = [HasStdExtZbc, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZbs, HasStdExtZabha, HasStdExtZacas,
                             HasStdExtZawrs, HasStdExtZcmt, HasStdExtZicond,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasStdExtZfa, HasVInstructions,
                             HasVInstructionsI64];
}

//===----------------------------------------------------------------------===//
// Processor resources
//===----------------------------------------------------------------------===//

// EX1 is the single-issue, in-order execution stage. BufferSize = 0 makes it a
// *dispatch hazard*: while it is occupied no other instruction can be issued.
// That is exactly the documented behaviour for the instructions whose EX1
// occupancy exceeds one cycle (divide, fence), and it also gives the model its
// "one instruction per cycle" throughput.
def TensixBabyEX1 : ProcResource<1> { let BufferSize = 0; }

// EX2 is a second, pipelined stage that only multiply and FP arithmetic reach
// ("spend one cycle in EX2"). Pipelined, so back-to-back multiplies are one
// per cycle with a latency of two.
def TensixBabyEX2 : ProcResource<1>;

// The Load/Store Unit. Occupancy is one cycle for a load or a store regardless
// of the memory latency [ISA-R], so it is pipelined: two independent loads can
// be in flight back to back.
def TensixBabyLSU : ProcResource<1>;

//===----------------------------------------------------------------------===//
// Scheduling classes that exist only for this model
//===----------------------------------------------------------------------===//

// `fence` (and `fence.tso`) are declared with `Sched<[]>` in RISCVInstrInfo.td,
// i.e. they carry no scheduling information in *any* RISC-V model, which is why
// stock llvm-mca reports them as "lack-sched". They are the one instruction
// class this model has to define from scratch.
def TensixBabyWriteFence : SchedWrite;

// `fence.i` is a different instruction. The babies do not implement Zifencei
// and the upstream docs describe it as behaving like a nop; measured on
// Blackhole p150a it costs 8 cycles net [MEAS].
def TensixBabyWriteFenceI : SchedWrite;

let SchedModel = TensixBabyModel in {

//===----------------------------------------------------------------------===//
// Branching
//===----------------------------------------------------------------------===//
// "Branch instructions (jal, jalr, beq, bne, blt, bge, etc) occupy EX1 for one
// cycle" [ISA-R]. The 4-cycle mispredict bubble is NOT modelled: llvm-mca does
// not model branch prediction at all, and it analyses straight-line code.
def : WriteRes<WriteJmp, [TensixBabyEX1]>;
def : WriteRes<WriteJal, [TensixBabyEX1]>;
def : WriteRes<WriteJalr, [TensixBabyEX1]>;

//===----------------------------------------------------------------------===//
// Integer arithmetic and logic
//===----------------------------------------------------------------------===//
// "Most integer instructions spend exactly one cycle in EX1" and "There is a
// forwarding path allowing the result of the Integer Unit to be consumed by the
// very next instruction. This means that integer operations (other than
// multiply and divide) appear to execute with latency 1 and throughput 1."
def : WriteRes<WriteIALU32, [TensixBabyEX1]>;
def : WriteRes<WriteIALU, [TensixBabyEX1]>;
def : WriteRes<WriteShiftImm32, [TensixBabyEX1]>;
def : WriteRes<WriteShiftImm, [TensixBabyEX1]>;
def : WriteRes<WriteShiftReg32, [TensixBabyEX1]>;
def : WriteRes<WriteShiftReg, [TensixBabyEX1]>;

// Zba -- "Zba provides three instructions: sh1add, sh2add, and sh3add" [ISA-I].
// Plain EX1 work, same as any other integer ALU op.
def : WriteRes<WriteSHXADD, [TensixBabyEX1]>;
// sh1add.uw/sh2add.uw/sh3add.uw are RV64-only; the baby is RV32.
def : WriteRes<WriteSHXADD32, [TensixBabyEX1]>;

// Zbb -- andn/clz/cpop/ctz/max/min/orc.b/orn/rev8/rol/ror/rori/sext.b/sext.h/
// xnor/zext.h [ISA-I]. All single-cycle EX1 integer operations.
def : WriteRes<WriteRotateImm, [TensixBabyEX1]>;
def : WriteRes<WriteRotateImm32, [TensixBabyEX1]>;
def : WriteRes<WriteRotateReg, [TensixBabyEX1]>;
def : WriteRes<WriteRotateReg32, [TensixBabyEX1]>;
def : WriteRes<WriteCLZ, [TensixBabyEX1]>;
def : WriteRes<WriteCLZ32, [TensixBabyEX1]>;
def : WriteRes<WriteCTZ, [TensixBabyEX1]>;
def : WriteRes<WriteCTZ32, [TensixBabyEX1]>;
def : WriteRes<WriteCPOP, [TensixBabyEX1]>;
def : WriteRes<WriteCPOP32, [TensixBabyEX1]>;
def : WriteRes<WriteREV8, [TensixBabyEX1]>;
def : WriteRes<WriteORCB, [TensixBabyEX1]>;
def : WriteRes<WriteIMinMax, [TensixBabyEX1]>;

// Zbkb, the part the babies implement: `pack` and `brev8` [ISA-I].
def : WriteRes<WriteBREV8, [TensixBabyEX1]>;
def : WriteRes<WritePACK, [TensixBabyEX1]>;

//===----------------------------------------------------------------------===//
// Integer multiplication
//===----------------------------------------------------------------------===//
// "Some instructions spend exactly one cycle in EX1, and then exactly one cycle
// in EX2: Integer multiply instructions (mul, mulh, mulhsu, mulhu)" [ISA-R].
// Latency 2, throughput 1 -- EX2 is a separate pipelined stage, so unlike
// Wormhole ("the next instruction cannot enter the unit until the multiply
// instruction has finished") a Blackhole multiply does not stall its successor.
def : WriteRes<WriteIMul, [TensixBabyEX1, TensixBabyEX2]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteIMul32, [TensixBabyEX1, TensixBabyEX2]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}

//===----------------------------------------------------------------------===//
// Integer division and remainder
//===----------------------------------------------------------------------===//
// "Divide instructions (div, divu, rem, remu) occupy EX1 for a variable number
// of cycles ... When dividing by zero or one, two cycles are required. When
// doing signed division or remainder of -2^31 by -1, two cycles are required.
// In all other cases, between six and 33 cycles are required, dependent upon
// the magnitude of the dividend." [ISA-R]
//
// [CHOICE] The ISA gives a range, not a number, and the model format cannot
// express "depends on the operand values". 20 is the midpoint of the 6..33
// case that dominates real code. Treat any llvm-mca number for a kernel that
// divides as +/- 13 cycles per divide, and prefer an explicit
// `# LLVM-MCA-LATENCY <n>` when the divisor is known.
//
// Occupying EX1 for the whole latency is the documented behaviour: divide
// blocks the instruction behind it.
def : WriteRes<WriteIDiv, [TensixBabyEX1]> {
  let Latency = 20;
  let ReleaseAtCycles = [20];
}
def : WriteRes<WriteIDiv32, [TensixBabyEX1]> {
  let Latency = 20;
  let ReleaseAtCycles = [20];
}
def : WriteRes<WriteIRem, [TensixBabyEX1]> {
  let Latency = 20;
  let ReleaseAtCycles = [20];
}
def : WriteRes<WriteIRem32, [TensixBabyEX1]> {
  let Latency = 20;
  let ReleaseAtCycles = [20];
}

//===----------------------------------------------------------------------===//
// Memory: stores
//===----------------------------------------------------------------------===//
// "the instruction spends one cycle in Load/Store and then proceeds directly to
// the Retire Unit" [ISA-R] -- so a store is EX1 (address computation) + one
// cycle of Load/Store Unit occupancy, latency 2, and it writes no register.
//
// [GAP] Sustained store throughput is *not* modelled as one per cycle. The
// docs say stores to L1 coalesce to one per cycle only if the store queue can
// merge them into aligned 128-bit blocks, and otherwise degrade to one
// coalesced store every five cycles; stores to any other region are one per
// cycle [ISA-R]. The model format keys on the opcode, not the address, so it
// cannot tell an L1 store from a NoC-register store. The optimistic (one per
// cycle) reading is used, which is correct for MMIO stores and for a store
// queue that is coalescing well.
def : WriteRes<WriteSTB, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteSTH, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteSTW, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteSTD, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteFST32, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteFST16, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteFST64, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}

//===----------------------------------------------------------------------===//
// Memory: loads
//===----------------------------------------------------------------------===//
// "The latency of a load instruction depends upon the memory region being
// accessed ... The minimum possible load latency is two cycles, due to all load
// instructions spending at least one cycle in EX1 and one cycle in the
// Load/Store Unit." [ISA-R]
//
// The documented per-region table [ISA-R] is:
//
//   local data RAM (MEM_LOCAL_BASE) / L1 with L0 data cache hit ....  2
//   mailboxes, PCBufs, TTSync, Tensix semaphores ................... >= 3
//   Tensix GPRs, Tensix backend configuration ...................... >= 4
//   tile control/debug/status, PIC, NoC 0/1, NoC overlay ........... >= 7
//   local data RAM via the slow (0xFFB1_4xxx) path ................. >= 8
//   L1 scratchpad with an L0 data cache miss ....................... >= 8
//   L1 scratchpad with an atomic instruction ....................... >= 12
//
// [GAP] llvm-mca keys a scheduling class on the *opcode*, so it cannot tell
// `lw` from local RAM (2) apart from `lw` from a NoC-overlay register (>= 7).
// The model uses the documented minimum, 2, which is right for the stack and
// for globals placed in local data RAM -- the placement the ISA docs recommend
// ("Software is strongly encouraged to place the call stack in this RAM"). A
// kernel that polls MMIO will therefore be predicted optimistically; annotate
// those loads with `# LLVM-MCA-LATENCY 7` (or 8, or 12) to correct it.
//
// Note that all load *sizes* share the same latency here, unlike Rocket where
// byte/halfword loads cost one cycle more. The baby's Load/Store Unit spends
// one cycle on the access whatever its width.
def : WriteRes<WriteLDB, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteLDH, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteLDW, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteLDD, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteFLD32, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteFLD16, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteFLD64, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}

//===----------------------------------------------------------------------===//
// Atomics (Zaamo)
//===----------------------------------------------------------------------===//
// The babies implement Zaamo but not Zalrsc [ISA-I], and "Atomics behave
// similarly to loads" [ISA-R]. A Zaamo op against L1 costs >= 12 cycles
// [ISA-R], and it also flushes the entire L0 data cache [ISA-I].
def : WriteRes<WriteAtomicW, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 12;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteAtomicD, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 12;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteAtomicLDW, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 12;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteAtomicLDD, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 12;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteAtomicSTW, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}
def : WriteRes<WriteAtomicSTD, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
}

//===----------------------------------------------------------------------===//
// fence
//===----------------------------------------------------------------------===//
// [ISA-I] gives three effects, all of which are ordering, not arithmetic:
//
//   * "Once a fence instruction leaves the frontend, the next instruction will
//     not leave the frontend until the fence instruction has retired" -- this
//     is the frontend serialization, and it is what the EX1 occupancy below
//     encodes. It is also why a fence's latency cannot be hidden by the
//     instructions that follow it.
//   * "A fence instruction will not enter the Load/Store Unit until the store
//     queue has been drained and all in-flight loads have determined their
//     result value."
//   * "When a fence instruction enters the Load/Store Unit, the entire L0 data
//     cache will be flushed."
//
// The docs give no cycle count. Measured on Blackhole p150a, BRISC [MEAS]:
//
//   bare fence, pipeline drained, store queue empty ....  9 cycles
//   back-to-back fences, 2nd onwards ..................   4 cycles each
//
// 9 is modelled. Note the second row: the model charges every fence 9, so a
// run of adjacent fences is over-estimated by 5 cycles each. And the measured
// 9 was taken with an *empty* store queue; the docs' other component (waiting
// for the store queue to drain, default merge timer 16 cycles) is not
// exercised, so a fence with an unretired store behind it is more expensive
// than the model says.
def : WriteRes<TensixBabyWriteFence, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 9;
  let ReleaseAtCycles = [9, 1];
}
def : WriteRes<TensixBabyWriteFenceI, [TensixBabyEX1, TensixBabyLSU]> {
  let Latency = 8;
  let ReleaseAtCycles = [8, 1];
}
def : InstRW<[TensixBabyWriteFence], (instrs FENCE, FENCE_TSO)>;
def : InstRW<[TensixBabyWriteFenceI], (instrs FENCE_I)>;

//===----------------------------------------------------------------------===//
// Single precision (partial F)
//===----------------------------------------------------------------------===//
// The babies implement fadd.s/fsub.s/fmul.s/fmadd.s et al. but NOT fdiv.s or
// fsqrt.s, which are left unsupported below. "Floating-point arithmetic
// instructions spend one cycle in EX1, and then exactly one cycle in EX2"
// [ISA-R] -- latency 2, same as an integer multiply.
let Latency = 2 in {
def : WriteRes<WriteFAdd32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMul32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMA32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFSGNJ32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMinMax32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCmp32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFClass32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtI32ToF32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF32ToI32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMovF32ToI32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMovI32ToF32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF32ToF64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF64ToF32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF16ToF32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF32ToF16, [TensixBabyEX1, TensixBabyEX2]>;
}

//===----------------------------------------------------------------------===//
// Half precision (partial Zfh)
//===----------------------------------------------------------------------===//
// Same caveats as F above [ISA-I]. Modelled identically to single precision.
// (The babies also have a custom CSR bit that switches these to BF16; the
// timing is unchanged.)
let Latency = 2 in {
def : WriteRes<WriteFAdd16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMul16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMA16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFSGNJ16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMinMax16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCmp16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFClass16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF16ToI32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtI32ToF16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMovF16ToI16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMovI16ToF16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF16ToF64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF64ToF16, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF16ToI64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtI64ToF16, [TensixBabyEX1, TensixBabyEX2]>;
}

//===----------------------------------------------------------------------===//
// Double precision (D)
//===----------------------------------------------------------------------===//
// The baby cores do not implement D -- -mcpu=tt-bh is ilp32 soft-float. The
// classes are defined anyway so that a stray f64 op is not silently dropped by
// llvm-mca's "lack-sched" path; the numbers are F's, which is a guess, and the
// feature set is what keeps them out of reach.
let Latency = 2 in {
def : WriteRes<WriteFAdd64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMul64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMA64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFSGNJ64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMinMax64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCmp64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFClass64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtI64ToF64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF64ToI64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtI32ToF64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF64ToI32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtI64ToF32, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFCvtF32ToI64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMovF64ToI64, [TensixBabyEX1, TensixBabyEX2]>;
def : WriteRes<WriteFMovI64ToF64, [TensixBabyEX1, TensixBabyEX2]>;
}

//===----------------------------------------------------------------------===//
// System
//===----------------------------------------------------------------------===//
// [GAP] The docs list csrrw/csrrs/csrrc/csrrwi/csrrsi/csrrci alongside fence as
// frontend-serializing ("An instruction can enter EX1 provided that ... all
// previous csrr* ... have retired" [ISA-R]) but give no cycle count for a CSR
// access. Modelled as a plain one-cycle EX1 op, which is a lower bound: it
// models the occupancy but not the serialization.
def : WriteRes<WriteCSR, [TensixBabyEX1]>;

// nop writes nothing and uses no unit. (The Retire Unit still takes a queue
// entry for it, but with single-issue in-order execution that is free.)
def : WriteRes<WriteNop, [TensixBabyEX1]>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
//===----------------------------------------------------------------------===//
// All zero. The forwarding network is strong enough that an integer op's result
// is usable by the very next instruction [ISA-R], which is what a producer
// latency of 1 with a 0-cycle advance already gives; there is nothing to buy
// back. (The one documented caveat is not expressible here: "If there are
// multiple populated queue entries which will eventually write to the same
// register, peeking might not be possible from _any_ of them", which is why
// the docs tell compilers to use distinct destination registers for the seven
// instructions following a load.)
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;

def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;

def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMinMax, 0>;

def : ReadAdvance<ReadBREV8, 0>;
def : ReadAdvance<ReadPACK, 0>;

def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;

def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;

def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFClass64, 0>;

//===----------------------------------------------------------------------===//
// Extensions the baby cores do not implement
//===----------------------------------------------------------------------===//
// These stay "Unsupported" so that llvm-mca reports them rather than inventing
// a cost. Note that D and Q are included: -mcpu=tt-bh is soft-float ilp32.
defm : UnsupportedSchedQ;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedXsf;

// Floating-point divide and square root are the one documented hole in the
// babies' F/Zfh subset: "fdiv.s and fsqrt.s are not implemented" [ISA-I].
//
// Everything else in this block is here for a different reason: TableGen
// requires the processor to define a resource for every SchedRead/SchedWrite
// that any instruction in the target can use, even for extensions the CPU does
// not have. Zfa's 16/64/128-bit forms (fli.h, fcvtmod.w.d, ...), the Zbkb
// instructions the babies lack (packh, zip, unzip), and D's divide/sqrt are
// therefore declared Unsupported rather than left undefined.
let Unsupported = true in {
def : WriteRes<WriteFDiv16, []>;
def : WriteRes<WriteFDiv32, []>;
def : WriteRes<WriteFDiv64, []>;
def : WriteRes<WriteFSqrt16, []>;
def : WriteRes<WriteFSqrt32, []>;
def : WriteRes<WriteFSqrt64, []>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;

// Zbkb, the parts the babies do NOT implement ("packh and zip and unzip are
// not" [ISA-I]).
def : WriteRes<WritePACK32, []>;
def : WriteRes<WriteZIP, []>;
def : ReadAdvance<ReadPACK32, 0>;
def : ReadAdvance<ReadZIP, 0>;
}

} // SchedModel = TensixBabyModel
