# RegisterAssignmentRVC Native ILP Pass: Build and Run Guide

The native RISC-V ILP pass runs after LLVM's Greedy register allocator. It refines the initial assignment to improve code compression for the RVC extension. The pass is intended for research and measurement.

The current pass builds its model inside LLVM and uses Gurobi. This guide covers its build, run modes, and solution reuse.

## Repository contents and target setup

The repository contains the LLVM patch, the RISC-V pass implementation and header, and small C test inputs such as `example1.c` and `example2.c`. The patch integrates the pass into LLVM's RISC-V pipeline and adds the experimental pass option.

Use the LLVM revision expected by the patch. The experiments accompanying this project used LLVM 22 at commit `fc4aad7`, with the RISC-V target enabled. The intended target configuration is RV32IMC with the ILP32 ABI, generally compiled with `-Oz` and `-mno-outline`.

## Requirements

- LLVM source with the RISC-V backend enabled, at a revision compatible with the patch.
- C++ exceptions enabled for `LLVMRISCVCodeGen`.
- For solving or exporting models: Gurobi headers, libraries, and a valid license.
- For solution reuse only: Gurobi is not required if the pass is built with `RISCV_ILP_USE_GUROBI=0`.

## Install the LLVM changes

From the LLVM source root, apply the supplied patch:

```bash
cd <LLVM_ROOT>
patch -p1 < /path/to/RegisterAssignmentRVC/llvm/llvm.patch
```

The patch updates files under `llvm/include/llvm/CodeGen`, `llvm/lib/CodeGen`, and `llvm/lib/Target/RISCV`, including the pass registration and build configuration.

Copy the pass source and header into the RISC-V target directory:

```bash
cp /path/to/RegisterAssignmentRVC/llvm/RISCVILPRegAlloc.cpp \
  <LLVM_ROOT>/llvm/lib/Target/RISCV/
cp /path/to/RegisterAssignmentRVC/llvm/RISCVILPRegAlloc.h \
  <LLVM_ROOT>/llvm/lib/Target/RISCV/
```

Review the patch against your LLVM checkout before applying it, particularly if you use a revision other than the one used for the experiments.

## Build LLVM

A typical configuration is:

```bash
cmake -G Ninja <LLVM_ROOT>/llvm \
  -DLLVM_ENABLE_PROJECTS=clang \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja clang
```

### Build for solution reuse without Gurobi

`RISCV_ILP_USE_GUROBI` defaults to `0`. In this build, omit Gurobi discovery, include, and link directives from the RISC-V CMake configuration. The pass can validate and apply precomputed SOL files in `reuse-only` mode. It cannot solve models or export LP/MST files.

### Build with Gurobi

Define `RISCV_ILP_USE_GUROBI=1` when compiling the pass, and configure the Gurobi include and link paths in the RISC-V build. This enables solving and LP/MST export. All modes that start a Gurobi environment require a valid license, including `emit`.

## Enable the pass

Enable the pass with `-riscv-enable-experimental-ilp-reg-alloc`. For example:

```bash
clang --target=riscv32 \
  -march=rv32imc \
  -mabi=ilp32 \
  -Oz \
  -mno-outline \
  -mllvm -riscv-enable-experimental-ilp-reg-alloc \
  -c tests/example1.c \
  -o example1_ilp.o
```

Use the same flags for `tests/example2.c`. To compare against LLVM Greedy, compile the same inputs without `-riscv-enable-experimental-ilp-reg-alloc`, then compare object sizes or inspect the generated instructions:

```bash
llvm-objdump -d example1_ilp.o
```

## Pass modes

Pass options are forwarded through Clang with `-mllvm`. Existing SOL files are read only in `reuse-only` mode.

| Mode | Behavior | Gurobi license required? | Directory required? |
|---|---|---:|---:|
| `solve` | Builds and optimizes in memory; does not export LP/MST. This is the Gurobi-build default. | Yes | No |
| `emit` | Writes the LP and Greedy MST; retains the Greedy allocation. | Yes | Yes |
| `emit-solve` | Writes the LP/MST and performs a new optimization. | Yes | Yes |
| `reuse-only` | Reads and validates a SOL; does not optimize. Retains Greedy if no valid SOL is found. | No | Yes |

In a build with `RISCV_ILP_USE_GUROBI=0`, `reuse-only` is the only supported mode and the default. Example:

```bash
CLANG=/path/to/llvm-build/bin/clang
ILP_DIR=/path/to/ilp-results

"$CLANG" --target=riscv32 -march=rv32imc -mabi=ilp32 -Oz -mno-outline \
  -mllvm -riscv-enable-experimental-ilp-reg-alloc \
  -mllvm "-riscv-ilp-dir=$ILP_DIR" \
  -mllvm -riscv-ilp-mode=reuse-only \
  -c tests/example1.c -o example1.o
```

## Export and solve models across machines

The `emit` mode in this implementation starts Gurobi before writing the LP, so it is not license-free. Generate the LP/MST on a machine with a valid Gurobi license:

```bash
"$CLANG" --target=riscv32 -march=rv32imc -mabi=ilp32 -Oz -mno-outline \
  -mllvm -riscv-enable-experimental-ilp-reg-alloc \
  -mllvm "-riscv-ilp-dir=$ILP_DIR" \
  -mllvm -riscv-ilp-mode=emit \
  -c tests/example1.c -o example1.o
```

Transfer the LP/MST to the machine that will solve the model. After solving it there, copy the SOL file or files back into the compiler's `ILP_DIR`, then run `reuse-only`. A machine without a Gurobi license can reuse existing solutions, but this pass cannot generate LP files there. License-free LP export would require a solver-independent model exporter.

## Compatibility with the earlier experiments

The initial register-allocation experiments used a Python-based implementation where the solver was replaceable (Gurobi, CBC etc.). The `-riscv-ilp-legacy-hash=true` option was added to keep the native pass aligned with those experiments by enabling reuse of compatible precomputed solutions. Use it only with `reuse-only`:

```bash
"$CLANG" --target=riscv32 -march=rv32imc -mabi=ilp32 -Oz -mno-outline \
  -mllvm -riscv-enable-experimental-ilp-reg-alloc \
  -mllvm "-riscv-ilp-dir=$ILP_DIR" \
  -mllvm -riscv-ilp-mode=reuse-only \
  -mllvm -riscv-ilp-legacy-hash=true \
  -c tests/example1.c -o example1.o
```

The option does not run Python or Gurobi. A Python-based variant can be provided if an alternative to the current solver is of interest.

## Native cache and SOL selection

Without `-riscv-ilp-legacy-hash=true`, the pass uses its native hash-based file layout. Before applying a solution, it validates binary assignments, completeness, legal physical registers, register classes, reserved registers, and interference constraints. It does not re-check optimality; a feasible time-limited solution can be reused.

For each key, the pass selects the matching SOL with the greatest numeric suffix (`_10.sol` before `_2.sol`). If there are no numbered matches, it tries the unnumbered `.sol`. If the selected file is invalid, Greedy is retained; lower-numbered files are not tried.

Typical native output files are:

| File | Purpose |
|---|---|
| `<key>.lp` | Exported model from `emit` or `emit-solve` |
| `<key>.mst` | Greedy start assignment exported with the LP |
| `<key>_<index>.sol` | Numbered intermediate solution from Gurobi `SolFiles` output |
| `<key>.config.txt` | Requested solver configuration snapshot |
| `sha_mapping.txt` | Native key mapped to source filename and function |

Keep separate runs and configurations in separate directories. Numbered SOL files are ordered by suffix, not by objective value, so the highest-numbered file must be the one you intend to reuse.

## Time limits and solver parameters

The default Gurobi time limit is 600 seconds per model optimization. It excludes model construction and other compiler work. Example options:

```bash
-mllvm -riscv-ilp-time-limit=300 \
-mllvm -riscv-ilp-gurobi-param=Threads=4 \
-mllvm -riscv-ilp-gurobi-param=MIPGap=0
```

A PRM file can be supplied with `-riscv-ilp-gurobi-prm=/path/to/settings.prm`. Precedence is defaults, then PRM settings, then repeated command-line `Name=Value` settings. In `reuse-only`, the requested configuration is compared with the saved snapshot when available. A mismatch produces a warning but does not prevent reuse of a valid allocation.
