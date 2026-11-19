//=== -----------------RISCVILPRegAlloc.cpp --------------------- ==
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Place in llvm/lib/Target/RISCV. Requires C++ exceptions for LLVMRISCVCodeGen.
// RISCV_ILP_USE_GUROBI=1 additionally requires Gurobi headers/libraries/license.
// RISCV_ILP_USE_GUROBI=0 includes no Gurobi API and defaults to reuse-only.
// In CMake, omit Gurobi discovery/include/link directives for the 0 build;
// changing this source macro cannot remove unconditional CMake link libraries.
//
// Options passed through clang with -mllvm (directly to llc otherwise):
//   -riscv-ilp-dir=/path/to/cache
//   -riscv-ilp-mode=solve|emit|emit-solve|reuse-only (default: solve)
//   -riscv-ilp-legacy-hash=true (reuse-only; function-name files, PuLP SOL names)
//   -riscv-ilp-time-limit=600
//   -riscv-ilp-gurobi-prm=/path/to/settings.prm
//   -riscv-ilp-gurobi-param=Threads=4  (repeatable Name=Value)
//
// solve always solves without writing LP/MST; emit only writes LP/MST and
// keeps the initial Greedy allocation; emit-solve writes LP/MST and solves.
// reuse-only reads matching native SOL files, without building or solving.
// A missing/invalid SOL keeps the initial Greedy allocation. With no directory, 
// solve works in memory. A directory enables numbered SolFiles output and 
// sha_mapping.txt. reuse selects only the greatest numeric suffix; .
// Requested parameters are saved in <hash>.config.txt and checked on reuse.
// One <hash>.config.txt describes all solutions from the same run.
// Keep different runs/configurations in separate directories.
// reuse avoids creating a Gurobi environment. Modes that build a model
// (including emit) need RISCV_ILP_USE_GUROBI=1 and a valid license.
// Legacy reuse searches <base>/<source basename>/ for
// <function>_<index>.sol (greatest numeric index), or
// <function>.sol only when there are no matching numbered files. Its config
// is <base>/<source basename>/<function>.config.txt. Enable with -riscv-ilp-legacy-hash=true in reuse-only
// mode; the option name is retained for compatibility but performs no hashing.
// Native (non-legacy) naming/exports remain hash-based. Use separate directories
// when different source trees have the same source basename and function name.
// Legacy reuse does not append sha_mapping.txt.

#include "RISCVILPRegAlloc.h"

#define DEBUG_TYPE "riscv-ilp-regalloc"

// Set to 1 for the full solver, or 0 for SOL reuse without the Gurobi SDK.
// A compiler -DRISCV_ILP_USE_GUROBI=0/1 can override this default.
#ifndef RISCV_ILP_USE_GUROBI
#define RISCV_ILP_USE_GUROBI 0
#endif
#if RISCV_ILP_USE_GUROBI != 0 && RISCV_ILP_USE_GUROBI != 1
#error "RISCV_ILP_USE_GUROBI must be 0 or 1"
#endif
#if RISCV_ILP_USE_GUROBI
#include "gurobi_c++.h"
#endif
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA256.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <locale>
#include <iomanip>
#include <limits>
#include <iterator>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ILPInstruction = std::pair<std::string, std::vector<MachineOperand>>;
using ILPCall = std::pair<SlotIndex, std::set<std::pair<Register, int>>>;

// Each mode selects one action; Reuse reads existing SOL files.
enum class ILPMode { Solve, Emit, EmitSolve, Reuse };
static cl::opt<ILPMode> NativeILPMode(
    "riscv-ilp-mode", cl::desc("ILP action: solve, emit, emit-solve, or reuse"),
    cl::init(RISCV_ILP_USE_GUROBI ? ILPMode::Solve : ILPMode::Reuse),
    cl::values(clEnumValN(ILPMode::Solve, "solve", "Solve in memory"),
               clEnumValN(ILPMode::Emit, "emit", "Write LP/MST; keep initial allocation"),
               clEnumValN(ILPMode::EmitSolve, "emit-solve", "Write LP/MST and solve"),
               clEnumValN(ILPMode::Reuse, "reuse-only", "Only reuse SOL; otherwise keep initial allocation")));
static cl::opt<std::string> NativeILPDir(
    "riscv-ilp-dir", cl::desc("ILP base directory: native hashes or legacy <source filename>/<function>.sol"),
    cl::init(""));
static cl::opt<bool> NativeILPLegacyHash(
    "riscv-ilp-legacy-hash",
    cl::desc("reuse-only: use function-name files and PuLP SOL names (compatibility flag)"),
    cl::init(false));
static cl::opt<double> NativeILPTimeLimit(
    "riscv-ilp-time-limit", cl::desc("Default Gurobi time limit in seconds"),
    cl::init(600.0));
static cl::opt<std::string> NativeILPParamFile(
    "riscv-ilp-gurobi-prm", cl::desc("Optional Gurobi PRM parameter file"),
    cl::init(""));
static cl::list<std::string> NativeILPParams(
    "riscv-ilp-gurobi-param", cl::desc("Gurobi Name=Value; may be repeated"),
    cl::ZeroOrMore);

// Bump this when changing constraints, weights, filters, or variable naming.
// Length-delimited input fields avoid ambiguous concatenations in the Python hash.
static constexpr const char *NativeILPModelVersion = "riscv-native-ilp-v1";
static std::string nativeILPHash(
    MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
    MachineFunction &MF, VirtRegMap &VRM,
    const std::vector<Register> &VirtualRegisters,
    const std::vector<Register> &Phys,
    const std::vector<ILPInstruction> &Instructions,
    const std::vector<std::pair<Register, Register>> &Interferences,
    const std::vector<ILPCall> &Calls,
    const std::set<Register> &SavedCalleeSaved, bool CanReturn, bool BigStack) {
  SHA256 Hash;
  auto Add = [&](const std::string &Value) {
    Hash.update(std::to_string(Value.size()));
    Hash.update(":");
    Hash.update(Value);
  };
  auto Number = [&](uint64_t N) { Add(std::to_string(N)); };
  Add(NativeILPModelVersion);
  Add(LLVM_VERSION_STRING);
  Number(CanReturn);
  Number(BigStack);
  BitVector Reserved = TRI.getReservedRegs(MF);
  Number(Phys.size());
  for (Register P : Phys)
    Number(P.id());
  Number(VirtualRegisters.size());
  for (Register V : VirtualRegisters) {
    Number(V.id());
    // Include the initial assignment: repeated comparisons with the same
    // incumbent hit the cache, but a different incumbent receives a new key.
    Number(VRM.getPhys(V));
    for (Register P : Phys)
      Number(MRI.getRegClass(V)->contains(P) && !Reserved.test(P));
  }
  Number(Instructions.size());
  for (const auto &I : Instructions) {
    Add(I.first);
    // The model only consults the first three explicit operands.
    // Keep total operand count as well to distinguish instruction layouts.
    Number(I.second.size());
    for (size_t J = 0; J < std::min<size_t>(3, I.second.size()); ++J) {
      const MachineOperand &Op = I.second[J];
      if (Op.isReg()) {
        Add("reg");
        Number(Op.getReg().id());
        Number(Op.getSubReg());
      } else if (Op.isImm()) {
        Add("imm");
        Add(std::to_string(Op.getImm()));
      } else if (Op.isFI()) {
        Add("fi");
        Add(std::to_string(Op.getIndex()));
      } else if (Op.isMBB()) {
        Add("mbb");
        Add(std::to_string(Op.getMBB()->getNumber()));
      } else {
        Add("other");
      }
    }
  }
  Number(Interferences.size());
  for (const auto &Pair : Interferences) {
    Number(Pair.first.id());
    Number(Pair.second.id());
  }
  Number(Calls.size());
  for (const auto &Call : Calls) {
    Number(Call.second.size());
    // Preserve call order: the script uses the last observed store cost.
    for (const auto &Live : Call.second) {
      Number(Live.first.id());
      Add(std::to_string(Live.second));
    }
  }
  Number(SavedCalleeSaved.size());
  for (Register R : SavedCalleeSaved)
    Number(R.id());
  auto Digest = Hash.final();
  const char *Hex = "0123456789abcdef";
  std::string Key;
  Key.reserve(64);
  for (uint8_t Byte : Digest) {
    Key += Hex[Byte >> 4];
    Key += Hex[Byte & 15];
  }
  // Solver parameters are intentionally NOT part of the key: they do not
  // change this model, and an externally solved LP can be reused with new
  // time limits. Disable reuse explicitly when comparing solver settings.
  return Key;
}

// Legacy SOL paths are <base>/<source basename>/<function>[_index].sol.
// The source basename omits its extension; absolute source paths are discarded.
// The existing legacy-hash CLI spelling is retained for command compatibility;
// in this version it selects function-name lookup plus PuLP variable names.
static std::string nativeILPLegacyFunctionKey(MachineFunction &MF) {
  std::string Name = MF.getName().str();
  if (Name.empty() || Name == "." || Name == ".." ||
      Name.find('/') != std::string::npos ||
      Name.find('\\') != std::string::npos ||
      Name.find('\0') != std::string::npos)
    throw std::runtime_error("function name cannot be used as a single filename");
  std::string SourceName = std::filesystem::path(
      MF.getFunction().getParent()->getSourceFileName()).stem().string();
  if (SourceName.empty() || SourceName == "." || SourceName == ".." ||
      SourceName.find('\\') != std::string::npos ||
      SourceName.find('\0') != std::string::npos)
    throw std::runtime_error("source filename cannot be used as a directory name");
  return (std::filesystem::path(SourceName) / Name).generic_string();
}

static std::string nativeILPLegacyAssignmentName(
    Register V, Register P, const TargetRegisterInfo &TRI) {
  std::string VName, PName;
  raw_string_ostream VOut(VName), POut(PName);
  VOut << printReg(V, &TRI);
  POut << printReg(P, &TRI);
  // PuLP dicts uses str((V, P)); its name sanitizer replaces spaces and
  // -+[] ->/ with underscores. Tuple punctuation and quotes are retained.
  std::string Name = "RegisterCorrespondence_('" + VOut.str() + "', '" +
                     POut.str() + "')";
  for (char &C : Name)
    if (std::string("-+[] ->/").find(C) != std::string::npos) C = '_';
  return Name;
}

static std::string nativeILPPath(const std::string &Key, const char *Suffix) {
  return (std::filesystem::path(NativeILPDir.getValue()) / (Key + Suffix)).string();
}

// A requested-configuration snapshot can be compared on cache hits without
// starting Gurobi. It does not claim to capture implicit gurobi.env settings.
static std::string nativeILPRequestedConfig() {
  std::map<std::string, std::string> Params;
  auto Trim = [](std::string S) {
    size_t A = S.find_first_not_of(" \t\r\n");
    if (A == std::string::npos) return std::string();
    return S.substr(A, S.find_last_not_of(" \t\r\n") - A + 1);
  };
  auto Set = [&](std::string Name, std::string Value) {
    Name = Trim(Name); Value = Trim(Value);
    if (Name.empty() || Value.empty())
      throw std::runtime_error("empty Gurobi parameter name or value");
    for (char &C : Name)
      if (C >= 'A' && C <= 'Z') C += 'a' - 'A';
    // Normalize numeric spellings such as 600 and 600.0.
    std::istringstream In(Value);
    In.imbue(std::locale::classic());
    long double N;
    if ((In >> N) && std::isfinite(N)) {
      In >> std::ws;
      if (In.eof()) {
        std::ostringstream Out;
        Out.imbue(std::locale::classic());
        Out << std::setprecision(std::numeric_limits<long double>::max_digits10) << N;
        Value = Out.str();
      }
    }
    Params[Name] = Value;
  };
  Set("OutputFlag", "0");
  std::ostringstream Limit;
  Limit.imbue(std::locale::classic());
  Limit << std::setprecision(std::numeric_limits<double>::max_digits10)
        << NativeILPTimeLimit.getValue();
  Set("TimeLimit", Limit.str());
  if (!NativeILPParamFile.getValue().empty()) {
    std::ifstream File(NativeILPParamFile.getValue());
    if (!File) throw std::runtime_error("cannot read current Gurobi parameter file");
    std::string Line;
    while (std::getline(File, Line)) {
      Line = Trim(Line);
      if (Line.empty() || Line[0] == '#') continue;
      size_t Split = Line.find_first_of(" \t");
      if (Split == std::string::npos)
        throw std::runtime_error("invalid Gurobi parameter file line: " + Line);
      Set(Line.substr(0, Split), Line.substr(Split + 1));
    }
    if (File.bad()) throw std::runtime_error("error reading current parameter file");
  }
  for (const std::string &Setting : NativeILPParams) {
    size_t Split = Setting.find('=');
    if (Split == std::string::npos)
      throw std::runtime_error("expected Gurobi Name=Value, got: " + Setting);
    Set(Setting.substr(0, Split), Setting.substr(Split + 1));
  }
  // SolFiles is set by the pass; changing its output path is not a solver change.
  Params.erase("solfiles");
  std::ostringstream Out;
  Out << "# Native ILP requested configuration v1\n";
  Out << "# Model " << NativeILPModelVersion << "\n";
  Out << "# LLVM " << LLVM_VERSION_STRING << "\n";
  for (const auto &Entry : Params)
    Out << Entry.first << '=' << Entry.second << '\n';
  return Out.str();
}

static void nativeILPWriteConfig(const std::string &Path, const std::string &Text) {
  SmallString<256> Temp;
  int FD;
  if (auto EC = sys::fs::createUniqueFile(Path + ".tmp-%%%%%%.txt", FD, Temp))
    throw std::runtime_error("cannot create config file: " + EC.message());
  { raw_fd_ostream CloseFD(FD, true); }
  try {
    std::ofstream Out(Temp.str().str(), std::ios::binary);
    Out.write(Text.data(), Text.size());
    Out.close();
    if (!Out) throw std::runtime_error("cannot write config file");
    if (auto EC = sys::fs::rename(Temp, Path))
      throw std::runtime_error("cannot publish config file: " + EC.message());
  } catch (...) {
    sys::fs::remove(Temp);
    throw;
  }
}

static void nativeILPCheckConfig(const std::string &Key, const std::string &Sol,
                                  const std::string &Current, bool CurrentValid) {
  std::string Path = nativeILPPath(Key, ".config.txt");
  std::ifstream In(Path, std::ios::binary);
  if (!In) {
    errs() << "ILP: warning: config file not found in the directory provided"
           << " for " << Sol << "; the parameters may not be the same\n";
    return;
  }
  std::string Saved((std::istreambuf_iterator<char>(In)),
                    std::istreambuf_iterator<char>());
  if (In.bad() || !CurrentValid) {
    errs() << "ILP: warning: config comparison unavailable; the parameters may not be the same\n";
    return;
  }
  {
    // Shared config describes the latest numbered export. Old SOL files
    // predating it must not be attributed to the latest run.
    std::error_code A, B;
    auto ConfigTime = std::filesystem::last_write_time(Path, A);
    auto SolTime = std::filesystem::last_write_time(Sol, B);
    if (A || B || SolTime < ConfigTime) {
      errs() << "ILP: warning: config file found, but solution provenance is uncertain;"
             << " the parameters may not be the same\n";
      return;
    }
  }
  if (Saved == Current)
    errs() << "ILP: config file found: the requested parameters seem to be the same\n";
  else
    errs() << "ILP: warning: config file found, but the requested settings differ;"
           << " the parameters may not be the same\n";
}

static std::vector<std::string> nativeILPSolutionFiles(const std::string &Key) {
  std::vector<std::pair<unsigned long long, std::string>> Numbered;
  std::vector<std::string> Result;
  std::string Unnumbered;
  std::error_code EC;
  // Key may contain the source subdirectory in legacy mode. Match only the
  // function basename inside that directory; native hash keys have no parent.
  std::filesystem::path RelativeKey(Key);
  std::filesystem::path Directory =
      std::filesystem::path(NativeILPDir.getValue()) / RelativeKey.parent_path();
  std::string FileKey = RelativeKey.filename().string();
  std::filesystem::directory_iterator It(Directory, EC), End;
  if (EC == std::errc::no_such_file_or_directory)
    return Result;
  if (EC)
    throw std::runtime_error("cannot list ILP directory " + Directory.string() +
                             ": " + EC.message());
  for (; It != End; It.increment(EC)) {
    if (EC)
      break;
    std::string Name = It->path().filename().string();
    if (Name == FileKey + ".sol") {
      Unnumbered = It->path().string();
      continue;
    }
    std::string Prefix = FileKey + "_";
    if (Name.size() <= Prefix.size() + 4 ||
        Name.compare(0, Prefix.size(), Prefix) != 0 ||
        Name.compare(Name.size() - 4, 4, ".sol") != 0)
      continue;
    std::string Digits = Name.substr(Prefix.size(), Name.size() - Prefix.size() - 4);
    if (!std::all_of(Digits.begin(), Digits.end(),
                     [](char C) { return C >= '0' && C <= '9'; }))
      continue;
    try {
      Numbered.emplace_back(std::stoull(Digits), It->path().string());
    } catch (const std::out_of_range &) {
      // Ignore invalid pool suffixes, e.g. enormous decimal numbers.
    }
  }
  if (EC)
    throw std::runtime_error("cannot list ILP directory: " + EC.message());
  // Only the maximum numeric suffix is eligible; never fall back to a
  // lower index or an unnumbered SOL if that solution is invalid.
  if (!Numbered.empty()) {
    auto Latest = std::max_element(
      Numbered.begin(), Numbered.end(),
      [](const auto &A, const auto &B) {
        return A.first < B.first;
      });
    Result.push_back(Latest->second);
  } else if (!Unnumbered.empty()) {
    Result.push_back(Unnumbered);
  }

  return Result;
}

static bool nativeILPValidateAssignment(
    MachineFunction &MF, MachineRegisterInfo &MRI,
    const TargetRegisterInfo &TRI, const std::vector<Register> &VirtualRegisters,
    const std::vector<Register> &Phys,
    const std::vector<std::pair<Register, Register>> &Interferences,
    const std::map<Register, Register> &Candidate, std::string &Reason) {
  BitVector Reserved = TRI.getReservedRegs(MF);
  if (Candidate.size() != VirtualRegisters.size()) {
    Reason = "incomplete assignment";
    return false;
  }
  for (Register V : VirtualRegisters) {
    auto It = Candidate.find(V);
    if (It == Candidate.end()) {
      Reason = "missing assignment for virtual register " + std::to_string(V.id());
      return false;
    }
    Register P = It->second;
    if (std::find(Phys.begin(), Phys.end(), P) == Phys.end() ||
        Reserved.test(P) || !MRI.getRegClass(V)->contains(P)) {
      Reason = "illegal physical register for virtual register " + std::to_string(V.id());
      return false;
    }
  }
  for (const auto &Pair : Interferences) {
    auto A = Candidate.find(Pair.first);
    auto B = Candidate.find(Pair.second);
    if (A == Candidate.end() || (!Pair.second.isPhysical() && B == Candidate.end())) {
      Reason = "interference references an unmodeled virtual register";
      return false;
    }
    Register Other = Pair.second.isPhysical() ? Pair.second : B->second;
    if (A->second == Other) {
      Reason = "interfering registers assigned to the same physical register";
      return false;
    }
  }
  return true;
}

// Read a Gurobi text SOL directly. Model.read("file.sol") would only create
// a MIP start; it would NOT populate X without calling optimize().
// Only assignment variables matter for applying the register map. Validate
// one-hot assignments, target legality, and every collected interference;
// auxiliary compression values and any claimed objective are not trusted.
static bool nativeILPReadSolution(
    const std::string &Path, MachineFunction &MF, MachineRegisterInfo &MRI,
    const TargetRegisterInfo &TRI, const std::vector<Register> &VirtualRegisters,
    const std::vector<Register> &Phys,
    const std::vector<std::pair<Register, Register>> &Interferences,
    std::map<Register, Register> &Result, std::string &Reason) {
  std::ifstream Input(Path);
  if (!Input) {
    Reason = "cannot open solution";
    return false;
  }
  std::map<std::string, std::pair<Register, Register>> AssignmentNames;
  for (Register V : VirtualRegisters)
    for (Register P : Phys) {
      std::string Name = NativeILPLegacyHash
          ? nativeILPLegacyAssignmentName(V, P, TRI)
          : "RegisterCorrespondence_" + std::to_string(V.id()) + "_" +
                std::to_string(P.id());
      AssignmentNames.emplace(Name, std::make_pair(V, P));
    }
  std::set<std::string> Seen;
  std::map<Register, Register> Candidate;
  std::string Line;
  const std::string Prefix = "RegisterCorrespondence_";
  while (std::getline(Input, Line)) {
    std::istringstream Stream(Line);
    Stream.imbue(std::locale::classic());
    std::string Name, ValueText, Extra;
    if (!(Stream >> Name) || Name[0] == '#')
      continue;
    if (Name.compare(0, Prefix.size(), Prefix) != 0)
      continue;
    double Value;
    if (!(Stream >> Value) || !std::isfinite(Value) || (Stream >> Extra) ||
        std::abs(Value - std::round(Value)) > 1e-5 ||
        Value < -1e-5 || Value > 1.0 + 1e-5) {
      Reason = "malformed or non-binary assignment: " + Name;
      return false;
    }
    if (!Seen.insert(Name).second) {
      Reason = "duplicate assignment variable: " + Name;
      return false;
    }
    auto It = AssignmentNames.find(Name);
    if (It == AssignmentNames.end()) {
      Reason = "solution references an unknown register or assignment name: " + Name;
      return false;
    }
    if (Value > 0.5 && !Candidate.emplace(It->second.first, It->second.second).second) {
      Reason = "multiple physical assignments for one virtual register";
      return false;
    }
  }
  if (Input.bad()) {
    Reason = "error reading solution";
    return false;
  }
  if (!nativeILPValidateAssignment(MF, MRI, TRI, VirtualRegisters, Phys,
                                  Interferences, Candidate, Reason))
    return false;
  Result.swap(Candidate);
  return true;
}

static void nativeILPRecordHash(const std::string &Key, MachineFunction &MF) {
  auto Clean = [](std::string S) {
    for (char &C : S)
      if (C == '\n' || C == '\r' || C == '\t')
        C = ' ';
    return S;
  };
  std::string Record = Key + "\t" +
      Clean(MF.getFunction().getParent()->getSourceFileName()) + "\t" +
      Clean(MF.getName().str()) + "\n";
  // One append write; duplicate records across compile runs are intentional.
  std::ofstream Mapping(nativeILPPath("sha_mapping", ".txt"), std::ios::app);
  Mapping.write(Record.data(), Record.size());
  Mapping.flush();
  if (!Mapping)
    errs() << "ILP: warning: could not append sha_mapping.txt\n";
}

#if RISCV_ILP_USE_GUROBI
// Write to a unique file beside the destination, then publish by rename.
// Keeping the actual LP/MST/SOL extension lets Gurobi select the file format.
static void nativeILPWriteAtomic(GRBModel &Model, const std::string &Path) {
  SmallString<256> Temp;
  int FD;
  std::string Suffix = std::filesystem::path(Path).extension().string();
  std::error_code EC = sys::fs::createUniqueFile(Path + ".tmp-%%%%%%" + Suffix, FD, Temp);
  if (EC)
    throw std::runtime_error("cannot create output file: " + EC.message());
  { raw_fd_ostream CloseFD(FD, true); }
  try {
    Model.write(Temp.str().str());
    if (auto RenameEC = sys::fs::rename(Temp, Path))
      throw std::runtime_error("cannot publish output file: " + RenameEC.message());
  } catch (...) {
    sys::fs::remove(Temp);
    throw;
  }
}

#endif // RISCV_ILP_USE_GUROBI

// These are the compression heuristics, including its immediate bounds and 
// one-byte branch/preference weights.
// This models scalar GPR refinement, not vector or floating-point allocation.
static bool solveNativeRegAlloc(
    MachineFunction &MF, MachineRegisterInfo &MRI,
    const TargetRegisterInfo &TRI, VirtRegMap &VRM,
    const std::vector<Register> &VirtualRegisters,
    const std::vector<ILPInstruction> &Instructions,
    const std::vector<std::pair<Register, Register>> &Interferences,
    const std::vector<ILPCall> &Calls,
    const std::set<Register> &SavedCalleeSaved, bool CanReturn, bool BigStack,
    std::map<Register, Register> &Result) {
  if (VirtualRegisters.empty()) {
    errs() << "ILP: skipping " << MF.getName() << ": no virtual registers\n";
    return false;
  }
  try {
    bool Emit = NativeILPMode == ILPMode::Emit || NativeILPMode == ILPMode::EmitSolve;
    bool ReuseOnly = NativeILPMode == ILPMode::Reuse;
#if !RISCV_ILP_USE_GUROBI
    if (!ReuseOnly)
      throw std::runtime_error("this build has RISCV_ILP_USE_GUROBI=0; only reuse-only is supported");
#endif
    bool HasDir = !NativeILPDir.getValue().empty();
    if (NativeILPLegacyHash && !ReuseOnly)
      throw std::runtime_error("riscv-ilp-legacy-hash requires reuse-only mode");
    if ((Emit || ReuseOnly) && !HasDir)
      throw std::runtime_error("riscv-ilp-dir is required for emit, emit-solve, or reuse");
    // Explicit target enums: do not assume their numeric values match xN.
    const std::vector<Register> Phys = {
        RISCV::X5, RISCV::X6, RISCV::X7, RISCV::X8, RISCV::X9,
        RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14,
        RISCV::X15, RISCV::X16, RISCV::X17, RISCV::X18, RISCV::X19,
        RISCV::X20, RISCV::X21, RISCV::X22, RISCV::X23, RISCV::X24,
        RISCV::X25, RISCV::X26, RISCV::X27, RISCV::X28, RISCV::X29,
        RISCV::X30, RISCV::X31, RISCV::X1};
    const std::set<Register> Compact = {
        RISCV::X8, RISCV::X9, RISCV::X10, RISCV::X11,
        RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15};
    // x1 deliberately follows the Python model's save-cost classification.
    const std::set<Register> Callee = {
        RISCV::X1, RISCV::X8, RISCV::X9, RISCV::X18, RISCV::X19,
        RISCV::X20, RISCV::X21, RISCV::X22, RISCV::X23, RISCV::X24,
        RISCV::X25, RISCV::X26, RISCV::X27};
    std::string RequestedConfig;
    bool ConfigValid = false;
    if (HasDir) {
      try {
        RequestedConfig = nativeILPRequestedConfig();
        ConfigValid = true;
      } catch (const std::exception &E) {
        errs() << "ILP: warning: cannot snapshot current configuration: "
               << E.what() << "\n";
      }
    }
    std::string Key;
    if (HasDir) {
      if (auto EC = sys::fs::create_directories(NativeILPDir.getValue()))
        throw std::runtime_error("cannot create ILP directory: " + EC.message());
      if (NativeILPLegacyHash)
        Key = nativeILPLegacyFunctionKey(MF);
      else
        Key = nativeILPHash(MRI, TRI, MF, VRM, VirtualRegisters, Phys, Instructions,
                           Interferences, Calls, SavedCalleeSaved, CanReturn, BigStack);
      if (NativeILPLegacyHash) {
        errs() << "ILP: legacy function-name key " << Key << "\n";
      } else {
        errs() << "ILP: hash " << Key << " for " << MF.getName() << "\n";
        nativeILPRecordHash(Key, MF);
      }
      if (ReuseOnly) {
        for (const std::string &Sol : nativeILPSolutionFiles(Key)) {
          std::string Reason;
          if (nativeILPReadSolution(Sol, MF, MRI, TRI, VirtualRegisters, Phys,
                                    Interferences, Result, Reason)) {
            nativeILPCheckConfig(Key, Sol, RequestedConfig, ConfigValid);
            errs() << "ILP: reused validated allocation from " << Sol
                   << " for " << MF.getName() << " (optimality not rechecked)\n";
            return true;
          }
          errs() << "ILP: ignoring " << Sol << ": " << Reason << "\n";
        }
      }
    }
    if (ReuseOnly) {
      errs() << "ILP: no valid cached solution for " << MF.getName()
             << "; keeping initial allocation\n";
      return false;
    }

#if RISCV_ILP_USE_GUROBI
    errs() << "ILP: building model for " << MF.getName() << "\n";
    GRBEnv Env(true);
    if (HasDir)
      Env.set("SolFiles", nativeILPPath(Key, ""));
    Env.set(GRB_IntParam_OutputFlag, 0);
    Env.set(GRB_DoubleParam_TimeLimit, NativeILPTimeLimit.getValue());
    if (!NativeILPParamFile.getValue().empty())
      Env.readParams(NativeILPParamFile.getValue());
    // Settings precedence: defaults, PRM file, repeated command-line params.
    for (const std::string &Setting : NativeILPParams) {
      size_t Equal = Setting.find('=');
      if (Equal == std::string::npos || Equal == 0 || Equal + 1 == Setting.size())
        throw std::runtime_error("expected Gurobi Name=Value, got: " + Setting);
      Env.set(Setting.substr(0, Equal), Setting.substr(Equal + 1));
    }
    Env.start();
    GRBModel Model(Env);
    const BitVector Reserved = TRI.getReservedRegs(MF);
    std::map<std::pair<Register, Register>, GRBVar> Assignment;
    auto Binary = [&](const std::string &Name) {
      return Model.addVar(0.0, 1.0, 0.0, GRB_BINARY, Name);
    };
    for (Register V : VirtualRegisters) {
      GRBLinExpr Sum = 0;
      for (Register P : Phys) {
        // Additional legality bounds absent from the Python model: honor
        // each virtual register's class and the function's reserved registers.
        bool Legal = MRI.getRegClass(V)->contains(P) && !Reserved.test(P);
        GRBVar X = Model.addVar(0.0, Legal ? 1.0 : 0.0, 0.0, GRB_BINARY,
            "RegisterCorrespondence_" + std::to_string(V.id()) + "_" +
            std::to_string(P.id()));
        X.set(GRB_DoubleAttr_Start, VRM.getPhys(V) == P.asMCReg() ? 1.0 : 0.0);
        Assignment.emplace(std::make_pair(V, P), X);
        Sum += X;
      }
      Model.addConstr(Sum == 1);
    }
    auto Assigned = [&](Register V, Register P) -> GRBLinExpr {
      if (V.isPhysical())
        return GRBLinExpr(V == P ? 1.0 : 0.0);
      auto It = Assignment.find(std::make_pair(V, P));
      if (It == Assignment.end())
        throw std::runtime_error("instruction references an unmodeled register");
      return GRBLinExpr(It->second);
    };
    auto IsCompact = [&](Register R) {
      GRBLinExpr Sum = 0;
      for (Register P : Compact)
        Sum += Assigned(R, P);
      return Sum;
    };
    auto And = [&](GRBLinExpr A, GRBLinExpr B, const std::string &Name) {
      GRBVar Y = Binary(Name);
      Model.addConstr(Y >= A + B - 1);
      Model.addConstr(Y <= A);
      Model.addConstr(Y <= B);
      return Y;
    };
    auto SameReg = [&](Register A, Register B, const std::string &Name) {
      GRBLinExpr Sum = 0;
      for (Register P : Phys) {
        GRBVar DS = And(Assigned(A, P), Assigned(B, P),
                       Name + "_" + std::to_string(P.id()));
        Sum += DS;
      }
      GRBVar Y = Binary(Name);
      Model.addConstr(Y == Sum);
      return Y;
    };
    for (const auto &Pair : Interferences) {
      if (Pair.second.isPhysical()) {
        if (std::find(Phys.begin(), Phys.end(), Pair.second) != Phys.end())
          Model.addConstr(Assigned(Pair.first, Pair.second) == 0);
      } else {
        for (Register P : Phys)
          Model.addConstr(Assigned(Pair.first, P) +
                          Assigned(Pair.second, P) <= 1);
      }
    }

    GRBLinExpr Objective = 0;
    const std::set<std::string> Offsets = {"ADDI", "ANDI", "SLLI", "SRLI", "SRAI"};
    const std::set<std::string> Commutative = {"ADD", "AND", "OR", "XOR"};
    const std::set<std::string> CompactSame = {"SUB", "AND", "OR", "XOR", "SRLI", "SRAI", "ANDI"};
    unsigned Idx = 0;
    for (const auto &Entry : Instructions) {
      const std::string &Op = Entry.first;
      const auto &Ops = Entry.second;
      std::string Tag = std::to_string(Idx++);
      if (Ops.size() < 2 || !Ops[0].isReg())
        continue;
      Register D = Ops[0].getReg();
      bool HasS = Ops[1].isReg();
      Register S = HasS ? Ops[1].getReg() : Register();
      bool HasImm = Ops.size() > 2 && Ops[2].isImm();
      int64_t Imm = HasImm ? Ops[2].getImm() : 0;
      bool IsOffset = Offsets.count(Op);
      bool IsComm = Commutative.count(Op);
      bool SameCandidate = (IsOffset || IsComm || Op == "SUB" || Op == "COPY") &&
          HasS && S != RISCV::X0 && S != RISCV::X2 &&
          (!IsOffset || (HasImm && Imm < 64));
      if (SameCandidate) {
        GRBVar Same = SameReg(D, S, "CompressionDS_" + Tag);
        if (IsComm && Ops.size() > 2 && Ops[2].isReg()) {
          Register S2 = Ops[2].getReg();
          // Python treats x0/sp in the second source as noncommutative.
          if (S2 != RISCV::X0 && S2 != RISCV::X2) {
            GRBVar Same2 = SameReg(D, S2, "CompressionDS2_" + Tag);
            GRBVar Either = Binary("CompressionDS_Reg_Reg_" + Tag);
            Model.addConstr(Either <= Same + Same2);
            Model.addConstr(Either >= Same);
            Model.addConstr(Either >= Same2);
            Same = Either;
          }
        }
        if (CompactSame.count(Op)) {
          GRBLinExpr Sources = IsCompact(S);
          if (!IsOffset && Ops.size() > 2 && Ops[2].isReg() &&
              Ops[2].getReg().isVirtual())
            Sources = And(Sources, IsCompact(Ops[2].getReg()),
                          "CompressionReg_Src1_Src2_" + Tag);
          Objective += 2 * And(Same, Sources,
                               "CompressionDS_Reg_ComprimableReg_" + Tag);
        } else {
          Objective += 2 * Same;
        }
      }

      bool IsMem = Op == "LW" || Op == "SW";
      bool IsBranch = Op == "BEQ" || Op == "BNE";
      bool IsSP = Op == "ADDI";
      bool CompactD = D.isVirtual() || Compact.count(D);
      bool TargetOK = (HasImm && Imm >= 0 && Imm < 128) ||
                      (Ops.size() > 2 && Ops[2].isMBB());
      bool Compressed = ((IsMem || IsBranch) && HasS &&
          ((IsMem && S != RISCV::X2 && S != RISCV::X0) ||
           (IsBranch && S == RISCV::X0)) && CompactD && TargetOK) ||
          (IsSP && D.isVirtual() &&
           ((HasS && S == RISCV::X2 && HasImm && Imm <= 1023) || Ops[1].isFI()));
      if (Compressed) {
        GRBLinExpr C = IsCompact(D);
        if (Compact.count(D))
          C = IsCompact(S);
        else if (IsMem)
          C = And(IsCompact(D), IsCompact(S), "CompressionReg_Src_" + Tag);
        GRBVar Y = Binary("CompressionReg_" + Tag);
        Model.addConstr(Y == C);
        // ADDI belongs to Python's Instructions_x0; its weight is one.
        Objective += (IsMem ? 2 : 1) * Y;
      }
      bool Preferable = (!Compressed && (IsMem || IsSP) &&
                         Ops[1].isFI() && D.isVirtual()) ||
                        (Op == "SW" && D == RISCV::X0 && HasS && S.isVirtual());
      if (Preferable) {
        GRBVar Y = Binary("PreferableCompressionReg_" + Tag);
        Model.addConstr(Y == IsCompact(D == RISCV::X0 ? S : D));
        Objective += Y;
      }
    }
    for (Register P : Phys) {
      GRBVar Used = Binary("UsedRegister_" + std::to_string(P.id()));
      GRBLinExpr Sum = 0;
      for (Register V : VirtualRegisters) {
        GRBLinExpr X = Assigned(V, P);
        Model.addConstr(Used >= X);
        Sum += X;
      }
      Model.addConstr(Used <= Sum);
      if (Callee.count(P) && !SavedCalleeSaved.count(P))
        Objective -= (4 + 4 * BigStack) * CanReturn * Used;
    }
    // Python charges reloads per call, and stores once per virtual register.
    // As in Python, the last call entry supplies the store-count coefficient.
    std::map<Register, int> StoreCost;
    for (const auto &Call : Calls)
      for (const auto &Live : Call.second) {
        for (Register P : Phys)
          if (!Callee.count(P))
            Objective -= (2 + 2 * BigStack) * Assigned(Live.first, P);
        StoreCost[Live.first] = 2 * Live.second + 2 * BigStack;
      }
    for (const auto &Entry : StoreCost) {
      GRBVar Need = Binary("NeedStoreCall_" + std::to_string(Entry.first.id()));
      GRBLinExpr Caller = 0;
      for (Register P : Phys)
        if (!Callee.count(P)) {
          GRBLinExpr X = Assigned(Entry.first, P);
          Model.addConstr(Need >= X);
          Caller += X;
        }
      Model.addConstr(Need <= Caller);
      Objective -= Entry.second * Need;
    }
    Model.setObjective(Objective, GRB_MAXIMIZE);
    Model.update();
    // Record settings before exporting/solving. Reuse does not overwrite them.
    if (HasDir && ConfigValid)
      nativeILPWriteConfig(nativeILPPath(Key, ".config.txt"), RequestedConfig);
    if (Emit) {
      nativeILPWriteAtomic(Model, nativeILPPath(Key, ".lp"));
      nativeILPWriteAtomic(Model, nativeILPPath(Key, ".mst"));
      errs() << "ILP: exported " << nativeILPPath(Key, ".lp")
             << " and greedy MST for " << MF.getName() << "\n";
      if (NativeILPMode == ILPMode::Emit) {
        errs() << "ILP: export only; keeping initial allocation for "
               << MF.getName() << "\n";
        return false;
      }
    }
    auto SolveStart = std::chrono::steady_clock::now();
    errs() << "ILP: starting solve for " << MF.getName() << "\n";
    Model.optimize();
    double SolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - SolveStart).count();
    errs() << "ILP: solve finished for " << MF.getName()
       << ", status=" << Model.get(GRB_IntAttr_Status)
       << ", solutions=" << Model.get(GRB_IntAttr_SolCount)
         << ", solve seconds=" << SolveSeconds << "\n";
    int Status = Model.get(GRB_IntAttr_Status);
    if (Model.get(GRB_IntAttr_SolCount) == 0) {
      errs() << "ILP: no feasible incumbent for " << MF.getName()
             << " (Gurobi status " << Status << "); keeping initial allocation\n";
      return false;
    }
    std::map<Register, Register> Candidate;
    for (Register V : VirtualRegisters) {
      for (Register P : Phys) {
        double X = Assignment.at(std::make_pair(V, P)).get(GRB_DoubleAttr_X);
        if (!std::isfinite(X) || std::abs(X - std::round(X)) > 1e-5 ||
            X < -1e-5 || X > 1.0 + 1e-5)
          throw std::runtime_error("non-binary assignment in Gurobi solution");
        if (X > 0.5 && !Candidate.emplace(V, P).second)
          throw std::runtime_error("multiple physical assignments in Gurobi solution");
      }
    }
    std::string Reason;
    if (!nativeILPValidateAssignment(MF, MRI, TRI, VirtualRegisters, Phys,
                                    Interferences, Candidate, Reason))
      throw std::runtime_error("invalid Gurobi assignment: " + Reason);
    // SolFiles publishes numbered solutions during optimize().
    Result.swap(Candidate);
    errs() << "ILP: "
           << (Status == GRB_OPTIMAL ? "optimal" : "feasible incumbent")
           << " for " << MF.getName() << ", objective "
           << Model.get(GRB_DoubleAttr_ObjVal) << "\n";
    return true;

#endif // RISCV_ILP_USE_GUROBI
#if RISCV_ILP_USE_GUROBI
  } catch (const GRBException &E) {
    errs() << "ILP: Gurobi error " << E.getErrorCode() << ": "
           << E.getMessage() << "; keeping initial allocation\n";
#endif // RISCV_ILP_USE_GUROBI
  } catch (const std::exception &E) {
    errs() << "ILP: " << E.what()
           << "; keeping initial allocation\n";
  }

  return false;
}
} // namespace


static bool isLoadFromStackSlot(const TargetInstrInfo *TII, MachineInstr &MI,
                                int StackSlot) {
  int FI = -1;
  TypeSize Size = TypeSize::getZero();
  Register Reg = TII->isLoadFromStackSlot(MI, FI, Size);
  return Reg && FI == StackSlot;
}

static bool isStoreToStackSlot(const TargetInstrInfo *TII, MachineInstr &MI,
                               int StackSlot) {
  int FI = -1;
  TypeSize Size = TypeSize::getZero();
  Register Reg = TII->isStoreToStackSlot(MI, FI, Size);
  return Reg && FI == StackSlot;
}

void RISCVILPRegAlloc::computeLiveStackInterval(MachineFunction &MF,
                                              int StackSlot,
                              const TargetRegisterClass &RC,
                              const TargetInstrInfo *TII, SlotIndexes *SI,
                              LiveStacks *LS) {
  LiveInterval &StackLI = LS->getOrCreateInterval(StackSlot, &RC);
  StackLI.clear();

  DenseMap<MachineBasicBlock *, bool> LiveIn;
  DenseMap<MachineBasicBlock *, bool> LiveOut;

  // Backward fixed-point liveness.
  bool Changed = true;
  while (Changed) {
    Changed = false;

    for (MachineBasicBlock &MBB : llvm::reverse(MF)) {
      bool OldIn = LiveIn[&MBB];
      bool OldOut = LiveOut[&MBB];

      bool Out = false;
      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (LiveIn[Succ]) {
          Out = true;
          break;
        }
      }

      bool SeenUse = false;
      bool KilledByStore = false;

      // Scan forward:
      // A load before any store means the slot must be live-in.
      // A store kills the previous stack-slot value.
      for (MachineInstr &MI : MBB) {
        if (isStoreToStackSlot(TII, MI, StackSlot)) {
          KilledByStore = true;
          break;
        }

        if (isLoadFromStackSlot(TII, MI, StackSlot)) {
          SeenUse = true;
          break;
        }
      }

      bool In = SeenUse || (Out && !KilledByStore);

      LiveIn[&MBB] = In;
      LiveOut[&MBB] = Out;

      if (OldIn != In || OldOut != Out)
        Changed = true;
    }
  }

  VNInfo *CurrentVNI = nullptr;

  for (MachineBasicBlock &MBB : MF) {
    bool Live = LiveIn[&MBB];
    SlotIndex SegmentStart;

    if (Live && !MBB.empty())
      SegmentStart = SI->getMBBStartIdx(&MBB);

    for (MachineInstr &MI : MBB) {
      if (isStoreToStackSlot(TII, MI, StackSlot)) {
        SlotIndex StoreIdx = SI->getInstructionIndex(MI).getRegSlot();

        CurrentVNI = StackLI.getNextValue(StoreIdx, LS->getVNInfoAllocator());

        Live = true;
        SegmentStart = StoreIdx;
        continue;
      }

      if (isLoadFromStackSlot(TII, MI, StackSlot)) {
        if (!Live)
          continue;

        if (!CurrentVNI) {
          SlotIndex LoadIdx = SI->getInstructionIndex(MI).getRegSlot();
          CurrentVNI = StackLI.getNextValue(LoadIdx, LS->getVNInfoAllocator());
        }
      }
    }

    if (Live && LiveOut[&MBB] && !MBB.empty()) {
      SlotIndex End = SI->getMBBEndIdx(&MBB);
      if (SegmentStart < End)
        StackLI.addSegment(LiveRange::Segment(SegmentStart, End, CurrentVNI));
    } else if (Live && !MBB.empty()) {
      MachineInstr *LastLoad = nullptr;

      for (MachineInstr &MI : MBB) {
        if (isLoadFromStackSlot(TII, MI, StackSlot))
          LastLoad = &MI;
      }

      if (LastLoad) {
        SlotIndex End = SI->getInstructionIndex(*LastLoad).getDeadSlot();
        if (SegmentStart < End)
          StackLI.addSegment(LiveRange::Segment(SegmentStart, End, CurrentVNI));
      }
    }
  }

  bool OK = StackLI.verify();
  assert(OK && "Invalid LiveInterval");
}

int RISCVILPRegAlloc::isLoadOrConstant(MachineRegisterInfo* MRI,
    MachineInstr* MI,
    bool checkLoad) {

    if (MI->mayLoad() && MI->getOperand(1).isFI())
    return 1;

    return 0;


   /* if (checkLoad && MI->mayLoad()) {
        for (unsigned i = 1; i < MI->getNumOperands(); i++)
            if (MI->getOperand(i).isReg() && MI->getOperand(i).getReg().isVirtual())
                return 0;
        return 1;
    }
    switch (MI->getOpcode()) {
    case RISCV::LUI:
        return true;
    case RISCV::ADDI:
    case RISCV::SLLI:
    case RISCV::SRLI:
    case RISCV::SRAI:
        if ((MI->getOperand(1).isReg() && MI->getOperand(1).getReg() == RISCV::X0) ||
            MI->getOperand(1).isFI() ||
            (MI->getOperand(2).isGlobal() && !MI->getOperand(1).isReg()) ||
            (MI->getOperand(2).isBlockAddress() && !MI->getOperand(1).isReg()))
            return 1;
        else if ((MI->getOperand(2).isGlobal() && MI->getOperand(1).isReg()) ||
            (MI->getOperand(2).isBlockAddress() && MI->getOperand(1).isReg()) ||
            (MI->getOperand(2).isImm() && MI->getOperand(1).isReg())) {
            MachineInstr* AuxInst = MRI->getUniqueVRegDef(MI->getOperand(1).getReg());
            if (!AuxInst)
                return 0;
            if (AuxInst->getOpcode() == RISCV::LUI)
                return 2;
            else
                return 0;
        }
    default:
        return 0;
    }*/
}

int RISCVILPRegAlloc::getNumberOfStores(MachineRegisterInfo *MRI,
                                        Register Reg) {
  MachineInstr *MII = MRI->getUniqueVRegDef(Reg);
  if (MII)
    return isLoadOrConstant(MRI, MII) ? 0 : 1;
  int NumStores = 0;
  for (auto &Def : MRI->def_instructions(Reg))
      NumStores++;
  return NumStores;
}

bool RISCVILPRegAlloc::instructionDoesNotReturn(Instruction& I) {
    if (auto* CB = dyn_cast<CallBase>(&I))
        return CB->hasFnAttr(Attribute::NoReturn);
    return false;
}

// A basic block can only return if it terminates with a ReturnInst and does not
// contain calls to noreturn functions.
bool RISCVILPRegAlloc::basicBlockCanReturn(BasicBlock & BB) {
    if (!isa<ReturnInst>(BB.getTerminator()))
        return false;

    return std::none_of(
        BB.begin(), BB.end(),
        [this](Instruction& I) {
        return instructionDoesNotReturn(I);
    }
    );

}

// FIXME: this doesn't handle recursion.
bool RISCVILPRegAlloc::canReturn(Function& F) {
    SmallVector<BasicBlock*, 16> Worklist;
    SmallPtrSet<BasicBlock*, 16> Visited;

    Visited.insert(&F.front());
    Worklist.push_back(&F.front());

    do {
        BasicBlock* BB = Worklist.pop_back_val();
        if (basicBlockCanReturn(*BB))
            return true;
        for (BasicBlock* Succ : successors(BB))
            if (Visited.insert(Succ).second)
                Worklist.push_back(Succ);
    } while (!Worklist.empty());

    return false;
}


bool RISCVILPRegAlloc::applyILP(MachineFunction& MF) {

    std::vector<Register> VirtualRegisters;
    std::vector<std::pair<std::string, std::vector<MachineOperand>>> Instructions;
    std::vector<std::set<Register>> CopyRelated;
    std::set<Register> CSDefPhysicalRegisters;
    std::vector<std::pair<SlotIndex, std::set<std::pair<Register, int>>>> LiveRegsCallTmp;
    std::set<Register> UnusedRegisters = {RISCV::X0, RISCV::X2, RISCV::X3, RISCV::X4 };


    errs() << "**** Function: " << MF.getName() << " *****\n";
    MF.print(errs());
    errs() << "VRM before ILP:\n";
    VRM->print(errs());

    LLVM_DEBUG(dbgs() << "**** Function: " << MF.getName() << " *****\n");
    LLVM_DEBUG(MF.print(dbgs()));

    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
        Register Reg = Register::index2VirtReg(i);
        if (LIS->hasInterval(Reg) && !MRI->use_instructions(Reg).empty())
            VirtualRegisters.push_back(Reg);
    }
    for (df_iterator<MachineDomTreeNode*> DI = df_begin(MDT->getRootNode()),
        DE = df_end(MDT->getRootNode());
        DI != DE; ++DI) {
        MachineBasicBlock* MBB = DI->getBlock();
        for (auto& MI : *MBB) {
            if (MI.getOpcode() == RISCV::COPY &&
                MI.getOperand(0).getReg().isVirtual() &&
                MI.getOperand(1).getReg().isVirtual()) {
                for (auto& Set : CopyRelated)
                    if (Set.find(MI.getOperand(1).getReg()) != Set.end()) {
                        Set.insert(MI.getOperand(0).getReg());
                    }
            }
            else if (MI.getNumOperands() > 0 && MI.getOperand(0).isReg() &&
                MI.getOperand(0).isDef() &&
                MI.getOperand(0).getReg().isVirtual()) {
                std::set<Register> NewSet;
                NewSet.insert(MI.getOperand(0).getReg());
                CopyRelated.push_back(NewSet);
            }
            // For specific instructions that may require registers.
            if (MI.isInlineAsm())
                for (auto Op : MI.operands())
                    if (Op.isReg() && Op.getReg().isVirtual() && Op.isDead())
                        VirtualRegisters.push_back(Op.getReg());
        }
    }

    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
        Register Reg = Register::index2VirtReg(i);
        unsigned NumOfValues = 0;
        std::vector<unsigned> SetsNumbers;
        for (unsigned Idx = 0; Idx < CopyRelated.size(); Idx++) {
            if (CopyRelated[Idx].find(Reg) != CopyRelated[Idx].end()) {
                NumOfValues++;
                SetsNumbers.push_back(Idx);
            }
        }

        bool Remove = false;
        if (NumOfValues > 1) {
            for (unsigned Idx1 = 0; (Idx1 < SetsNumbers.size() - 1) && !Remove;
                Idx1++)
                for (unsigned Idx2 = Idx1 + 1; Idx2 < SetsNumbers.size() && !Remove;
                    Idx2++)
                    for (auto Elem1 : CopyRelated[SetsNumbers[Idx1]]) {
                        for (auto Elem2 : CopyRelated[SetsNumbers[Idx2]]) {
                            if (Elem1 != Elem2)
                                if (LIS->getInterval(Elem1).overlaps(LIS->getInterval(Elem2))) {
                                    Remove = true;
                                    break;
                                }
                        }
                        if (Remove)
                            break;
                    }
        }

        if (Remove)
            for (auto Idx : SetsNumbers)
                CopyRelated[Idx].erase(Reg);
    }

    std::set<Register> PhysicalRegisters;
    for (auto& MBB : MF)
        for (auto& MI : MBB) {
            std::vector<MachineOperand> Instruction;
            for (unsigned i = 0; i < MI.getNumOperands(); i++) {
                MachineOperand Op = MI.getOperand(i);
                Instruction.push_back(Op);
                if (Op.isReg() && Op.getReg().isPhysical()) {
                    Register PhysReg = Op.getReg();
                    PhysicalRegisters.insert(PhysReg);
                    if ((Op.isDef() || Op.isEarlyClobber()) &&
                        TRI->isCalleeSavedPhysReg(PhysReg.asMCReg(), MF))
                        CSDefPhysicalRegisters.insert(PhysReg);
                }
            }


            if (TII) {

                if (MI.isCall()) {
                    SlotIndex Location = SI->getInstructionIndex(MI).getRegSlot();
                    std::string IndexBuffer;
                    raw_string_ostream IndexStream(IndexBuffer);
                    IndexStream << Location;
                    Instructions.push_back(std::make_pair(
                        TII->getName(MI.getOpcode()).str() + " " + IndexBuffer,
                        Instruction));

                    std::set<std::pair<Register, int>> LiveRegisters;
                    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
                        Register Reg = Register::index2VirtReg(i);
                        if (LIS->hasInterval(Reg) && !LIS->getInterval(Reg).empty() &&
                            LIS->getInterval(Reg).liveAt(Location) && !MRI->use_instructions(Reg).empty()) {
                          int NumberOfStore = getNumberOfStores(MRI, Reg);

                            /* bool OtherSequentialUse = false;
                            if (MRI->hasOneUse(Reg)) {
                                MachineLoop* L = MLI->getLoopFor(MRI->use_begin(Reg)->getParent()->getParent());
                                bool SameLoop = true;
                                for (auto& Def : MRI->def_instructions(Reg)) {
                                    MachineLoop* DefL = MLI->getLoopFor(Def.getParent());
                                    if (DefL != L) {
                                        SameLoop = false;
                                        break;
                                    }
                                }
                                OtherSequentialUse = SameLoop;
                            }

                            if (MRI)
                                OtherUseThanAfterCall |=
                                OtherSequentialUse ? RISCVILPRegAlloc::NO_OTHER_USE
                                : RISCVILPRegAlloc::NO_LOAD;*/
                          LiveRegisters.insert(
                              std::make_pair(Reg, NumberOfStore));
                        }
                    }
                    LiveRegsCallTmp.push_back(std::make_pair(Location, LiveRegisters));
                }
                else
                    Instructions.push_back(
                        std::make_pair(TII->getName(MI.getOpcode()).str(), Instruction));
            }
            else
                Instructions.push_back(std::make_pair("UNKNOWN", Instruction));
        }

    df_iterator<MachineDomTreeNode*> DI = df_begin(MDT->getRootNode());
    SlotIndex FirstIndex = SI->getZeroIndex();

    std::map<Register, std::vector<std::pair<SlotIndex, SlotIndex>>>
        PhysicalIntervals;

    for (df_iterator<MachineDomTreeNode*> DI = df_begin(MDT->getRootNode()),
        DE = df_end(MDT->getRootNode());
        DI != DE; ++DI) {
        MachineBasicBlock* MBB = DI->getBlock();
        for (auto& Reg : MBB->getLiveIns()) {
            PhysicalIntervals[Reg.PhysReg].push_back(
                std::make_pair(SI->getMBBStartIdx(MBB), SI->getMBBStartIdx(MBB)));
        }

        for (auto& MI : *MBB) {
            for (unsigned i = 0; i < MI.getNumOperands(); i++) {
                if (MI.getOperand(i).isReg() &&
                    MI.getOperand(i).getReg().isPhysical() &&
                    UnusedRegisters.find(MI.getOperand(i).getReg()) ==
                    UnusedRegisters.end()) {
                    if (MI.getOperand(i).isUse() && !MI.getOperand(i).isUndef())
                        PhysicalIntervals[MI.getOperand(i).getReg()].back().second =
                        SI->getInstructionIndex(MI);
                    else if (MI.getOperand(i).isDef()) {
                        if (PhysicalIntervals[MI.getOperand(i).getReg()].size() >= 1 &&
                            PhysicalIntervals[MI.getOperand(i).getReg()].back().second ==
                            SI->getMBBStartIdx(MBB))
                            PhysicalIntervals[MI.getOperand(i).getReg()].pop_back();
                        PhysicalIntervals[MI.getOperand(i).getReg()].push_back(
                            std::make_pair(SI->getInstructionIndex(MI).getRegSlot(),
                                SI->getInstructionIndex(MI).getRegSlot()));
                    }
                }
            }
        }
    }

    std::map<Register, LiveInterval*> PhysicalLIs;
    for (auto Interval : PhysicalIntervals) {
        PhysicalLIs[Interval.first] =
            &LIS->createEmptyInterval(Interval.first + MRI->getNumVirtRegs());
        for (auto& Segment : Interval.second) {
            VNInfo* VN = PhysicalLIs[Interval.first]->getNextValue(
                Segment.first, LIS->getVNInfoAllocator());
            PhysicalLIs[Interval.first]->addSegment(
                LiveRange::Segment(Segment.first, Segment.second, VN));
        }
    }

    std::vector<std::pair<Register, Register>> Interferences;
    for (int i = 0; i < VirtualRegisters.size(); i++) {
        LiveInterval& LI = LIS->getInterval(VirtualRegisters[i]);
        std::set<Register> CopySet = {};
        for (int c = 0; c < CopyRelated.size(); c++)
            if (CopyRelated[c].find(VirtualRegisters[i]) != CopyRelated[c].end()) {
                CopySet.insert(CopyRelated[c].begin(), CopyRelated[c].end());
            }

        for (int j = i + 1; j < VirtualRegisters.size(); j++) {
            if (CopySet.find(VirtualRegisters[j]) != CopySet.end())
                continue;
            LiveInterval& LJ = LIS->getInterval(VirtualRegisters[j]);
            if (LI.overlaps(LJ)) {
                Interferences.emplace_back(VirtualRegisters[i], VirtualRegisters[j]);
            }
        }

        for (auto& LIP : PhysicalLIs) {
            LiveInterval* LJ = LIP.second;
            for (auto& Segm : LJ->segments)
                if (LI.overlaps(Segm.start, Segm.end)) {
                    if (LI.segments.size() == 1) {
                        MachineInstr* MI = MRI->getVRegDef(VirtualRegisters[i]);
                        MachineInstr* MI2 = SI->getInstructionFromIndex(Segm.start);
                        if ((MI && MI->getOpcode() == RISCV::COPY &&
                            LIP.first == MI->getOperand(1).getReg() &&
                            SI->getInstructionIndex(*MI) == Segm.start) ||
                            (MI2 && MI2->getOpcode() == RISCV::COPY &&
                                VirtualRegisters[i] == MI2->getOperand(1).getReg())) {
                            bool OverlapOtherSegment = false;
                            for (auto Segm2 : LJ->segments) {
                                if (Segm2 != Segm && LI.overlaps(Segm2.start, Segm2.end)) {
                                    OverlapOtherSegment = true;
                                    break;
                                }
                            }
                            if (!OverlapOtherSegment)
                                continue;
                        }
                    }
                    Interferences.emplace_back(VirtualRegisters[i], LIP.first);
                    break;
                }
        }
    }


    std::map<Register, Register> InternalRegMap;
    if (!solveNativeRegAlloc(MF, *MRI, *TRI, *VRM, VirtualRegisters,
                             Instructions, Interferences, LiveRegsCallTmp,
                             CSDefPhysicalRegisters, canReturn(MF.getFunction()),
                             MFI->estimateStackSize(MF) >= 512, InternalRegMap))
        return false;

    std::set<std::pair<Register, int>> StackSlotRegs;
    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
        Register Reg = Register::index2VirtReg(i);
        if (VRM->getStackSlot(Reg) != VirtRegMap::NO_STACK_SLOT) {
            StackSlotRegs.insert(std::make_pair(Reg, VRM->getStackSlot(Reg)));
        }

    }

    auto AllocatedReg = [&](Register R) {
        if (R.isPhysical())
            return R;
        auto It = InternalRegMap.find(R);
        return It == InternalRegMap.end() ? Register() : It->second;
    };
    const std::set<unsigned> CommutativeInstructionsOpcodes = {
        RISCV::ADD, RISCV::XOR, RISCV::AND, RISCV::OR};
    for (auto &MBB : MF)
        for (auto &MI : MBB)
            if (CommutativeInstructionsOpcodes.count(MI.getOpcode()) &&
                MI.getNumOperands() >= 3 && MI.getOperand(0).isReg() &&
                MI.getOperand(1).isReg() && MI.getOperand(2).isReg()) {
                Register Dest = AllocatedReg(MI.getOperand(0).getReg());
                if (Dest && Dest == AllocatedReg(MI.getOperand(2).getReg()) &&
                    Dest != AllocatedReg(MI.getOperand(1).getReg()))
                {
                    Register R = MI.getOperand(1).getReg();
                    unsigned Sub = MI.getOperand(1).getSubReg();
                    bool Kill = MI.getOperand(1).isKill();
                    bool Undef = MI.getOperand(1).isUndef();
                    MI.getOperand(1).setReg(MI.getOperand(2).getReg());
                    MI.getOperand(1).setSubReg(MI.getOperand(2).getSubReg());
                    MI.getOperand(1).setIsKill(MI.getOperand(2).isKill());
                    MI.getOperand(1).setIsUndef(MI.getOperand(2).isUndef());
                    MI.getOperand(2).setReg(R);
                    MI.getOperand(2).setSubReg(Sub);
                    MI.getOperand(2).setIsKill(Kill);
                    MI.getOperand(2).setIsUndef(Undef);
                }
            }

    std::set<std::pair<MachineInstr*, Register>> NewMIs;
    std::set<Register> IgnoredRegisters;
    std::map<Register, unsigned> FrameIndices;

    for (auto &LiveRegisters : LiveRegsCallTmp) {
        MachineInstr* MI = SI->getInstructionFromIndex(LiveRegisters.first);
        // avoid tail calls
        bool IsCall = MI->isCall() && !MI->isReturn();
        std::set<std::tuple<Register, Register, unsigned>> LivePhys = {};

        if (IsCall) {
            for (auto Reg : LiveRegisters.second) {
                Register PhysReg = InternalRegMap[Reg.first];
                if (IsCall && !TRI->isCalleeSavedPhysReg(PhysReg.asMCReg(), MF) &&
                    LIS->getInterval(Reg.first).liveAt(LiveRegisters.first)) {
                  MachineInstr *ReloadAfter = MI->getNextNode();
                  MachineBasicBlock &MBB = *MI->getParent();

                  while (ReloadAfter != nullptr && ReloadAfter != MBB.end() &&
                         ((ReloadAfter->isCopy() &&
                           ReloadAfter->getOperand(1).isReg() &&
                           ReloadAfter->getOperand(1).getReg().isPhysical() &&
                           ReloadAfter->getOperand(1).getReg() >= RISCV::X10 &&
                           ReloadAfter->getOperand(1).getReg() <= RISCV::X17) ||
                          ReloadAfter->getOpcode() == RISCV::ADJCALLSTACKUP))
                    ReloadAfter = ReloadAfter->getNextNode(); 
                  Register VirtReg = Reg.first;
                  int NumberOfStores = Reg.second;

                  const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
                  MachineInstr *Def = MRI->getVRegDef(VirtReg);

                  if (NumberOfStores) {
                    if (FrameIndices.find(VirtReg) == FrameIndices.end()) {
                      unsigned Size = TRI->getSpillSize(RC);
                      Align Alignment = TRI->getSpillAlign(RC);
                      int FrameIdx =
                          MFI->CreateSpillStackObject(Size, Alignment);
                      FrameIndices[VirtReg] = FrameIdx;
                      for (auto &Def : MRI->def_instructions(VirtReg)) {
                        TII->storeRegToStackSlot(*Def.getParent(),
                                                 Def.getNextNode() != MBB.end()
                                                     ? Def.getNextNode()
                                                     : MBB.end(),
                                                 PhysReg, /* Kill = */ false,
                                                 FrameIdx, &RC, VirtReg);
                        SI->insertMachineInstrInMaps(*Def.getNextNode());
                      }
                    }
                    TII->loadRegFromStackSlot(MBB, ReloadAfter, PhysReg,
                                              FrameIndices[VirtReg], &RC,
                                              PhysReg);
            if (ReloadAfter && ReloadAfter != MBB.end())
                        SI->insertMachineInstrInMaps(*ReloadAfter->getPrevNode());
                    else
                        SI->insertMachineInstrInMaps(*MBB.getLastNonDebugInstr());
                  } else {
                    MachineInstr *NewMI = MF.CloneMachineInstr(Def);
                    //  NewMIs.insert(std::make_pair(NewMI, PhysReg));
                    if (!ReloadAfter || ReloadAfter == MBB.end()) {
                      MBB.push_back(NewMI);
                    } else {
                      MBB.insert(ReloadAfter, NewMI);
                    }
                    SI->insertMachineInstrInMaps(*NewMI);
                  }
                }
            }

        }
    }

    for (auto FI : FrameIndices) 
      computeLiveStackInterval(MF, FI.second, *MRI->getRegClass(FI.first), TII, SI, LS);

    VRM->init(MF);

    for (auto Reg : InternalRegMap)
        VRM->assignVirt2Phys(Reg.first, Reg.second);

    for (auto Reg : StackSlotRegs)
        VRM->assignVirt2StackSlot(Reg.first, Reg.second);

    for (auto& MBB : MF)
        for (auto& MI : MBB)
            for (auto& Op : MI.operands())
                if (Op.isReg())
                    Op.setIsOptimalAllocated(true);

    return true;
}

bool RISCVILPRegAlloc::runOnMachineFunction(MachineFunction & MF) {
    ST = &MF.getSubtarget<RISCVSubtarget>();
    TII = ST->getInstrInfo();
    MRI = &MF.getRegInfo();
    TRI = MRI->getTargetRegisterInfo();
    MFI = &MF.getFrameInfo();
    LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    SI = &getAnalysis<SlotIndexesWrapperPass>().getSI();
    MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
    MPDT = &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
    VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    LS = &getAnalysis<LiveStacksWrapperLegacy>().getLS();

    return applyILP(MF);
}

char RISCVILPRegAlloc::ID = 0;


INITIALIZE_PASS_BEGIN(RISCVILPRegAlloc, DEBUG_TYPE,
    "RISCV ILP RegAlloc", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariablesWrapperLegacy)
INITIALIZE_PASS_END(RISCVILPRegAlloc, DEBUG_TYPE,
    "RISCV ILP RegAlloc", false, false)


FunctionPass* llvm::createRISCVILPRegAlloc() {
   return new RISCVILPRegAlloc();
}
