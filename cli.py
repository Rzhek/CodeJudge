#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import tempfile
import shutil
import difflib
import textwrap
import time
from pathlib import Path
from typing import List, Tuple, Optional

# ---------------- Colors ----------------
class Color:
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[1;36m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def color(text, code):
    return f"{code}{text}{Color.RESET}"

def status_text(status: str) -> str:
    if status == "PASS":
        return color("PASS", Color.GREEN)
    elif status == "FAIL":
        return color("FAIL", Color.RED)
    return color(status, Color.YELLOW)

# ---------------- File helpers ----------------
IN_EXTS = {".in", ".inp", ".input"}
OUT_EXTS = {".out", ".ans", ".expected", ".res"}

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def collect_files(root: Path) -> Tuple[List[Path], List[Path]]:
    inputs, outputs = [], []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IN_EXTS:
            inputs.append(p)
        elif ext in OUT_EXTS:
            outputs.append(p)
        elif "in" in p.name.lower():
            inputs.append(p)
        elif "out" in p.name.lower() or "ans" in p.name.lower():
            outputs.append(p)
    return sorted(inputs, key=lambda p: natural_key(str(p))), sorted(outputs, key=lambda p: natural_key(str(p)))

def stem_key(p: Path) -> str:
    return re.sub(r"[_\-\s]", "", p.stem.lower())

def num_key(p: Path) -> Optional[str]:
    m = re.search(r"(\d+)", p.stem)
    return m.group(1) if m else None

def pair_tests(inputs: List[Path], outputs: List[Path]) -> List[Tuple[Path, Path]]:
    pairs = []
    used_out = set()

    def try_match(keyfunc):
        nonlocal pairs, used_out
        for inp in inputs:
            k = keyfunc(inp)
            if not k:
                continue
            for outp in outputs:
                if outp in used_out:
                    continue
                if keyfunc(outp) == k:
                    pairs.append((inp, outp))
                    used_out.add(outp)
                    break

    try_match(lambda p: p.stem.lower())
    try_match(num_key)
    try_match(stem_key)

    remaining_in = [i for i in inputs if all(i != a for a, _ in pairs)]
    remaining_out = [o for o in outputs if o not in used_out]

    if len(remaining_in) == len(remaining_out):
        for a, b in zip(remaining_in, remaining_out):
            pairs.append((a, b))
    else:
        if remaining_in:
            print(color("\n[warn] Unmatched inputs:", Color.YELLOW))
            for i in remaining_in:
                print("   •", i)
        if remaining_out:
            print(color("\n[warn] Unmatched outputs:", Color.YELLOW))
            for o in remaining_out:
                print("   •", o)
    return pairs

# ---------------- Compile & run ----------------
def compile_cpp(cpp: Path, bin_path: Path, std="c++17"):
    print(color(f"[build] g++ -std={std} -O2 {cpp}", Color.CYAN))
    try:
        subprocess.run(["g++", "-std=" + std, "-O2", str(cpp), "-o", str(bin_path)], check=True)
    except subprocess.CalledProcessError:
        print(color("[error] Compilation failed!", Color.RED))
        sys.exit(1)

def run_case(bin_path: Path, input_path: Path, timeout_sec: float = 2.0):
    with input_path.open("rb") as f:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(bin_path)],
                input=f.read(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            elapsed = time.perf_counter() - start
            return proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace"), elapsed
        except subprocess.TimeoutExpired:
            return 124, "", f"TIMEOUT after {timeout_sec}s", timeout_sec

# ---------------- Comparison ----------------
def normalize_text(s: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in s.splitlines()]
    return "\n".join(line for line in lines if line.strip())

def float_equal(a: str, b: str, eps: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= eps or abs(float(a)-float(b)) / abs(float(a)) <= eps
    except ValueError:
        return False

def compare_outputs(expected: str, got: str, eps: float) -> bool:
    a_lines = normalize_text(expected).splitlines()
    b_lines = normalize_text(got).splitlines()
    if len(a_lines) != len(b_lines):
        return False
    for x, y in zip(a_lines, b_lines):
        ax, ay = x.split(), y.split()
        if len(ax) != len(ay):
            return False
        for tok1, tok2 in zip(ax, ay):
            if tok1 == tok2:
                continue
            if eps > 0 and float_equal(tok1, tok2, eps):
                continue
            return False
    return True

def print_diff(expected: str, got: str, expected_name: str, got_name: str):
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        got.splitlines(keepends=True),
        fromfile=expected_name,
        tofile=got_name,
        n=3
    )
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            print(color(line, Color.GREEN), end='')
        elif line.startswith('-') and not line.startswith('---'):
            print(color(line, Color.RED), end='')
        else:
            print(line, end='')

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Run all test cases for a C++ file with colorful output.")
    parser.add_argument("cpp_file", help="Path to the C++ source file")
    parser.add_argument("test_folder", help="Folder containing .in/.out files (recursively)")
    parser.add_argument("--std", default="c++17", help="C++ standard (default: c++17)")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout per test (seconds)")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for float comparison (default: 1e-6; use 0 for exact)")
    args = parser.parse_args()

    cpp = Path(args.cpp_file).resolve()
    root = Path(args.test_folder).resolve()

    if not cpp.exists():
        print(color("[error] C++ file not found!", Color.RED))
        sys.exit(1)
    if not root.is_dir():
        print(color("[error] Test folder not found!", Color.RED))
        sys.exit(1)

    tmpdir = Path(tempfile.mkdtemp(prefix="cpp-runner-"))
    bin_path = tmpdir / "a.out"

    compile_cpp(cpp, bin_path, args.std)

    inputs, outputs = collect_files(root)
    if not inputs or not outputs:
        print(color("[error] No .in/.out files found!", Color.RED))
        sys.exit(1)

    pairs = pair_tests(inputs, outputs)
    if not pairs:
        print(color("[error] Could not match any input/output pairs.", Color.RED))
        sys.exit(1)

    print(color(f"\n[info] Found {len(pairs)} test case(s).", Color.CYAN))
    print(color(f"[info] Epsilon tolerance: {args.eps}\n", Color.CYAN))
    passed = 0

    for i, (inp, outp) in enumerate(pairs, 1):
        exp = outp.read_text(encoding="utf-8", errors="replace")
        code, got, err, secs = run_case(bin_path, inp, args.timeout)
        ok = (code == 0 and compare_outputs(exp, got, args.eps))
        status = "PASS" if ok else "FAIL"

        print(f"=== Case {i}/{len(pairs)} :: {inp.name} → {outp.name} :: {status_text(status)}  ({secs:.3f}s)")
        if err.strip():
            print(color("stderr:\n", Color.YELLOW) + textwrap.indent(err.strip(), "  ") + "\n")
        if not ok:
            print_diff(exp, got, str(outp), "your_output")
            print()
        if ok:
            passed += 1

    print()
    summary_color = Color.GREEN if passed == len(pairs) else Color.RED
    print(color(f"Summary: {passed}/{len(pairs)} passed", summary_color))

    shutil.rmtree(tmpdir, ignore_errors=True)
    sys.exit(0 if passed == len(pairs) else 2)

if __name__ == "__main__":
    main()
