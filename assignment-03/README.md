# Assignment 3 — Parallel Matrix Multiplication

Thread-count scaling study of dense square matrix multiplication
(float64, n x n) on shared memory. The research question: **how does the
thread count affect the runtime of a shared-memory implementation, and when
does parallel execution help?**

The sequential baseline and the parallel method are the *same* cache-blocked
i-k-j kernel (tile side 64): the parallel implementation partitions the
kernel's i-tiles across a fixed pool of `p` worker threads, so every output
entry is written by exactly one thread and accumulates over k in the same
ascending order as the sequential kernel (element-wise identical results,
verified by tests and by an out-of-JVM NumPy cross-check).

## Reuse from earlier assignments

Each earlier artefact reused here is identified with its version:

- **Assignments 1 and 2** — the dense matrix domain (`DenseMatrix`), the
  input loader, the kernel-only timing harness (`BenchmarkHarness`), the
  peak-memory tools (`PeakResidentMemory` = `VmHWM`, `PeakHeap`), the CSV
  writer pattern and the benchmark CLI skeleton (assignment 2).
- **Assignment 2** — the validated cache-blocked kernel `BlockedMultiplier`
  (block size 64) is the sequential baseline of this assignment; its tile
  kernel is reused verbatim by the parallel method.
- **Assignments 1 and 2** — the shared dense inputs `data/inputs/*.bin`
  (same bytes: the generator seeds NumPy PCG64 with `[42, n]`; sizes 500,
  1000 and 1500 are byte-identical to the assignment 2 campaign inputs,
  size 2000 is new). The reference matrices `data/reference/*.bin` are the
  5x5 matrices of assignments 1 and 2.

## Layout

```
pom.xml                          Maven build (Java 17, shaded jar)
src/main/java/es/ulpgc/bigdata/matrixmul/
  BlockedMultiplier.java         sequential baseline: cache-blocked i-k-j
  ParallelBlockedMultiplier.java same kernel partitioned across p threads
  DenseMatrix.java, MatrixMultiplier.java, InputMatrixLoader.java
  benchmark/
    BenchmarkMain.java           CLI entry point
    BenchmarkHarness.java        kernel-only timing (warm-up, batches, budget)
    Multipliers.java             method-name registry (one vocabulary)
    ResultsCsvWriter.java        raw CSV schemas (threads column)
    HardwareInfo.java            cores / logical processors / affinity facts
    PeakHeap.java, PeakResidentMemory.java, TimingSample.java
src/test/java/                   JUnit equivalence and harness tests
python/
  generate_inputs.py             shared dense inputs (seed 42, PCG64)
  run_experiments.py             campaign orchestrator
  correctness_check.py           jar vs NumPy cross-check
  plot_results.py                raw CSVs -> report figures (PDF)
  requirements.txt               pinned analysis toolchain
data/
  inputs/                        .bin matrices (not committed; regenerate)
  reference/                     5x5 reference matrices (committed)
  results/                       raw CSVs + run_log.txt (committed)
report/                          LaTeX report (main.tex, figures/)
```

## Environment

Everything runs inside WSL2 (Ubuntu) on the student's machine. Exact
versions used for the campaign:

- OpenJDK 17.0.20.1, Maven 3.9.12
- Python 3.14.4 project venv `.venv` with numpy 2.5.3, pandas 3.0.6,
  matplotlib 3.11.2 (same toolchain as assignments 1 and 2)

Hardware of the campaign machine: Intel Core i7-10750H (6 physical cores,
12 logical processors), 16 GB RAM; the JVM sees 12 processors. The
affinity mask and the exact facts recorded per run are logged by the CLI
(`hardware:` line in `data/results/run_log.txt`).

## Build and test

```bash
mvn -q verify          # compiles + JUnit equivalence tests
mvn -q -DskipTests package   # shaded jar only
```

## Correctness

```bash
.venv/bin/python python/correctness_check.py
```

Runs the compiled jar on the 5x5 reference pair and on the shared n=500
input with the sequential method and the parallel method at p = 1, 2, 3, 4,
8, and compares the printed checksums against a NumPy float64 reference
(`rtol=1e-9`, `atol=1e-12`). In-JVM, `ParallelMultiplierEquivalenceTest`
additionally asserts that the parallel kernel reproduces the sequential
kernel entry-wise across sizes and thread counts and that the checksum is
invariant with p (evidence of safe access to the shared output).

## Campaign

```bash
.venv/bin/python python/generate_inputs.py     # if data/inputs is empty
.venv/bin/python python/run_experiments.py     # build + all configurations
```

Declared configuration of the campaign:

- **Sizes:** n in {500, 1000, 1500, 2000}
- **Methods:** `blocked_ijk_sequential` (baseline, threads = 1) and
  `blocked_ijk_parallel` at p in {1, 2, 4, 6, 8, 12, 24} (p = 6 is the
  physical-core count, p = 12 the logical-processor count, p = 24
  oversubscribes the hardware on purpose to identify the effect)
- **Timing boundary:** kernel only (task dispatch + tile computation +
  join); thread creation and input loading are outside it
- **Protocol:** 3 warm-up runs (excluded, recorded) + 7 measured
  repetitions per configuration; `System.nanoTime()`; median and IQR
  reported; batches for multiplications shorter than ~2 ms; budgets 60 s
  and ~2 GB; every raw sample kept
- **Derived metrics:** speedup `S(p) = Tseq(n) / Tp(n, p)` and efficiency
  `E(p) = S(p) / p` from medians

Raw CSVs (`data/results/`):

```
parallel_time_raw.csv: size,threads,method,numeric_type,repetition,warmup,batch_size,time_ns_per_multiply,checksum
memory_raw.csv:        size,threads,method,metric,tool,value_bytes,theoretical_bytes
```

Runner exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded (a
result, not a failure), 4 missing input files.

## Figures

```bash
.venv/bin/python python/plot_results.py
```

Regenerates every figure of the report from the raw CSVs alone into
`report/figures/` (PDF, for LaTeX).

## Deliverables

- `Assignment03_Lorenzo_Report.pdf` — research report (attached to the
  `assignment-03` GitHub release).
- `Assignment03_Lorenzo_Materials.zip` — source, this README, raw CSVs,
  input manifest and scripts (attached to the same release).
