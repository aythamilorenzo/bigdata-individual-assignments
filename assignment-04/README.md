# Assignment 4 — Distributed Matrix Multiplication and Vectorisation

Thread-count scaling study of dense square matrix multiplication
(float64, n × n) on shared memory using Apache Spark. The research question:
**how does the thread count affect the runtime, speedup, efficiency and
overhead of a distributed shared-memory implementation, and when does
distributed execution help?**

The distributed method is the *same* cache-blocked i-k-j kernel (tile side
64) from assignments 1-3, partitioned by row blocks across p Spark workers
in `local` mode: each worker computes a disjoint set of rows of the output,
so results are element-wise identical to the sequential and parallel
methods (verified by tests and an out-of-JVM NumPy cross-check).

## Reuse from earlier assignments

Each earlier artefact reused here is identified with its version:

- **Assignments 1, 2 and 3** — the dense matrix domain (`DenseMatrix`), the
  input loader, the kernel-only timing harness (`BenchmarkHarness`), the
  peak-memory tools (`PeakResidentMemory` = `VmHWM`, `PeakHeap`), the CSV
  writer pattern and the benchmark CLI skeleton (assignment 2).
- **Assignment 2** — the validated cache-blocked kernel `BlockedMultiplier`
  (block size 64) is the sequential baseline of this assignment; its tile
  kernel is reused verbatim by the parallel (assignment 3) and distributed
  methods.
- **Assignment 3** — the parallel implementation `ParallelBlockedMultiplier`
  is the thread-pool baseline of this assignment; the distributed method
  uses Spark instead of a thread pool.
- **Assignments 1 and 2** — the shared dense inputs `data/inputs/*.bin`
  (same bytes: the generator seeds NumPy PCG64 with `[42, n]`; sizes 500,
  1000 and 1500 are byte-identical to the assignment 2 campaign inputs,
  size 2000 is new). The reference matrices `data/reference/*.bin` are the
  5×5 matrices of assignments 1 and 2.

## Layout

```
pom.xml                          Maven build (Java 17, shaded jar)
src/main/java/es/ulpgc/bigdata/matrixmul/
  BlockedMultiplier.java         sequential baseline: cache-blocked i-k-j
  ParallelBlockedMultiplier.java thread-pool baseline (from assignment 3)
  DistributedMultiplier.java     Spark-based distributed implementation
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
  plot_results.py                raw CSVs -> report figures (PNG)
  consolidate_results.py         consolidate all CSVs into summary
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

- OpenJDK 17.0.20.1, Maven 3.9.12, Apache Spark 3.5.1
- Python 3.14.4 project venv `.venv` with numpy 2.5.3, pandas 3.0.6,
  matplotlib 3.11.2, seaborn 0.13.0 (same toolchain as assignments 1-3)

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
input with the sequential method, the parallel method and the distributed
method at p = 1, 2, 3, 4, 8, and compares the printed checksums against a
NumPy float64 reference (`rtol=1e-9`, `atol=1e-12`). In-JVM,
`DistributedMultiplierEquivalenceTest` additionally asserts that the
distributed kernel reproduces the sequential kernel entry-wise across sizes
and thread counts and that the checksum is invariant with p (evidence of
safe access to the shared output).

## Campaign

```bash
.venv/bin/python python/generate_inputs.py     # if data/inputs is empty
.venv/bin/python python/run_experiments.py     # build + all configurations
```

Declared configuration of the campaign:

- **Sizes:** n in {500, 1000, 1500, 2000}
- **Methods:** `blocked_ijk_sequential` (baseline, threads = 1),
  `blocked_ijk_parallel` (thread-pool baseline from assignment 3) and
  `blocked_ijk_distributed` at p in {1, 2, 4, 8, 12} (p = 8 is the
  physical-core count rounded, p = 12 the logical-processor count)
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
distributed_time_raw.csv: size,threads,method,numeric_type,repetition,warmup,batch_size,time_ns_per_multiply,checksum
memory_raw.csv:           size,threads,method,metric,tool,value_bytes,theoretical_bytes
```

Runner exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded (a
result, not a failure), 4 missing input files.

## Figures

```bash
.venv/bin/python python/plot_results.py
```

Regenerates every figure of the report from the raw CSVs alone into
`report/figures/` (PNG, for LaTeX).

## Deliverables

- `Assignment04_Lorenzo_Report.pdf` — research report (attached to the
  `assignment-04` GitHub release).
- `Assignment04_Lorenzo_Materials.zip` — source, this README, raw CSVs,
  input manifest and scripts (attached to the same release).
