# Assignment 4 — Distributed Matrix Multiplication

Thread-count scaling study of dense square matrix multiplication
(float64, n × n) comparing three implementations of the *same* cache-blocked
i-k-j kernel (tile side 64): the sequential baseline, its thread-pool
distribution (assignment 3) and its Apache Spark distribution. The research
question: **how does the thread count affect runtime, speedup, efficiency,
framework overhead and memory of a distributed shared-memory
implementation, and when does distributed execution help?**

## Method families

All three methods are element-wise identical (same tiles, same k order):

| method | mechanism | threads |
|---|---|---|
| `blocked_ijk_sequential` | calling thread only | 1 |
| `blocked_ijk_parallel` | fixed `ThreadPoolExecutor` pool (assignment 3) | p |
| `blocked_ijk_distributed` | Spark `local[p]`: row-block partitions, broadcast B, collect blocks | p |

**Workers vs threads vs machines (local execution):** Spark `local[p]`
starts **p worker threads inside a single JVM process**, sharing one address
space and one garbage-collected heap. It does **not** launch isolated worker
JVM processes nor use more than one physical machine. A Spark standalone
cluster would launch one worker process per machine; that configuration is
out of scope here (instructor-approved local execution) and the report
labels this limitation explicitly.

## Reuse from earlier assignments

- **Assignments 1–3** — dense matrix domain (`DenseMatrix`), input loader,
  peak-memory tools (`PeakResidentMemory` = `VmHWM`, `PeakHeap`), CSV
  patterns, benchmark CLI skeleton.
- **Assignment 2** — validated cache-blocked kernel `BlockedMultiplier`
  (tile 64): sequential baseline here; its tile order is reused verbatim by
  both distributed methods.
- **Assignment 3** — `ParallelBlockedMultiplier` (thread pool) reused
  verbatim as the thread-pool baseline of this assignment.
- **Assignments 1–2** — shared dense inputs (`data/inputs/*.bin`, NumPy
  PCG64 seeded `[42, n]`): sizes 500–2000 byte-identical to assignment 2;
  3000 and 4000 generated with the same generator.

## Layout

```
pom.xml                          Maven build (Java 17, shaded jar, Spark 3.5.1)
src/main/java/es/ulpgc/bigdata/matrixmul/
  BlockedMultiplier.java         sequential baseline: cache-blocked i-k-j
  ParallelBlockedMultiplier.java thread-pool baseline (assignment 3)
  DenseMatrix.java, MatrixMultiplier.java, InputMatrixLoader.java
  benchmark/                     timing, CSV writers, hardware info, memory
  distributed/
    DistributedMultiplier.java   Spark row-block partition + broadcast + collect
    DistributedBenchmarkMain.java per-configuration benchmark CLI
src/test/java/                   JUnit equivalence tests (14 tests)
python/
  generate_inputs.py             shared dense inputs (seed 42, PCG64)
  run_experiments.py             campaign orchestrator (one JVM per config)
  correctness_check.py           jar vs NumPy cross-check (all 3 methods)
  plot_results.py                raw CSVs -> report figures (PNG)
  requirements.txt               pinned analysis toolchain
data/
  inputs/                        .bin matrices (not committed; regenerate)
  reference/                     5x5 reference matrices (committed)
  results/                       raw CSVs + run_log.txt (committed)
report/                          LaTeX report (main.tex, figures/)
```

## Environment

WSL2 (Ubuntu) on the student's machine. Exact versions:

- OpenJDK 17.0.20.1, Maven 3.9.12, Apache Spark 3.5.1 (compile scope)
- Python 3.14.4 project venv `.venv` (numpy, pandas, matplotlib, seaborn)

Hardware: Intel Core i7-10750H (6 physical cores, 12 logical processors),
16 GB RAM; JVM sees 12 processors. The `hardware:` line of
`data/results/run_log.txt` records model, cores, logical processors,
affinity mask and JVM-visible processors.

## Build and test

```bash
mvn -q verify                  # compiles + 14 JUnit equivalence tests
mvn -q -DskipTests package     # shaded jar
```

## Correctness

```bash
.venv/bin/python python/correctness_check.py
```

Runs the jar with all three methods on the 5x5 reference pair and the shared
n=500 input and compares the printed checksums against a NumPy float64
reference (`rtol=1e-9`, `atol=1e-12`), asserting checksum invariance across
methods and thread counts.

## Campaign

```bash
.venv/bin/python python/generate_inputs.py     # if data/inputs is empty
.venv/bin/python python/run_experiments.py     # all configurations
```

Declared configuration:

- **Sizes:** n ∈ {500, 1000, 1500, 2000, 3000, 4000}. n = 8000 is excluded
  on purpose: the sequential baseline needs ~230 s per multiplication,
  exceeding the 60 s budget.
- **Threads:** p ∈ {1, 2, 4, 6, 8, 12, 24} — 6 = physical cores, 12 =
  logical processors, 24 = deliberate oversubscription (identified as such).
- **Methods:** `blocked_ijk_sequential` (p = 1), `blocked_ijk_parallel` and
  `blocked_ijk_distributed` at every p.
- **Timing boundary:** end-to-end = one multiplication call (for Spark:
  broadcast of B + job + driver-side assembly). Input loading and the
  per-configuration setup (SparkContext / pool creation) are outside it and
  reported in their own column. Monotonic clock only (`System.nanoTime()`).
- **Protocol:** 3 warm-ups (recorded, excluded) + 7 measured repetitions,
  one JVM per configuration, every raw sample kept, median and IQR.
- **Budgets:** 60 s per multiplication (exceedance recorded as a result),
  ~2 GB peak memory; JVM capped at `-Xmx3g`.
- **Memory:** peak RSS from `/proc/self/status` (`VmHWM`) and Java peak heap
  (`MemoryPoolMXBean`), with theoretical dense storage (24 n² bytes)
  reported alongside as a different quantity.

Raw CSVs (`data/results/`):

```
distributed_time_raw.csv: size,threads,method,numeric_type,repetition,warmup,
    batch_size,end_to_end_ns,broadcast_ns,job_wall_ns,assemble_ns,
    task_compute_ns,checksum
memory_raw.csv:           size,threads,method,metric,tool,value_bytes,
    theoretical_bytes
```

`task_compute_ns` is the worker-side compute time summed across tasks; it is
**not** additive with the driver-side columns (workers run concurrently
inside the job stage) and is reported as a separate quantity.

Runner exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded (a
result, not a failure), 4 missing input files.

## Figures

```bash
.venv/bin/python python/plot_results.py
```

Regenerates all six figures from the raw CSVs alone: median time, speedup,
efficiency, framework overhead (Spark vs thread pool), Spark phase
breakdown (stacked) and peak RSS.

## Deliverables

- `Assignment04_Lorenzo_Report.pdf` — research report.
- `Assignment04_Lorenzo_Materials.zip` — source, this README, raw CSVs,
  input manifest and scripts.
