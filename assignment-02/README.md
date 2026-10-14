# Assignment 2 — Optimised Matrix Multiplication and Sparse Matrices

This project answers the two research questions of the assignment:

> **RQ1 (dense optimisation):** *How do selected optimisations affect the
> runtime and memory of dense multiplication as the matrix size \(n\) grows?*
>
> **RQ2 (sparse):** *Under which tested sparsity conditions does a sparse CSR
> representation outperform a dense method on matched logical inputs, and where
> do indexing, conversion and output-densification costs erode the advantage?*

Part A compares **three sequential dense methods** — the validated **i-j-k
baseline** reused from assignment 1, an **i-k-j loop reordering** and **cache
blocking** — with the same inputs, numeric type and timing boundary. Part B
adds a **sparse-sparse CSR multiplication** (Gustavson's row-wise algorithm,
own implementation) and compares it against a dense method on **identical
matrices** at declared densities {1, 5, 10, 20, 50}%.

---

## 1. Reuse from assignment 1

Each submission must stand on its own and identify what it reuses. This
project reuses from `assignment-01/` (and copies it in, so the folder is
self-contained):

- the validated `TripleLoopMultiplier` (i-j-k) baseline and its JUnit tests;
- the `DenseMatrix` / `InputMatrixLoader` / `MatrixMultiplier` domain;
- the kernel-only `BenchmarkHarness`, `TimingSample`, `PeakResidentMemory`,
  `PeakHeap`, `BudgetExceededException` benchmark infrastructure;
- the shared dense inputs `data/inputs/matrixA_n.bin` / `matrixB_n.bin`
  (same bytes and seed 42 as assignment 1; `n = 1500` added for this study);
- the Python analysis toolchain (numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2)
  and the plotting conventions.

---

## 2. Environment

**Physical hardware (the student's machine):**

| Item | Value |
|---|---|
| CPU | Intel Core i7-10750H @ 2.6 GHz (6 cores / 12 threads) |
| Memory | 16 GB DDR4 |
| Host OS | Windows 11 |

**Execution environment (where every build, test, benchmark and plot was run):**

| Item | Value |
|---|---|
| OS | WSL2 — Ubuntu 26.04 LTS, kernel `6.18.33.2-microsoft-standard-WSL2` |
| RAM visible to the VM | ~7.7 GB |
| Logical processors | 12 |
| Java | OpenJDK 17.0.20.1, Maven 3.9.12 |
| Python | 3.14.4 project venv `.venv` — numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2 |

> Native Windows binaries are blocked by Smart App Control on this machine, so
> the whole campaign runs inside WSL2.

---

## 3. Repository layout

```
pom.xml                       Maven project (Java 17, JUnit 5, shaded runner jar)
src/main/java/...             Java: dense kernels, CSR domain and kernel, benchmark runner
src/test/java/...             Java: JUnit equivalence and CSR correctness tests
python/generate_inputs.py     Dense input generator (reused from assignment 1, seed 42)
python/generate_sparse_inputs.py  Declared-density sparse inputs + manifest
python/correctness_check.py   Independent checksum check of the jar vs NumPy
python/run_experiments.py     Campaign orchestrator (Part A + Part B)
python/plot_results.py        Regenerates every report figure from the CSVs
data/inputs/                  Shared dense .bin inputs (float64 LE, row-major)
data/sparse/                  Declared-density .bin inputs + inputs_manifest.csv
data/reference/               5x5 reference matrices (reused from assignment 1)
data/results/                 dense_time_raw.csv, sparse_time_raw.csv,
                              memory_raw.csv, run_log.txt (campaign)
report/figures/*.pdf          Report figures (regenerated from the CSVs alone)
```

---

## 4. The four methods

| Declared name (`method` column) | What it does | Expected effect |
|---|---|---|
| `triple_loop_ijk_dense` | Baseline: i-j-k triple loop, column access to `b` is strided (reused from assignment 1) | Reference |
| `triple_loop_ikj_dense` | Loop reordering to i-k-j: `b` and `out` are swept contiguously | Fewer cache misses, same \(O(n^3)\) work |
| `blocked_ijk_dense` | i-k-j restricted to 64×64 tiles (32 KB per tile of `b`) | Better cache reuse at large \(n\); block size fixed and declared |
| `sparse_csr_gustavson` | Gustavson row-wise sparse-sparse product in canonical CSR | Work scales with nonzero counts, not \(n^3\) |

All dense methods are sequential, use float64 (`double`) and overwrite the
whole output; parallelism is out of scope here (assignment 3).

**CSR layout and storage:** `values` (8 B float64) + `columnIndices` (4 B int)
per nonzero, `rowPointers` (4 B int) per row ⇒ theoretical storage
\(12 \cdot \mathrm{nnz} + 4(n+1)\) bytes, versus \(24n^2\) bytes for three
dense \(n \times n\) matrices.

---

## 5. Experimental protocol (summary)

- **Timing boundary:** kernel only. The timer surrounds the multiplication
  (dense: the triple loops into a pre-allocated buffer; sparse: construction of
  the output CSR). Input loading, dense→CSR conversion and checksums are
  outside. `System.nanoTime()` throughout.
- **Repetitions:** 3 warm-ups (recorded with `warmup=true`, excluded) + 7
  measured runs; every raw sample is kept; report median and IQR. Batches for
  multiplications under ~2 ms (`batch_size` column).
- **Budgets:** 60 s per multiplication, ~2 GB memory. Overruns are results
  (exit code 3, logged), never invented times.
- **Sparse inputs:** densities {1, 5, 10, 20, 50}% of nonzeros; positions drawn
  uniformly without replacement, values uniform [0, 1), seeds recorded per
  configuration in `data/sparse/inputs_manifest.csv`.
- **Matched comparison:** the CSR kernel and the dense comparison method
  multiply the identical `.bin` bytes for every (size, density) configuration;
  rows are labelled `representation=csr|dense`. Above the largest fully
  compared size the dense comparison runs only at the declared probe density
  10%, because dense runtime does not depend on entry density.
- **Correctness:** JUnit equivalence of every optimised kernel against the
  baseline (identity, zero, known-answer 3×3, seeded random matrices) plus
  `.venv/bin/python python/correctness_check.py`, which compares the compiled
  jar's checksums with a NumPy float64 reference (`rtol=1e-9`, `atol=1e-12`).

### Raw CSV schemas

- `dense_time_raw.csv`:
  `size,method,numeric_type,repetition,warmup,batch_size,time_ns_per_multiply,checksum`
- `sparse_time_raw.csv`:
  `size,density,representation,method,numeric_type,repetition,warmup,batch_size,nnz_a,nnz_b,nnz_c,conversion_time_ns,kernel_time_ns_per_multiply,checksum`
- `memory_raw.csv`:
  `size,density,method,metric,tool,value_bytes,theoretical_bytes`

---

## 6. How to build, verify and run

All commands run from `assignment-02/` inside WSL2.

### Toolchain

```bash
# one-time: recreate the Python environment (system Python has no ensurepip)
python3 -m venv --without-pip .venv
../assignment-01/.venv/bin/python -m pip --python .venv install -r python/requirements.txt
```

### Build and verify (gates)

```bash
mvn -q verify                                   # compiles + JUnit tests green
mvn -q -DskipTests package                      # shaded runner jar
.venv/bin/python python/correctness_check.py    # jar checksums vs NumPy
.venv/bin/python -m compileall python           # Python syntax gate
```

### Inputs

```bash
# dense inputs (n = 1500 added for this study; smaller sizes are the same
# files as assignment 1, regenerated identically from seed 42)
.venv/bin/python python/generate_inputs.py --sizes 1500

# declared-density sparse inputs + manifest
.venv/bin/python python/generate_sparse_inputs.py
```

### Campaign

```bash
# full campaign (Part A dense + Part B sparse), fresh CSVs
.venv/bin/python python/run_experiments.py

# options: --part a|b|all, --sizes, --sparse-sizes, --densities,
#          --dense-comparison-method, --warmup, --repetitions, --append
```

### Figures

```bash
.venv/bin/python python/plot_results.py   # writes report/figures/*.pdf
```

---

## 7. Deliverables

- `Assignment02_Lorenzo_Report.pdf` — 4–6 page report (LaTeX, English,
  compiled on Overleaf from `report/main.tex`).
- `Assignment02_Lorenzo_Materials.zip` — source code, README, raw CSVs,
  input manifest and scripts from this folder.

Both files are attached to the `assignment-02` release of the global
repository (`aythamilorenzo/bigdata-individual-assignments`); the assignment is
marked with the annotated tag `assignment-02`.
