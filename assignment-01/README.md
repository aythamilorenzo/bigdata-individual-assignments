# Assignment 1 — Basic Matrix Multiplication Benchmark (Python · Java · C · Go)

This project benchmarks a **basic dense matrix multiplication** implemented with a
straightforward **i-j-k triple loop** in four languages and answers the research
question:

> *How does the matrix size \(n\) affect the runtime and the peak memory of a
> basic \(O(n^3)\) triple-loop multiplication across the four implementations?*

Python, Java and C are the three implementations required by the assignment; the
Go implementation is a clearly labelled **extension** that follows the identical
algorithm and protocol.

---

## 1. Environment

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
| C compiler | gcc 15.2.0 (`-O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L`) |
| Java | OpenJDK 17.0.20.1, Maven 3.9.12 |
| Go | 1.27.2 (user-space install in `~/.local/go`) |
| Python | 3.14.4 project venv `.venv` — numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2 |

> Native Windows binaries are blocked by Smart App Control on this machine, so
> the whole campaign runs inside WSL2. No git repository is used for this project.

---

## 2. Repository layout

```
pom.xml                       Maven project (Java 17, JUnit 5, shaded runner jar)
src/main/java/...             Java: matrix, loader, kernel, benchmark runner
src/test/java/...             Java: JUnit correctness and harness tests
c/                            C11: kernel, benchmark runner, correctness test, build.sh
go/                           Go module: kernel (tripleloop) + cmd/benchmark runner
python/                       Python kernel, input generator, correctness check,
                              benchmark runner, campaign orchestrator, plotting
data/inputs/                  Shared little-endian float64 row-major .bin inputs
data/reference/               5x5 reference matrices + NumPy-computed expected C
data/results/                 time_raw.csv, memory_raw.csv, run_log.txt (campaign)
report/figures/*.pdf          Report figures (regenerated from the CSVs alone)
```

---

## 3. Experimental protocol (summary)

- **Algorithm:** dense triple loop, loop order **i-j-k**, numeric type
  **float64 (double)**, row-major storage, no BLAS/NumPy inside the measured
  kernel.
- **Shared inputs:** every language loads the same `data/inputs/matrixA_n.bin`
  and `matrixB_n.bin` (generated once by `python/generate_inputs.py`, seed 42,
  deterministic per size). Loading happens **outside** the timed region.
- **Timing boundary:** the timer surrounds only the triple loop that writes into
  a pre-allocated output buffer; a sequential row-major checksum is computed
  after timing so the compiler cannot eliminate the work. Monotonic clocks only:
  `System.nanoTime()` (Java), `clock_gettime(CLOCK_MONOTONIC)` (C),
  `time.Now()`/`time.Since` (Go), `time.perf_counter_ns()` (Python).
- **Warm-up and repetitions:** 3 warm-up runs (recorded with `warmup=true`,
  excluded from analysis) + **7 measured repetitions** per configuration; the
  analysis reports the **median and IQR**. When one multiplication is shorter
  than ~2 ms, a batch of \(K\) multiplications runs inside the timed region and
  the CSV stores time per multiplication together with `batch_size` \(K\).
- **Budgets:** 60 s per multiplication, ~2 GB memory. Budget exhaustion is a
  *result* (exit code 3), never replaced by an invented number.
- **Memory metric:** **peak resident set size (peak RSS)** read from
  `/proc/self/status` (`VmHWM`) in all four languages — one metric, one kernel
  counter, one tool. The theoretical array storage \(24n^2\) bytes (three
  float64 \(n \times n\) matrices) and, for Java, the peak heap are reported as
  *separate* quantities and never compared as if identical.
- **Correctness before benchmarking:** identity, zero, known-answer 3×3 and
  random cases validated against a NumPy float64 reference with
  `rtol=1e-9, atol=1e-12`; cross-language checksums match on the shared inputs
  (up to 1 ulp at \(n = 1000\)).
- **Sizes:** \(n \in \{10, 25, 50, 100, 200, 500, 1000\}\).

**Raw CSV schemas:**

```
time_raw.csv:    size,language,method,numeric_type,repetition,warmup,batch_size,time_ns_per_multiply,checksum
memory_raw.csv:  size,language,metric,tool,value_bytes,theoretical_bytes
```

`language` ∈ `python|java|c|go`, `method` = `triple_loop_ijk_dense`.
Runner exit codes: `0` ok, `2` invalid arguments, `3` time budget exceeded,
`4` missing input files.

---

## 4. Reproducing everything

All commands run from the project root inside WSL2.

### 4.1 Toolchain (once)

```bash
# Go 1.27.2, user space, checksum-verified
curl -sL -o /tmp/go.tgz https://go.dev/dl/go1.27.2.linux-amd64.tar.gz
echo "ecbadb99091a3f46e31f5f934b068b1864eafa7995211b39eaddf76996045fe5  /tmp/go.tgz" | sha256sum -c -
mkdir -p ~/.local && tar -C ~/.local -xzf /tmp/go.tgz && rm /tmp/go.tgz
echo 'export PATH="$HOME/.local/go/bin:$HOME/gopath/bin:$PATH"' >> ~/.profile

# Project virtualenv (the system Python has no ensurepip, hence --without-pip)
python3 -m venv --without-pip .venv
curl -sL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py -q && rm /tmp/get-pip.py
.venv/bin/pip install -r python/requirements.txt

# C toolchain (needs sudo once)
sudo apt install -y gcc libc6-dev
```

### 4.2 Shared inputs

```bash
.venv/bin/python python/generate_inputs.py          # writes data/inputs/*.bin
```

### 4.3 Correctness gates (must pass before any benchmark)

```bash
mvn -q verify                                       # Java: compiles + JUnit green
bash c/build.sh                                     # C: zero warnings
./c/correctness_test data/reference                 # C vs NumPy reference
cd go && gofmt -l . && go vet ./... && go test ./... && cd ..
.venv/bin/python -m compileall python
.venv/bin/python python/correctness_check.py --reference-dir data/reference
```

### 4.4 Campaign

```bash
.venv/bin/python python/run_experiments.py          # fresh CSVs, ~12 min
# options: --sizes 10,25  --languages python,java  --append  --warmup 3 --repetitions 7
```

The orchestrator builds every runner once, executes each (size, language)
configuration with the shared CLI, logs everything to
`data/results/run_log.txt`, and prints a per-configuration row-count summary.

### 4.5 Figures

```bash
.venv/bin/python python/plot_results.py             # → report/figures/*.pdf
```

The figures are reproducible from `time_raw.csv` and `memory_raw.csv` alone.

---

## 5. Observed campaign results (sanity reference)

Median time per multiplication (ms), 7 measured repetitions after 3 warm-ups:

| n | C | Go | Java | Python |
|---:|---:|---:|---:|---:|
| 10 | 0.0009 | 0.0027 | 0.058 | 0.078 |
| 50 | 0.158 | 0.421 | 1.26 | 8.73 |
| 100 | 1.47 | 3.31 | 1.95 | 75.3 |
| 200 | 13.8 | 25.6 | 17.0 | 672 |
| 500 | 238 | 405 | 350 | 13 711 |
| 1000 | 5 672 | 6 414 | 17 539 | budget exceeded |

- **Python \(n = 1000\) exceeded the 60 s budget** (probe multiplication took
  178.6 s); the run exited with code 3 and no time or memory row was recorded —
  the largest completed Python size is \(n = 500\).
- Growth from \(n = 500\) to \(n = 1000\) is steeper than the pure \(n^3\)
  arithmetic law (×24 for C instead of ×8): at that point the three matrices
  (24 MB) exceed the 12 MB L3 cache, so memory traffic dominates.
- All four checksums agree on every size (≤ 1 ulp at \(n = 1000\)).

---

## 6. Deliverables

- `Assignment01_Lorenzo_Report.pdf` — 4–6 page report (LaTeX, English,
  compiled on Overleaf from `report/main.tex`).
- `Assignment01_Lorenzo_Materials.zip` — source code, README, raw CSVs
  and scripts from this folder.

Both files are attached to the `assignment-01` release of the global
repository (`aythamilorenzo/bigdata-individual-assigments`); the assignment is
marked with the annotated tag `assignment-01`.
