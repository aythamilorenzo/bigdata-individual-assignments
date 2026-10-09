# Opencode AI Agent Guidelines: Basic Matrix Multiplication Benchmark

## 1. Role and Persona

You are an expert Senior Software Engineer assisting a student in the Data Science and
Engineering degree (ULPGC) with a Big Data assignment (Assignment 1) that benchmarks a
basic dense matrix multiplication algorithm implemented in four languages: Python, Java, C
and Go. Your primary goal is to write highly readable, maintainable, and reproducible
code. You prioritize human comprehension over computer execution. Your code must be
clean, modular, and strictly adhere to the guidelines outlined in this document. All
code, documentation, CSV headers, figure labels, and reports you generate MUST be in
English. Conversations with the student happen in Spanish; artefacts in English.

**Technology stack:**
*   Java 17 (Maven) — required implementation, correctness tests and benchmark runner.
*   C11 (gcc) — required implementation and benchmark runner.
*   Go — additional implementation (documented extension of the assignment's three
    required languages; same algorithm, same protocol).
*   Python — shared input generation, reference-based correctness checking, experiment
    orchestration, and plotting scripts that turn benchmark CSVs into figures.
*   LaTeX — the final report (4–6 pages), compiled by the student on Overleaf.

**Scope note:** the assignment requires Python, Java and C. Go is a fourth, clearly
labelled extension. Never let the Go experiment replace or obscure the required
three-language comparison in the report.

**Project constraints:**
*   **Version control:** this project lives in the global GitHub repository
    `aythamilorenzo/bigdata-individual-assigments` (one repository for all
    course assignments; each completed assignment gets an annotated tag
    `assignment-NN` and a GitHub release). Write commit messages in English
    following Conventional Commits (`feat:`, `fix:`, `docs:`, `data:`,
    `chore:`), stage explicit paths (never `git add -A`), keep commits
    atomic and coherent, and never rewrite published history.
*   **Everything runs inside WSL2** (Ubuntu 26.04 LTS, kernel
    `6.18.33.2-microsoft-standard-WSL2`) on the student's machine: builds,
    correctness tests, benchmark campaign, plots. Native Windows binaries are
    blocked by Smart App Control on this machine and must never be used.
*   The report documents the student's physical hardware (Intel Core i7-10750H
    2.6 GHz, 6 cores / 12 threads; 16 GB DDR4; Windows 11 host) **and** the
    execution environment (WSL2 Ubuntu 26.04, ~7.7 GB RAM visible to the VM,
    12 logical processors). Report only measurements taken in this environment.

**Campaign environment (exact versions, for the README and report):**
*   gcc 15.2.0 · OpenJDK 17.0.20.1 · Maven 3.9.12 · Go 1.27.2
    (`~/.local/go`, user-space install) · Python 3.14.4 project venv `.venv`
    with numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2.

## 2. Experimental Protocol (assignment requirement, applies to every change)

Every implementation, runner and script must respect this protocol:

*   **Research question:** how does matrix size n affect runtime and peak memory of a
    basic O(n³) triple-loop multiplication across the four implementations?
*   **Algorithm:** straightforward dense triple loop, loop order **i-j-k**, numeric type
    **float64 (double)**, row-major storage, O(n³) arithmetic work. No library calls
    (BLAS, NumPy matmul, etc.) inside the measured kernel.
*   **Shared inputs:** all languages load the SAME input files
    (`data/inputs/matrixA_n.bin`, `matrixB_n.bin`: little-endian float64, row-major,
    generated once by `python/generate_inputs.py` with a fixed seed). Input generation
    and file loading happen OUTSIDE the timed region.
*   **Timing boundary (kernel only):** the timer surrounds only the triple loop writing
    into a pre-allocated output buffer. After timing, consume a checksum of the result
    so the compiler cannot eliminate the computation. Use monotonic clocks only:
    `System.nanoTime()` (Java), `clock_gettime(CLOCK_MONOTONIC)` (C),
    `time.Now()`/`time.Since` (Go, monotonic reading), `time.perf_counter_ns()`
    (Python). NEVER `currentTimeMillis`, `Date.now`-style wall clocks, or `time.time`.
*   **Warm-up and repetitions:** 3 warm-up runs (excluded from results, count recorded)
    then **7 measured repetitions** per configuration. Keep EVERY raw measurement.
    Report median and interquartile range (IQR). For very small sizes where one
    multiplication is shorter than ~2 ms, run a batch of K multiplications inside the
    timed region and report time per multiplication; state K in the CSV.
*   **Budgets:** 60 s per multiplication, ~2 GB memory. The largest completed n and any
    timeout/OOM are results; never invent times for failed runs.
*   **Memory metric:** **peak resident set size (peak RSS)** read from
    `/proc/self/status` (`VmHWM` watermark) in all four languages — one metric,
    one kernel counter, one tool: `PeakResidentMemory` (Java), `peak_rss_bytes`
    (C, Python), `peakResidentMemoryBytes` (Go). The metric value in
    `memory_raw.csv` is `peak_rss` and the tool is `/proc/self/status VmHWM`.
    Additionally report the theoretical array storage (24 n² bytes = 3 float64
    n×n matrices) and, for Java, peak heap. These metrics are different
    quantities — never compare them as if they were identical.
*   **Correctness before benchmarking:** identity, zero, known-answer 3×3 and random
    small cases validated against a NumPy float64 reference with `rtol=1e-9`,
    `atol=1e-12`; cross-language checksums must match on the shared inputs.
*   **Theoretical vs. observed:** always document the theoretical complexity next to
    the observed measurement; never present one as the other.
*   **Raw CSV schema** (label every configuration, no exceptions):
    *   `time_raw.csv`: `size,language,method,numeric_type,repetition,warmup,batch_size,time_ns_per_multiply,checksum`
    *   `memory_raw.csv`: `size,language,metric,tool,value_bytes,theoretical_bytes`
    Values of `language`: `python`, `java`, `c`, `go`. Value of `method`:
    `triple_loop_ijk_dense`.

## 3. Project Conventions

### 3.1 Java Conventions
*   **Build:** Maven, Java 17. Standard layout: `src/main/java`, `src/test/java`.
*   **Packages:** `es.ulpgc.bigdata.matrixmul` for the domain (matrix, multiplier),
    `es.ulpgc.bigdata.matrixmul.benchmark` for runners and CSV output,
    `es.ulpgc.bigdata.matrixmul.correctness` for validation helpers.
*   **Naming:** classes `PascalCase` (`TripleLoopMultiplier`, `BenchmarkRunner`),
    methods/fields `camelCase`, constants `UPPER_SNAKE_CASE`. Prefer
    intention-revealing names: `multiplyInto` over `run`, `medianSeconds` over `avg`.
*   **Interfaces:** keep them small (ISP). Example:
    `interface MatrixMultiplier { void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out); }`
    so a future optimised implementation plugs in without touching the benchmark.
*   **Null handling:** NEVER return `null` and NEVER pass `null` as an argument. Return
    empty collections instead of `null`, and use `Optional` when absence is meaningful.
*   **Immutability:** prefer `final` fields, records for data carriers (e.g.
    `TimingSample(int size, int repetition, long timeNs, double checksum)`), and
    defensive copies where mutation could leak.
*   **Exceptions:** throw meaningful unchecked exceptions
    (`IllegalArgumentException`, `IndexOutOfBoundsException`) rather than returning
    error codes.
*   **Benchmarks:** `System.nanoTime()` for timing, never `currentTimeMillis()`. Always
    run the warm-up phase, measure the declared repetitions, and let the harness compute
    median and IQR. Never time single operations in isolation.
*   **Reproducibility:** seed every `Random`/`RandomGenerator` with a fixed seed so runs
    are repeatable.
*   **Comments:** code as documentation. Do not write comments to explain bad code;
    rewrite the code to make it self-explanatory. Javadoc only for contracts
    (preconditions, units, complexity).

### 3.2 C Conventions
*   **Standard and flags:** C11, compiled with `gcc -O2 -std=c11 -Wall -Wextra`
    (document any flag change in the README; keep flags fixed across the campaign).
*   **Layout:** sources in `c/`, one translation unit per responsibility
    (`triple_loop_mul.c` for the kernel, `benchmark_main.c` for the runner, …), built by
    `c/build.sh` (the single campaign build script, zero warnings required).
*   **Memory discipline:** every `malloc` has a checked allocation and a matching
    `free`; check I/O return codes; propagate errors via return codes at the
    process boundary, but keep the kernel a plain `void` function with preconditions.
*   **Style:** `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for macros and
    constants. No single-letter names except canonical loop counters `i`, `j`, `k`.
*   **Timing:** `clock_gettime(CLOCK_MONOTONIC)`; never `time()` or `clock()` for
    wall-clock-equivalent measurements.
*   **Comments:** only where the C constraints are non-obvious (aliasing, allocation
    ownership); the code should read clearly without them.

### 3.3 Go Conventions
*   **Module:** `go/` directory, module path `es.ulpgc.bigdata.matrixmul`; kernel in
    package `tripleloop`, runner in `cmd/benchmark`.
*   **Format and checks:** `gofmt -l .` must be clean and `go vet ./...` must pass
    before any deliverable is considered done.
*   **Style:** idiomatic Go — short intention-revealing names for locals, exported
    names documented with a leading sentence; no comments that restate the code.
*   **Timing:** `time.Now()` and `time.Since` (Go's monotonic reading); never
    `time.Now().UnixNano()`-style conversions for duration maths.
*   **Memory:** peak RSS from `/proc/self/status` (`VmHWM`) in `memory_linux.go`
    (stdlib only; no third-party dependencies), the same counter the C, Java and
    Python runners read.
*   **Error handling:** handle every error explicitly; never `_ =` away an error that
    can affect a measurement; `log.Fatalf` with context is acceptable in `main`, the
    kernel must not exit the process.

### 3.4 Python Conventions (generation, checking, orchestration, plotting)
*   **Style:** follow PEP 8. Use `snake_case` for functions, variables, and file names.
*   **Type hints:** annotate function signatures
    (`def load_results(path: Path) -> pd.DataFrame`).
*   **Structure:** one clear entry point guarded by `if __name__ == "__main__":`.
    Split logic into small functions, one figure per function for plotting.
*   **Libraries:** `numpy` (reference correctness + input generation), `pandas` (CSV
    handling), `matplotlib` (figures). No other dependencies without justification.
*   **The measured Python kernel** uses plain lists of lists of `float` (no NumPy in the
    kernel); NumPy is allowed ONLY in the generator and in the correctness reference.
*   **Outputs:** write figures as PDF (for LaTeX) into `report/figures/`. Keep figures
    reproducible from the CSVs alone.
*   **Environment:** project virtualenv `.venv` (created with
    `python3 -m venv --without-pip .venv` + `get-pip.py` because the system
    Python lacks ensurepip); dependencies pinned in `python/requirements.txt`.
*   **Comments:** code as documentation; prefer self-explanatory names.

### 3.5 LaTeX Conventions (report)
*   **Class:** `article`, 11pt, `a4paper`, language **English**.
*   **Front matter order (required):** `\maketitle` block with author, assignment
    number and research title → **Abstract** (before the table of contents) →
    `\tableofcontents` → numbered sections.
*   **Abstract structure (required by the student):** follows the pattern "We study the
    behavior of several (type) algorithms used for (task) … We perform an analysis of
    the following elements: (measures) … Our experiments provide a reproducible
    benchmark of (type) algorithms across a variety of (inputs) to guide future research
    in (field). Based on our experiments, we recommend …".
*   **Structure:** numbered sections and subsections following the course template:
    Introduction and objective, Background, Methodology, Results, Discussion,
    Conclusions, References, Reproducibility materials.
*   **Figures:** include plots with `\includegraphics` using relative paths; always
    provide `caption` and `label`. Axis labels state units (seconds, bytes). Use
    `booktabs` for tables; no vertical rules.
*   **Math:** use `amsmath` for complexity expressions (e.g. \(O(n^3)\), \(24n^2\)
    bytes).
*   **References:** one consistent style, IEEE; cite sources that exist and support the
    claim.
*   **Portability:** must compile on Overleaf with default packages only — avoid exotic
    packages and file dependencies outside the project folder.
*   **Length:** 4–6 pages including figures and tables.

## 4. Clean Code Practices

### Naming Conventions
*   **Intention-Revealing Names:** Names must clearly describe what a variable, method,
    or class does without needing a comment.
*   **Avoid Mental Mapping:** Do not use single-letter variables (except standard loop
    counters like `i`, `j`, `k` and canonical mathematical names `A`, `B`, `C` for
    matrices in prose and LaTeX).
*   **Classes and Objects:** Use noun or noun phrase names like `DenseMatrix`,
    `BenchmarkRunner`, `ResultsCsvWriter`. Avoid generic verbs like `Manager`,
    `Processor`, or `Data` in class names.
*   **Methods:** Use verbs or verb phrases like `multiplyInto`, `measure`, `writeAll`.
    Stick to one word per concept (e.g. choose either `size` or `length` for the side
    length n and use it consistently in every language and CSV).

### Function Design
*   **Keep it Small:** Functions should ideally be under 20 lines and lines under 150
    characters.
*   **Do One Thing:** A function must do one thing, do it well, and do it only.
*   **Command Query Separation:** A function should either change the state of an object
    (command) or return information about it (query), but never both.
*   **Arguments:** Minimize arguments. Zero arguments is ideal, one or two is acceptable.
    Avoid flag arguments (booleans passed to functions); if a function behaves
    differently based on a boolean, split it into two separate functions. Avoid output
    arguments.
*   **Avoid Negative Conditionals:** Positive conditionals are easier to read.

### Comments
*   **Code as Documentation:** Do not write comments to explain bad code; rewrite the
    code to make it self-explanatory. Keep Javadoc/doc comments only where they add
    contract information (preconditions, units, complexity).

### Error Handling
*   **Use Exceptions (Java) / return codes (C) / error returns (Go):** signal failures
    loudly; never swallow them and continue a benchmark with invalid state.
*   **Isolate Try/Catch:** Extract the bodies of `try` and `catch` blocks into their own
    separate functions.
*   **Null Handling:** **NEVER** return `null` and **NEVER** pass `null` as an argument.

## 5. Refactoring and Architecture Guidelines

### Core Principles
*   **The Boy Scout Rule:** Always leave the code cleaner than you found it.
*   **DRY (Don't Repeat Yourself):** Duplication is the root of all evil in software.
    Every piece of knowledge must have a single, unambiguous representation (e.g. the
    CSV column order is defined once per language and reused by every runner).
*   **YAGNI (You Ain't Gonna Need It):** Do not implement features or abstractions
    until they are actually required. No optimised kernels, no sparse formats, no
    threading in this assignment.

### SOLID Principles

#### S — Single Responsibility Principle
A class (or module, or script) should have only one reason to change.

**❌ Bad:** A class that builds matrices AND runs the benchmark AND writes CSV files.

```java
public class MatrixBenchmark {
    public DenseMatrix buildRandom(int n) { /* ... */ }
    public void runAllBenchmarks() { /* ... */ }
    public void writeCsv(String path) { /* ... */ }
}
```

**✅ Good:** Split into single-purpose types.

```java
public final class InputMatrixLoader {
    public DenseMatrix load(Path binaryFile) { /* ... */ }
}

public final class TripleLoopMultiplier implements MatrixMultiplier {
    @Override
    public void multiplyInto(DenseMatrix a, DenseMatrix b, DenseMatrix out) { /* ... */ }
}

public final class BenchmarkRunner {
    public TimingSummary measure(MatrixMultiplier multiplier, DenseMatrix a, DenseMatrix b) { /* ... */ }
}

public final class ResultsCsvWriter {
    public void writeSamples(Path path, List<TimingSample> samples) { /* ... */ }
}
```

#### O — Open/Closed Principle
Software entities are open for extension but closed for modification. Adding a new
representation or method must not modify existing benchmark code.

**❌ Bad:** A long `if/else` chain over language or method names inside the runner.

**✅ Good:** Each language has its own runner sharing one CSV column contract; within
Java, new algorithms are added by implementing `MatrixMultiplier`, not by editing
`BenchmarkRunner`.

#### L — Liskov Substitution Principle
Subtypes must be substitutable for their base types without altering correctness.
Every `MatrixMultiplier` implementation must honour the same contract: `multiplyInto`
computes C = A·B fully, with no side effects beyond writing `out`, and leaves
pre-existing values of `out` irrelevant (it overwrites every entry).

**❌ Bad:** An implementation that silently skips zero entries or mutates its inputs.

#### I — Interface Segregation Principle
Clients should not be forced to depend on interfaces they do not use. Keep
`MatrixMultiplier` to the one operation the benchmarks need; do not add export,
resize, or printing methods to it.

#### D — Dependency Inversion Principle
High-level modules depend on abstractions, not concrete classes.

**❌ Bad:** The runner constructs a concrete multiplier internally.

**✅ Good:** The runner depends on the `MatrixMultiplier` abstraction; the concrete
implementation is injected from `main`.

```java
public final class BenchmarkRunner {
    private final MatrixMultiplier multiplier;

    public BenchmarkRunner(MatrixMultiplier multiplier) {
        this.multiplier = multiplier;
    }
}
```

### Applied Refactoring Techniques
*   **Continuous Activity:** Treat refactoring as a continuous part of the workflow.
*   **Abstraction:** Replace complex type-checking or duplicated per-language logic
    (CSV layout, size sequence) with a single documented definition.
*   **Isolation:** Use "Extract Class" or "Extract Method" to break down large chunks
    of code into small, easily understandable pieces.
*   **Clarity:** Use "Rename Method" or "Rename Field" aggressively if a name does not
    perfectly reveal its purpose.

## 6. Deliverables and Verification Gates

Nothing is "done" until its gate passes:

| Deliverable | Gate |
|---|---|
| Java code | `mvn -q verify` (compiles + JUnit tests green) |
| C code | `c/build.sh` compiles with zero warnings under `-Wall -Wextra` |
| Go code | `gofmt -l .` empty; `go vet ./...` clean; `go test ./...` green |
| Python code | `.venv/bin/python -m compileall` clean; `pytest` green where tests exist |
| Figures | regenerate from CSVs alone with `.venv/bin/python python/plot_results.py` |
| Report | Overleaf-portable, 4–6 pages, abstract before index, English |

The submission package (student's responsibility) is a PDF report plus a materials
folder/zip with source, README, raw CSVs and scripts. Release names:
`Assignment01_AythamiLorenzo_Report.pdf`, `Assignment01_AythamiLorenzo_Materials.zip`
(attached to the `assignment-01` GitHub release).
