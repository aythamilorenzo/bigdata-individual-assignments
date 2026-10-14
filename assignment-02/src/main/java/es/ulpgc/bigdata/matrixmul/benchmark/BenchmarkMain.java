package es.ulpgc.bigdata.matrixmul.benchmark;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.InputMatrixLoader;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;
import es.ulpgc.bigdata.matrixmul.sparse.CsrMatrix;
import es.ulpgc.bigdata.matrixmul.sparse.DenseToCsrConverter;
import es.ulpgc.bigdata.matrixmul.sparse.GustavsonMultiplier;
import es.ulpgc.bigdata.matrixmul.sparse.SparseMatrixMultiplier;

/**
 * Java entry point of the assignment 2 benchmark CLI.
 *
 * <pre>
 * java -jar matrixmul-optimised.jar --method NAME --inputs DIR --size N
 *      --time-csv FILE --memory-csv FILE [--density D]
 *      [--warmup W] [--repetitions R]
 * </pre>
 *
 * Without {@code --density} the run is a Part A dense configuration: the
 * shared dense inputs are multiplied by the named dense method and rows go to
 * {@code dense_time_raw.csv}. With {@code --density D} the run belongs to the
 * Part B sparse campaign on the declared-density inputs
 * ({@code matrixA_nN_dD.bin}): {@code sparse_csr_gustavson} measures the CSR
 * kernel (with the dense-to-CSR conversion timed separately), while a dense
 * method measures the matched dense comparison on the same logical matrices;
 * both write to {@code sparse_time_raw.csv} labelled with the representation.
 *
 * Exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded,
 * 4 missing input files.
 */
public final class BenchmarkMain {

    private static final int EXIT_OK = 0;
    private static final int EXIT_INVALID_ARGUMENTS = 2;
    private static final int EXIT_BUDGET_EXCEEDED = 3;
    private static final int EXIT_MISSING_INPUT = 4;

    private BenchmarkMain() {
    }

    public static void main(String[] arguments) {
        Arguments args = Arguments.parse(arguments);
        if (args == null) {
            System.err.println("invalid arguments; expected --method --inputs --size "
                    + "--time-csv --memory-csv [--density] [--warmup] [--repetitions]");
            System.exit(EXIT_INVALID_ARGUMENTS);
            return;
        }

        Path pathA = args.inputs().resolve(inputName("A", args.size(), args.density()));
        Path pathB = args.inputs().resolve(inputName("B", args.size(), args.density()));
        if (!Files.isRegularFile(pathA) || !Files.isRegularFile(pathB)) {
            System.err.printf(Locale.ROOT, "missing input files for n=%d density=%s%n",
                    args.size(), args.density() == null ? "-" : args.density());
            System.exit(EXIT_MISSING_INPUT);
            return;
        }

        InputMatrixLoader loader = new InputMatrixLoader();
        DenseMatrix matrixA = loader.load(pathA, args.size());
        DenseMatrix matrixB = loader.load(pathB, args.size());

        if (args.density() == null) {
            runDensePartA(args, matrixA, matrixB);
        } else if (Multipliers.isDense(args.method())) {
            runDenseComparisonOnSparseInputs(args, matrixA, matrixB);
        } else {
            runSparseKernel(args, matrixA, matrixB);
        }
        System.exit(EXIT_OK);
    }

    private static void runDensePartA(Arguments args, DenseMatrix matrixA, DenseMatrix matrixB) {
        DenseMatrix out = DenseMatrix.zeros(args.size());
        MatrixMultiplier multiplier = Multipliers.dense(args.method());

        PeakHeap.reset();
        BenchmarkHarness harness = new BenchmarkHarness(multiplier);
        BenchmarkHarness.Measurement measurement;
        try {
            measurement = harness.measure(matrixA, matrixB, out,
                    args.warmup(), args.repetitions());
        } catch (BudgetExceededException exceeded) {
            reportBudgetExceeded(args, exceeded);
            return;
        }

        ResultsCsvWriter writer = new ResultsCsvWriter();
        writer.writeDenseTimeSamples(args.timeCsv(), args.size(), args.method(),
                measurement.samples());
        writer.writeDenseMemoryRows(args.memoryCsv(), args.size(), args.method(),
                PeakResidentMemory.bytes(), PeakHeap.usedBytes());
        System.out.printf(Locale.ROOT, "java dense n=%d method=%s batch=%d checksum=%.17g%n",
                args.size(), args.method(), measurement.batchSize(), out.checksum());
    }

    private static void runDenseComparisonOnSparseInputs(Arguments args,
                                                         DenseMatrix matrixA, DenseMatrix matrixB) {
        DenseMatrix out = DenseMatrix.zeros(args.size());
        MatrixMultiplier multiplier = Multipliers.dense(args.method());

        PeakHeap.reset();
        BenchmarkHarness harness = new BenchmarkHarness(multiplier);
        BenchmarkHarness.Measurement measurement;
        try {
            measurement = harness.measure(matrixA, matrixB, out,
                    args.warmup(), args.repetitions());
        } catch (BudgetExceededException exceeded) {
            reportBudgetExceeded(args, exceeded);
            return;
        }

        ResultsCsvWriter writer = new ResultsCsvWriter();
        writer.writeSparseTimeSamples(args.sparseTimeCsv(), args.size(), args.density(),
                ResultsCsvWriter.REPRESENTATION_DENSE, args.method(),
                countNonZeros(matrixA), countNonZeros(matrixB), countNonZeros(out),
                null, measurement.samples());
        writer.writeComparisonMemoryRows(args.memoryCsv(), args.size(), args.density(),
                args.method(), ResultsCsvWriter.REPRESENTATION_DENSE,
                countNonZeros(matrixA), countNonZeros(matrixB), countNonZeros(out),
                PeakResidentMemory.bytes(), PeakHeap.usedBytes());
        System.out.printf(Locale.ROOT,
                "java dense n=%d density=%s method=%s batch=%d checksum=%.17g%n",
                args.size(), args.density(), args.method(), measurement.batchSize(), out.checksum());
    }

    private static void runSparseKernel(Arguments args, DenseMatrix matrixA, DenseMatrix matrixB) {
        DenseToCsrConverter converter = new DenseToCsrConverter();
        long conversionStart = System.nanoTime();
        CsrMatrix csrA = converter.convert(matrixA);
        CsrMatrix csrB = converter.convert(matrixB);
        long conversionNs = System.nanoTime() - conversionStart;

        SparseMatrixMultiplier multiplier = new GustavsonMultiplier(args.size());
        PeakHeap.reset();
        SparseBenchmarkHarness harness = new SparseBenchmarkHarness(multiplier);
        SparseBenchmarkHarness.Measurement measurement;
        try {
            measurement = harness.measure(csrA, csrB, args.warmup(), args.repetitions());
        } catch (BudgetExceededException exceeded) {
            reportBudgetExceeded(args, exceeded);
            return;
        }

        CsrMatrix product = measurement.result();
        ResultsCsvWriter writer = new ResultsCsvWriter();
        writer.writeSparseTimeSamples(args.sparseTimeCsv(), args.size(), args.density(),
                ResultsCsvWriter.REPRESENTATION_CSR, args.method(),
                csrA.nonZeros(), csrB.nonZeros(), product.nonZeros(),
                conversionNs, measurement.samples());
        writer.writeComparisonMemoryRows(args.memoryCsv(), args.size(), args.density(),
                args.method(), ResultsCsvWriter.REPRESENTATION_CSR,
                csrA.nonZeros(), csrB.nonZeros(), product.nonZeros(),
                PeakResidentMemory.bytes(), PeakHeap.usedBytes());
        System.out.printf(Locale.ROOT,
                "java sparse n=%d density=%s method=%s batch=%d nnz=%d checksum=%.17g%n",
                args.size(), args.density(), args.method(), measurement.batchSize(),
                product.nonZeros(), product.checksum());
    }

    private static void reportBudgetExceeded(Arguments args, BudgetExceededException exceeded) {
        System.err.printf(Locale.ROOT, "budget exceeded at n=%d density=%s: %.1f s%n",
                args.size(), args.density() == null ? "-" : args.density(),
                exceeded.probeNs() / 1e9);
        System.exit(EXIT_BUDGET_EXCEEDED);
    }

    private static String inputName(String side, int size, String density) {
        return density == null
                ? "matrix" + side + "_" + size + ".bin"
                : "matrix" + side + "_n" + size + "_d" + density + ".bin";
    }

    private static long countNonZeros(DenseMatrix matrix) {
        int n = matrix.size();
        long count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix.get(i, j) != 0.0) {
                    count++;
                }
            }
        }
        return count;
    }

    private record Arguments(Path inputs, int size, String method, String density,
                             Path timeCsv, Path sparseTimeCsv, Path memoryCsv,
                             int warmup, int repetitions) {

        static Arguments parse(String[] arguments) {
            String inputs = null;
            String size = null;
            String method = null;
            String density = null;
            String timeCsv = null;
            String sparseTimeCsv = null;
            String memoryCsv = null;
            int warmup = 3;
            int repetitions = 7;
            for (int i = 0; i + 1 < arguments.length; i += 2) {
                switch (arguments[i]) {
                    case "--inputs" -> inputs = arguments[i + 1];
                    case "--size" -> size = arguments[i + 1];
                    case "--method" -> method = arguments[i + 1];
                    case "--density" -> density = arguments[i + 1];
                    case "--time-csv" -> timeCsv = arguments[i + 1];
                    case "--sparse-time-csv" -> sparseTimeCsv = arguments[i + 1];
                    case "--memory-csv" -> memoryCsv = arguments[i + 1];
                    case "--warmup" -> warmup = Integer.parseInt(arguments[i + 1]);
                    case "--repetitions" -> repetitions = Integer.parseInt(arguments[i + 1]);
                    default -> {
                        return null;
                    }
                }
            }
            if (inputs == null || size == null || method == null || memoryCsv == null) {
                return null;
            }
            if (density == null && timeCsv == null) {
                return null;
            }
            if (density != null && sparseTimeCsv == null) {
                return null;
            }
            return new Arguments(Path.of(inputs), Integer.parseInt(size), method, density,
                    timeCsv == null ? null : Path.of(timeCsv),
                    sparseTimeCsv == null ? null : Path.of(sparseTimeCsv),
                    Path.of(memoryCsv), warmup, repetitions);
        }
    }
}
