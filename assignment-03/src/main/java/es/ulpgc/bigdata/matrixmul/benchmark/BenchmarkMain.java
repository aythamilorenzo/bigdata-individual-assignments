package es.ulpgc.bigdata.matrixmul.benchmark;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.InputMatrixLoader;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;

/**
 * Java entry point of the assignment 3 benchmark CLI.
 *
 * <pre>
 * java -jar matrixmul-parallel.jar --method NAME --threads P --inputs DIR
 *      --size N --time-csv FILE --memory-csv FILE [--warmup W] [--repetitions R]
 * </pre>
 *
 * The sequential family ({@code blocked_ijk_sequential}) runs on the calling
 * thread and always reports {@code threads = 1}; the parallel family
 * ({@code blocked_ijk_parallel}) distributes the same kernel across a pool of
 * {@code P} prestarted threads and reports {@code threads = P}. Both multiply
 * the same shared input files under the same kernel-only timing boundary.
 *
 * <p>Exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded,
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
            System.err.println("invalid arguments; expected --method --threads --inputs "
                    + "--size --time-csv --memory-csv [--warmup] [--repetitions]");
            System.exit(EXIT_INVALID_ARGUMENTS);
            return;
        }

        Path pathA = args.inputs().resolve("matrixA_" + args.size() + ".bin");
        Path pathB = args.inputs().resolve("matrixB_" + args.size() + ".bin");
        if (!Files.isRegularFile(pathA) || !Files.isRegularFile(pathB)) {
            System.err.printf(Locale.ROOT, "missing input files for n=%d%n", args.size());
            System.exit(EXIT_MISSING_INPUT);
            return;
        }

        System.out.println(HardwareInfo.summary());

        InputMatrixLoader loader = new InputMatrixLoader();
        DenseMatrix matrixA = loader.load(pathA, args.size());
        DenseMatrix matrixB = loader.load(pathB, args.size());
        DenseMatrix out = DenseMatrix.zeros(args.size());

        int reportedThreads = Multipliers.isParallel(args.method()) ? args.threads() : 1;
        MatrixMultiplier multiplier = Multipliers.create(args.method(), args.threads());
        try {
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
            writer.writeTimeSamples(args.timeCsv(), args.size(), reportedThreads,
                    args.method(), measurement.samples());
            writer.writeMemoryRows(args.memoryCsv(), args.size(), reportedThreads,
                    args.method(), PeakResidentMemory.bytes(), PeakHeap.usedBytes());
            System.out.printf(Locale.ROOT,
                    "java n=%d threads=%d method=%s batch=%d checksum=%.17g%n",
                    args.size(), reportedThreads, args.method(),
                    measurement.batchSize(), out.checksum());
        } finally {
            Multipliers.shutdown(multiplier);
        }
        System.exit(EXIT_OK);
    }

    private static void reportBudgetExceeded(Arguments args, BudgetExceededException exceeded) {
        System.err.printf(Locale.ROOT, "budget exceeded at n=%d: %.1f s%n",
                args.size(), exceeded.probeNs() / 1e9);
        System.exit(EXIT_BUDGET_EXCEEDED);
    }

    private record Arguments(Path inputs, int size, String method, int threads,
                             Path timeCsv, Path memoryCsv,
                             int warmup, int repetitions) {

        static Arguments parse(String[] arguments) {
            String inputs = null;
            String size = null;
            String method = null;
            String threads = null;
            String timeCsv = null;
            String memoryCsv = null;
            int warmup = 3;
            int repetitions = 7;
            for (int i = 0; i + 1 < arguments.length; i += 2) {
                switch (arguments[i]) {
                    case "--inputs" -> inputs = arguments[i + 1];
                    case "--size" -> size = arguments[i + 1];
                    case "--method" -> method = arguments[i + 1];
                    case "--threads" -> threads = arguments[i + 1];
                    case "--time-csv" -> timeCsv = arguments[i + 1];
                    case "--memory-csv" -> memoryCsv = arguments[i + 1];
                    case "--warmup" -> warmup = Integer.parseInt(arguments[i + 1]);
                    case "--repetitions" -> repetitions = Integer.parseInt(arguments[i + 1]);
                    default -> {
                        return null;
                    }
                }
            }
            if (inputs == null || size == null || method == null || threads == null
                    || timeCsv == null || memoryCsv == null) {
                return null;
            }
            return new Arguments(Path.of(inputs), Integer.parseInt(size), method,
                    Integer.parseInt(threads), Path.of(timeCsv), Path.of(memoryCsv),
                    warmup, repetitions);
        }
    }
}
