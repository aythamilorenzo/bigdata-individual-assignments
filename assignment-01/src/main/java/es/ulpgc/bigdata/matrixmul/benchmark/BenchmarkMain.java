package es.ulpgc.bigdata.matrixmul.benchmark;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.InputMatrixLoader;
import es.ulpgc.bigdata.matrixmul.TripleLoopMultiplier;

/**
 * Java entry point of the shared benchmark CLI.
 *
 * <pre>
 * java -jar matrixmul-benchmark.jar --inputs DIR --size N
 *      --time-csv FILE --memory-csv FILE --warmup W --repetitions R
 * </pre>
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
            System.exit(EXIT_INVALID_ARGUMENTS);
            return;
        }

        InputMatrixLoader loader = new InputMatrixLoader();
        Path pathA = args.inputs().resolve("matrixA_" + args.size() + ".bin");
        Path pathB = args.inputs().resolve("matrixB_" + args.size() + ".bin");
        if (!Files.isRegularFile(pathA) || !Files.isRegularFile(pathB)) {
            System.err.printf(Locale.ROOT, "missing input files for n=%d%n", args.size());
            System.exit(EXIT_MISSING_INPUT);
            return;
        }
        DenseMatrix matrixA = loader.load(pathA, args.size());
        DenseMatrix matrixB = loader.load(pathB, args.size());
        DenseMatrix out = DenseMatrix.zeros(args.size());

        PeakHeap.reset();
        BenchmarkHarness harness = new BenchmarkHarness(new TripleLoopMultiplier());
        BenchmarkHarness.Measurement measurement;
        try {
            measurement = harness.measure(matrixA, matrixB, out,
                    args.warmup(), args.repetitions());
        } catch (BudgetExceededException exceeded) {
            System.err.printf(Locale.ROOT, "budget exceeded at n=%d: %.1f s%n",
                    args.size(), exceeded.probeNs() / 1e9);
            System.exit(EXIT_BUDGET_EXCEEDED);
            return;
        }

        ResultsCsvWriter writer = new ResultsCsvWriter();
        writer.writeTimeSamples(args.timeCsv(), args.size(), measurement.samples());
        writer.writeMemoryRows(args.memoryCsv(), args.size(),
                PeakResidentMemory.bytes(), PeakHeap.usedBytes());
        System.out.printf(Locale.ROOT, "java n=%d batch=%d checksum=%.17g%n",
                args.size(), measurement.batchSize(), out.checksum());
    }

    private record Arguments(Path inputs, int size, Path timeCsv, Path memoryCsv,
                             int warmup, int repetitions) {

        static Arguments parse(String[] arguments) {
            String inputs = null;
            String size = null;
            String timeCsv = null;
            String memoryCsv = null;
            int warmup = 3;
            int repetitions = 7;
            for (int i = 0; i + 1 < arguments.length; i += 2) {
                switch (arguments[i]) {
                    case "--inputs" -> inputs = arguments[i + 1];
                    case "--size" -> size = arguments[i + 1];
                    case "--time-csv" -> timeCsv = arguments[i + 1];
                    case "--memory-csv" -> memoryCsv = arguments[i + 1];
                    case "--warmup" -> warmup = Integer.parseInt(arguments[i + 1]);
                    case "--repetitions" -> repetitions = Integer.parseInt(arguments[i + 1]);
                    default -> {
                        return null;
                    }
                }
            }
            if (inputs == null || size == null || timeCsv == null || memoryCsv == null) {
                return null;
            }
            return new Arguments(Path.of(inputs), Integer.parseInt(size),
                    Path.of(timeCsv), Path.of(memoryCsv), warmup, repetitions);
        }
    }
}
