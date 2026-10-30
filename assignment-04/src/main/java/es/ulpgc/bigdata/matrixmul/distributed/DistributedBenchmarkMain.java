package es.ulpgc.bigdata.matrixmul.distributed;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.InputMatrixLoader;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;
import es.ulpgc.bigdata.matrixmul.benchmark.HardwareInfo;
import es.ulpgc.bigdata.matrixmul.benchmark.Multipliers;
import es.ulpgc.bigdata.matrixmul.benchmark.PeakHeap;
import es.ulpgc.bigdata.matrixmul.benchmark.PeakResidentMemory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/**
 * Per-configuration benchmark CLI for assignment 4. One invocation measures
 * one (method, threads, size) configuration and appends its rows to the raw
 * campaign CSVs.
 *
 * <pre>
 * java -jar matrixmul-distributed.jar --method NAME --threads P --inputs DIR
 *      --size N --time-csv FILE --memory-csv FILE [--warmup W] [--repetitions R]
 * </pre>
 *
 * <p>Three method families share the same validated kernel:
 * {@code blocked_ijk_sequential} (p = 1), {@code blocked_ijk_parallel}
 * (thread pool, assignment 3) and {@code blocked_ijk_distributed} (Spark
 * local[P]).
 *
 * <p>Timing boundary: the end-to-end timer surrounds one multiplication call
 * (for the distributed family: broadcast of B, the Spark job and the
 * driver-side assembly). The setup cost (SparkContext creation) is timed
 * separately and reported in its own column; input loading is outside every
 * timer. Every row carries the phase decomposition; for the sequential and
 * thread-pool families the whole cost sits in the compute column.
 *
 * <p>Exit codes: 0 ok, 2 invalid arguments, 3 time budget exceeded,
 * 4 missing input files.
 */
public final class DistributedBenchmarkMain {

    private static final int EXIT_OK = 0;
    private static final int EXIT_INVALID_ARGUMENTS = 2;
    private static final int EXIT_BUDGET_EXCEEDED = 3;
    private static final int EXIT_MISSING_INPUT = 4;

    /** 60 s per multiplication; larger results are recorded as budget exceedance. */
    private static final long BUDGET_NS = 60_000_000_000L;

    private static final String TIME_HEADER =
            "size,threads,method,numeric_type,repetition,warmup,batch_size,"
                    + "end_to_end_ns,broadcast_ns,job_wall_ns,assemble_ns,task_compute_ns,checksum";

    private static final String MEMORY_HEADER =
            "size,threads,method,metric,tool,value_bytes,theoretical_bytes";

    private DistributedBenchmarkMain() {
    }

    public static void main(String[] arguments) throws Exception {
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

        boolean distributed = Multipliers.METHOD_BLOCKED_DISTRIBUTED.equals(args.method());
        PeakHeap.reset();

        JavaSparkContext sc = null;
        MatrixMultiplier multiplier = null;
        DistributedMultiplier distributedMultiplier = null;
        long setupNs = 0L;
        try {
            long setupStart = System.nanoTime();
            if (distributed) {
                SparkConf conf = new SparkConf()
                        .setAppName("Assignment4-" + args.method() + "-n" + args.size())
                        .setMaster("local[" + args.threads() + "]")
                        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                        .set("spark.kryoserializer.buffer.max", "256m");
                sc = new JavaSparkContext(conf);
                distributedMultiplier = new DistributedMultiplier(sc, args.threads());
            } else {
                multiplier = Multipliers.create(args.method(), args.threads());
            }
            setupNs = System.nanoTime() - setupStart;

            long budgetProbeNs = probeOneMultiplication(matrixA, matrixB, multiplier, distributedMultiplier);
            if (budgetProbeNs > BUDGET_NS) {
                System.err.printf(Locale.ROOT, "budget exceeded at n=%d method=%s threads=%d: %.1f s%n",
                        args.size(), args.method(), args.threads(), budgetProbeNs / 1e9);
                System.exit(EXIT_BUDGET_EXCEEDED);
                return;
            }

            runMeasurements(args, matrixA, matrixB, multiplier, distributedMultiplier,
                    distributed, setupNs);
            writeMemoryRows(args);
            System.out.printf(Locale.ROOT,
                    "java n=%d threads=%d method=%s setup_ns=%d checksum_recorded=true%n",
                    args.size(), args.threads(), args.method(), setupNs);
            System.exit(EXIT_OK);
        } finally {
            if (sc != null) {
                sc.stop();
            }
            if (multiplier != null) {
                Multipliers.shutdown(multiplier);
            }
        }
    }

    /** One throwaway multiplication to check the 60 s per-multiplication budget. */
    private static long probeOneMultiplication(DenseMatrix a, DenseMatrix b,
                                              MatrixMultiplier multiplier,
                                              DistributedMultiplier distributedMultiplier) {
        long start = System.nanoTime();
        if (distributedMultiplier != null) {
            distributedMultiplier.multiply(a, b);
        } else {
            multiplier.multiplyInto(a, b, DenseMatrix.zeros(a.size()));
        }
        return System.nanoTime() - start;
    }

    private static void runMeasurements(Arguments args, DenseMatrix a, DenseMatrix b,
                                        MatrixMultiplier multiplier,
                                        DistributedMultiplier distributedMultiplier,
                                        boolean distributed, long setupNs) throws IOException {
        int totalRuns = args.warmup() + args.repetitions();
        for (int run = 0; run < totalRuns; run++) {
            boolean warmup = run < args.warmup();
            double checksum;

            long endToEndNs;
            long broadcastNs = 0L;
            long jobWallNs = 0L;
            long assembleNs = 0L;
            long taskComputeNs = 0L;

            if (distributed) {
                DistributedMultiplier.MeasuredMultiplication measured =
                        distributedMultiplier.measure(a, b);
                DistributedMultiplier.PhaseTimings timings = measured.timings();
                endToEndNs = timings.endToEndNs();
                broadcastNs = timings.broadcastNs();
                jobWallNs = timings.jobWallNs();
                assembleNs = timings.assembleNs();
                taskComputeNs = timings.taskComputeNs();
                checksum = measured.result().checksum();
            } else {
                DenseMatrix out = DenseMatrix.zeros(a.size());
                long start = System.nanoTime();
                multiplier.multiplyInto(a, b, out);
                endToEndNs = System.nanoTime() - start;
                jobWallNs = endToEndNs;
                taskComputeNs = endToEndNs;
                checksum = out.checksum();
            }

            appendTimeRow(args.timeCsv(), args.size(), args.threads(), args.method(),
                    run - args.warmup(), warmup, endToEndNs, broadcastNs, jobWallNs,
                    assembleNs, taskComputeNs, checksum);
            System.out.printf(Locale.ROOT,
                    "java n=%d threads=%d method=%s rep=%d warmup=%s end_to_end_ns=%d checksum=%.17g%n",
                    args.size(), args.threads(), args.method(), run - args.warmup(),
                    warmup, endToEndNs, checksum);
        }
    }

    private static void appendTimeRow(Path csv, int size, int threads, String method,
                                      int repetition, boolean warmup, long endToEndNs,
                                      long broadcastNs, long jobWallNs, long assembleNs,
                                      long taskComputeNs, double checksum) throws IOException {
        boolean writeHeader = !Files.exists(csv) || Files.size(csv) == 0;
        try (BufferedWriter writer = Files.newBufferedWriter(csv,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (writeHeader) {
                writer.write(TIME_HEADER);
                writer.newLine();
            }
            writer.write(String.format(Locale.ROOT,
                    "%d,%d,%s,float64,%d,%s,1,%d,%d,%d,%d,%d,%.17g",
                    size, threads, method, repetition, warmup,
                    endToEndNs, broadcastNs, jobWallNs, assembleNs, taskComputeNs, checksum));
            writer.newLine();
        }
    }

    private static void writeMemoryRows(Arguments args) throws IOException {
        long theoretical = 24L * args.size() * args.size();
        appendMemoryRow(args.memoryCsv(), args.size(), args.threads(), args.method(),
                "peak_rss", "proc_self_status_VmHWM", PeakResidentMemory.bytes(), theoretical);
        appendMemoryRow(args.memoryCsv(), args.size(), args.threads(), args.method(),
                "peak_heap", "MemoryPoolMXBean", PeakHeap.usedBytes(), theoretical);
    }

    private static void appendMemoryRow(Path csv, int size, int threads, String method,
                                        String metric, String tool, long valueBytes,
                                        long theoreticalBytes) throws IOException {
        boolean writeHeader = !Files.exists(csv) || Files.size(csv) == 0;
        try (BufferedWriter writer = Files.newBufferedWriter(csv,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (writeHeader) {
                writer.write(MEMORY_HEADER);
                writer.newLine();
            }
            writer.write(String.format(Locale.ROOT,
                    "%d,%d,%s,%s,%s,%d,%d",
                    size, threads, method, metric, tool, valueBytes, theoreticalBytes));
            writer.newLine();
        }
    }

    private record Arguments(String method, int threads, Path inputs, int size,
                             Path timeCsv, Path memoryCsv, int warmup, int repetitions) {

        static Arguments parse(String[] arguments) {
            String method = null;
            String threads = null;
            String inputs = null;
            String size = null;
            String timeCsv = null;
            String memoryCsv = null;
            int warmup = 3;
            int repetitions = 7;
            for (int i = 0; i + 1 < arguments.length; i += 2) {
                switch (arguments[i]) {
                    case "--method" -> method = arguments[i + 1];
                    case "--threads" -> threads = arguments[i + 1];
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
            if (method == null || threads == null || inputs == null || size == null
                    || timeCsv == null || memoryCsv == null) {
                return null;
            }
            return new Arguments(method, Integer.parseInt(threads), Path.of(inputs),
                    Integer.parseInt(size), Path.of(timeCsv), Path.of(memoryCsv),
                    warmup, repetitions);
        }
    }
}
