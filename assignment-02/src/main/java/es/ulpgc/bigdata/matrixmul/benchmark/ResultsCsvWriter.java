package es.ulpgc.bigdata.matrixmul.benchmark;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

/**
 * Appends raw benchmark rows to the campaign CSV files of assignment 2.
 *
 * <p>Three schemas, each defined once here and shared with the Python
 * orchestrator and plotting scripts:
 *
 * <ul>
 *   <li>{@code dense_time_raw.csv} — Part A, dense methods on the shared
 *       dense inputs.</li>
 *   <li>{@code sparse_time_raw.csv} — Part B, every configuration labelled
 *       with density, representation, nonzero counts and the separately
 *       measured dense-to-CSR conversion cost.</li>
 *   <li>{@code memory_raw.csv} — both parts; the density column is empty for
 *       Part A rows.</li>
 * </ul>
 */
public final class ResultsCsvWriter {

    public static final String NUMERIC_TYPE = "float64";
    public static final String REPRESENTATION_DENSE = "dense";
    public static final String REPRESENTATION_CSR = "csr";

    private static final String DENSE_TIME_HEADER =
            "size,method,numeric_type,repetition,warmup,batch_size,"
                    + "time_ns_per_multiply,checksum";
    private static final String SPARSE_TIME_HEADER =
            "size,density,representation,method,numeric_type,repetition,warmup,"
                    + "batch_size,nnz_a,nnz_b,nnz_c,conversion_time_ns,"
                    + "kernel_time_ns_per_multiply,checksum";
    private static final String MEMORY_HEADER =
            "size,density,method,metric,tool,value_bytes,theoretical_bytes";

    public void writeDenseTimeSamples(Path path, int size, String method,
                                      List<TimingSample> samples) {
        StringBuilder block = new StringBuilder();
        if (needsHeader(path)) {
            block.append(DENSE_TIME_HEADER).append(System.lineSeparator());
        }
        for (TimingSample sample : samples) {
            block.append(String.format(Locale.ROOT,
                    "%d,%s,%s,%d,%s,%d,%.3f,%.17g%n",
                    size, method, NUMERIC_TYPE,
                    sample.repetition(), sample.warmup(), sample.batchSize(),
                    sample.timeNsPerMultiply(), sample.checksum()));
        }
        append(path, block.toString());
    }

    public void writeSparseTimeSamples(Path path, int size, String density,
                                       String representation, String method,
                                       long nnzA, long nnzB, long nnzC,
                                       Long conversionTimeNs,
                                       List<TimingSample> samples) {
        StringBuilder block = new StringBuilder();
        if (needsHeader(path)) {
            block.append(SPARSE_TIME_HEADER).append(System.lineSeparator());
        }
        String conversion = conversionTimeNs == null ? "" : conversionTimeNs.toString();
        for (TimingSample sample : samples) {
            block.append(String.format(Locale.ROOT,
                    "%d,%s,%s,%s,%s,%d,%s,%d,%d,%d,%d,%s,%.3f,%.17g%n",
                    size, density, representation, method, NUMERIC_TYPE,
                    sample.repetition(), sample.warmup(), sample.batchSize(),
                    nnzA, nnzB, nnzC, conversion,
                    sample.timeNsPerMultiply(), sample.checksum()));
        }
        append(path, block.toString());
    }

    public void writeDenseMemoryRows(Path path, int size, String method,
                                     long peakRssBytes, long peakHeapBytes) {
        long theoretical = 24L * size * size;
        StringBuilder block = new StringBuilder();
        if (needsHeader(path)) {
            block.append(MEMORY_HEADER).append(System.lineSeparator());
        }
        block.append(memoryRow(size, "", method, "peak_rss", "/proc/self/status VmHWM",
                peakRssBytes, theoretical));
        block.append(memoryRow(size, "", method, "peak_heap", "MemoryPoolMXBean",
                peakHeapBytes, theoretical));
        append(path, block.toString());
    }

    public void writeComparisonMemoryRows(Path path, int size, String density,
                                               String method, String representation,
                                               long nnzA, long nnzB, long nnzC,
                                               long peakRssBytes, long peakHeapBytes) {
        long theoretical = ResultsCsvWriter.REPRESENTATION_CSR.equals(representation)
                ? 12L * (nnzA + nnzB + nnzC) + 12L * (size + 1)
                : 24L * size * size;
        StringBuilder block = new StringBuilder();
        if (needsHeader(path)) {
            block.append(MEMORY_HEADER).append(System.lineSeparator());
        }
        block.append(memoryRow(size, density, method, "peak_rss", "/proc/self/status VmHWM",
                peakRssBytes, theoretical));
        block.append(memoryRow(size, density, method, "peak_heap", "MemoryPoolMXBean",
                peakHeapBytes, theoretical));
        append(path, block.toString());
    }

    private static String memoryRow(int size, String density, String method, String metric,
                                    String tool, long valueBytes, long theoreticalBytes) {
        return String.format(Locale.ROOT, "%d,%s,%s,%s,%s,%d,%d%n",
                size, density, method, metric, tool, valueBytes, theoreticalBytes);
    }

    private static boolean needsHeader(Path path) {
        return !Files.exists(path) || sizeOf(path) == 0L;
    }

    private static long sizeOf(Path path) {
        try {
            return Files.size(path);
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    private static void append(Path path, String block) {
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, block, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException exception) {
            throw new UncheckedIOException("cannot append to " + path, exception);
        }
    }
}
