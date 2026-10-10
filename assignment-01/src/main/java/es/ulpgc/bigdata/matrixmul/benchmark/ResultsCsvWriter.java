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
 * Appends raw benchmark rows to the campaign CSV files.
 *
 * <p>The column order below is the contract shared with the Python, C and Go
 * runners and documented in the README; it is defined once here.
 */
public final class ResultsCsvWriter {

    public static final String LANGUAGE = "java";
    public static final String METHOD = "triple_loop_ijk_dense";
    public static final String NUMERIC_TYPE = "float64";

    private static final String TIME_HEADER =
            "size,language,method,numeric_type,repetition,warmup,batch_size,"
                    + "time_ns_per_multiply,checksum";
    private static final String MEMORY_HEADER =
            "size,language,metric,tool,value_bytes,theoretical_bytes";

    public void writeTimeSamples(Path path, int size, List<TimingSample> samples) {
        StringBuilder block = new StringBuilder();
        if (needsHeader(path)) {
            block.append(TIME_HEADER).append(System.lineSeparator());
        }
        for (TimingSample sample : samples) {
            block.append(String.format(Locale.ROOT,
                    "%d,%s,%s,%s,%d,%s,%d,%.3f,%.17g%n",
                    size, LANGUAGE, METHOD, NUMERIC_TYPE,
                    sample.repetition(), sample.warmup(), sample.batchSize(),
                    sample.timeNsPerMultiply(), sample.checksum()));
        }
        append(path, block.toString());
    }

    public void writeMemoryRows(Path path, int size, long peakRssBytes,
                                long peakHeapBytes) {
        long theoretical = 24L * size * size;
        StringBuilder block = new StringBuilder();
        if (needsHeader(path)) {
            block.append(MEMORY_HEADER).append(System.lineSeparator());
        }
        block.append(memoryRow(size, "peak_rss", "/proc/self/status VmHWM",
                peakRssBytes, theoretical));
        block.append(memoryRow(size, "peak_heap", "MemoryPoolMXBean",
                peakHeapBytes, theoretical));
        block.append(memoryRow(size, "theoretical_array_storage", "analytical",
                theoretical, theoretical));
        append(path, block.toString());
    }

    private static String memoryRow(int size, String metric, String tool,
                                    long valueBytes, long theoreticalBytes) {
        return String.format(Locale.ROOT, "%d,%s,%s,%s,%d,%d%n",
                size, LANGUAGE, metric, tool, valueBytes, theoreticalBytes);
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
