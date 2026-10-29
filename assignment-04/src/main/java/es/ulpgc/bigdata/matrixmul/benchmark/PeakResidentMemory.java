package es.ulpgc.bigdata.matrixmul.benchmark;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Peak resident set size of the current process from {@code /proc/self/status}
 * (the {@code VmHWM} watermark).
 *
 * <p>The C, Go and Python runners read the very same kernel counter, so the
 * four memory figures share one metric and one tool. This is a physical-memory
 * quantity and must not be compared with the theoretical array storage or the
 * Java peak heap as if they were identical.
 */
public final class PeakResidentMemory {

    private static final Path PROC_STATUS = Path.of("/proc", "self", "status");
    private static final String PEAK_RSS_FIELD = "VmHWM:";

    private PeakResidentMemory() {
    }

    /** Return the peak RSS of this process in bytes. */
    public static long bytes() {
        try {
            List<String> lines = Files.readAllLines(PROC_STATUS);
            for (String line : lines) {
                if (line.startsWith(PEAK_RSS_FIELD)) {
                    return parseKibibytes(line);
                }
            }
        } catch (IOException exception) {
            throw new IllegalStateException("cannot read " + PROC_STATUS, exception);
        }
        throw new IllegalStateException(PEAK_RSS_FIELD + " not found in " + PROC_STATUS);
    }

    private static long parseKibibytes(String line) {
        String value = line.substring(line.indexOf(':') + 1).trim().split("\\s+")[0];
        return Long.parseLong(value) * 1024L;
    }
}
