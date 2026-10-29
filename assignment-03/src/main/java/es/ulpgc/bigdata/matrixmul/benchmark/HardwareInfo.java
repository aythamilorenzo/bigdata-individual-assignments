package es.ulpgc.bigdata.matrixmul.benchmark;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Hardware and affinity facts of the executing JVM, recorded once per
 * configuration so every CSV-producing run documents the execution contexts
 * it was measured on (assignment requirement: physical cores, logical
 * processors and affinity settings).
 *
 * <p>Sources: {@code /proc/cpuinfo} (model, cores, threads per core),
 * {@code /proc/self/status} ({@code Cpus_allowed_list} — the affinity mask
 * the process may run on) and {@link Runtime#availableProcessors()} (what the
 * JVM will use for its own sizing decisions).
 */
public final class HardwareInfo {

    private static final Path CPUINFO = Path.of("/proc", "cpuinfo");
    private static final Path SELF_STATUS = Path.of("/proc", "self", "status");

    private HardwareInfo() {
    }

    /** Key-value description in declaration order; missing facts are omitted. */
    public static Map<String, String> describe() {
        Map<String, String> facts = new LinkedHashMap<>();
        List<String> cpuInfo = readLines(CPUINFO);
        put(facts, "cpu_model", firstValue(cpuInfo, "model name"));
        put(facts, "logical_processors_cpuinfo", countMatches(cpuInfo, "processor"));
        put(facts, "physical_cores", distinctPairs(cpuInfo, "physical id", "core id"));
        put(facts, "threads_per_core", threadsPerCore(cpuInfo));
        put(facts, "affinity_allowed_list", firstValue(readLines(SELF_STATUS), "Cpus_allowed_list"));
        put(facts, "available_processors_jvm", Runtime.getRuntime().availableProcessors());
        return facts;
    }

    /** One-line rendering of {@link #describe()} for logs and reports. */
    public static String summary() {
        StringBuilder line = new StringBuilder("hardware:");
        describe().forEach((key, value) -> line.append(' ').append(key).append('=').append(value));
        return line.toString();
    }

    private static List<String> readLines(Path path) {
        try {
            return Files.readAllLines(path);
        } catch (IOException exception) {
            return List.of();
        }
    }

    private static void put(Map<String, String> facts, String key, String value) {
        if (value != null && !value.isBlank()) {
            facts.put(key, value);
        }
    }

    private static void put(Map<String, String> facts, String key, int value) {
        if (value > 0) {
            facts.put(key, Integer.toString(value));
        }
    }

    /** Value of the first {@code field :} line, tolerating spaces or tabs before the colon. */
    private static String firstValue(List<String> lines, String field) {
        for (String line : lines) {
            String value = valueOf(line, field);
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    /** Line value after the first colon when the trimmed line starts with {@code field}. */
    private static String valueOf(String line, String field) {
        String trimmed = line.trim();
        int colon = trimmed.indexOf(':');
        if (colon < 0 || !trimmed.substring(0, colon).trim().equals(field)) {
            return null;
        }
        return trimmed.substring(colon + 1).trim();
    }

    private static int countMatches(List<String> lines, String field) {
        int count = 0;
        for (String line : lines) {
            if (valueOf(line, field) != null) {
                count++;
            }
        }
        return count;
    }

    /** Distinct (physical id, core id) pairs — one per physical core. */
    private static int distinctPairs(List<String> lines, String socketField, String coreField) {
        java.util.Set<String> pairs = new java.util.HashSet<>();
        String socket = null;
        String core = null;
        for (String line : lines) {
            String socketValue = valueOf(line, socketField);
            String coreValue = valueOf(line, coreField);
            if (socketValue != null) {
                socket = socketValue;
            } else if (coreValue != null) {
                core = coreValue;
            } else if (line.trim().isEmpty() && socket != null && core != null) {
                pairs.add(socket + ":" + core);
                socket = null;
                core = null;
            }
        }
        if (socket != null && core != null) {
            pairs.add(socket + ":" + core);
        }
        return pairs.size();
    }

    /** Logical processors divided by physical cores, when both are known. */
    private static String threadsPerCore(List<String> lines) {
        int logical = countMatches(lines, "processor");
        int cores = distinctPairs(lines, "physical id", "core id");
        if (logical <= 0 || cores <= 0) {
            return null;
        }
        return String.format(Locale.ROOT, "%.1f", logical / (double) cores);
    }
}
