package es.ulpgc.bigdata.matrixmul.benchmark;

/**
 * One raw timing observation from the harness.
 *
 * @param repetition         1-based warm-up or measured repetition number
 * @param warmup             true for warm-up samples (excluded from analysis)
 * @param batchSize          multiplications executed inside the timed region
 * @param timeNsPerMultiply  elapsed nanoseconds divided by {@code batchSize}
 * @param checksum           deterministic row-major checksum of the result
 */
public record TimingSample(
        int repetition,
        boolean warmup,
        int batchSize,
        double timeNsPerMultiply,
        double checksum) {
}
