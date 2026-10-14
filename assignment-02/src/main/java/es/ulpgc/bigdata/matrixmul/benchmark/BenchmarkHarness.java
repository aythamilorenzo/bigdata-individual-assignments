package es.ulpgc.bigdata.matrixmul.benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import es.ulpgc.bigdata.matrixmul.DenseMatrix;
import es.ulpgc.bigdata.matrixmul.MatrixMultiplier;

/**
 * Kernel-only timing harness shared by the Java runner.
 *
 * <p>The timer surrounds only {@link MatrixMultiplier#multiplyInto}; inputs and
 * the output buffer exist before the timed region. A probe multiplication after
 * warm-up bookkeeping decides the time budget and whether small sizes are
 * measured as batches. Warm-up samples are recorded with {@code warmup=true}.
 */
public final class BenchmarkHarness {

    public static final long BUDGET_NS = 60_000_000_000L;
    public static final long BATCH_THRESHOLD_NS = 2_000_000L;
    public static final int MAX_BATCH = 100_000;

    private final MatrixMultiplier multiplier;

    public BenchmarkHarness(MatrixMultiplier multiplier) {
        this.multiplier = multiplier;
    }

    /** Result of one benchmark run: every raw sample plus the chosen batch size. */
    public record Measurement(List<TimingSample> samples, int batchSize) {

        public Measurement {
            samples = Collections.unmodifiableList(new ArrayList<>(samples));
        }
    }

    public Measurement measure(DenseMatrix a, DenseMatrix b, DenseMatrix out,
                               int warmups, int repetitions) {
        long probeNs = timeSingle(a, b, out);
        if (probeNs > BUDGET_NS) {
            throw new BudgetExceededException(probeNs);
        }
        int batchSize = chooseBatchSize(probeNs);

        List<TimingSample> samples = new ArrayList<>();
        for (int repetition = 1; repetition <= warmups; repetition++) {
            samples.add(sample(repetition, true, a, b, out, batchSize));
        }
        for (int repetition = 1; repetition <= repetitions; repetition++) {
            samples.add(sample(repetition, false, a, b, out, batchSize));
        }
        return new Measurement(samples, batchSize);
    }

    private TimingSample sample(int repetition, boolean warmup,
                                DenseMatrix a, DenseMatrix b, DenseMatrix out, int batchSize) {
        long start = System.nanoTime();
        for (int copy = 0; copy < batchSize; copy++) {
            multiplier.multiplyInto(a, b, out);
        }
        long elapsedNs = System.nanoTime() - start;
        double perMultiplyNs = elapsedNs / (double) batchSize;
        return new TimingSample(repetition, warmup, batchSize, perMultiplyNs, out.checksum());
    }

    private long timeSingle(DenseMatrix a, DenseMatrix b, DenseMatrix out) {
        long start = System.nanoTime();
        multiplier.multiplyInto(a, b, out);
        return System.nanoTime() - start;
    }

    static int chooseBatchSize(long probeNs) {
        if (probeNs >= BATCH_THRESHOLD_NS) {
            return 1;
        }
        long batch = (BATCH_THRESHOLD_NS + probeNs - 1) / probeNs;
        return (int) Math.min(MAX_BATCH, Math.max(1L, batch));
    }
}
